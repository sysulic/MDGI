'''
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
'''

import torch
from torch.nn import functional as F
from nltk.corpus import wordnet as wn
import os
import sys
import time
import math
import copy
import argparse
from tqdm import tqdm
import pickle
import csv
from pytorch_transformers import *

import random
import numpy as np
import ast
import xlrd
from models.util import *
from models.models import JointBiEncoderModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

parser = argparse.ArgumentParser(description='Gloss Informed Bi-encoder for WSD')

#training arguments
parser.add_argument('--rand_seed', type=int, default=42)
parser.add_argument('--grad-norm', type=float, default=1.0)
parser.add_argument('--silent', action='store_true',
    help='Flag to supress training progress bar for each epoch')
parser.add_argument('--multigpu', action='store_true')
parser.add_argument('--lr', type=float, default=0.00002)
parser.add_argument('--warmup', type=int, default=10000)
parser.add_argument('--context-max-length', type=int, default=128)
parser.add_argument('--gloss-max-length', type=int, default=32)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--context-bsz', type=int, default=1)
parser.add_argument('--gloss-bsz', type=int, default=8)
parser.add_argument('--encoder-name', type=str, default='bert-base',
    choices=['bert-base', 'bert-large', 'roberta-base', 'roberta-large', 'bert_base_Chinese'])
parser.add_argument('--ckpt', type=str, required=True,
    help='filepath at which to save best probing model (on dev set)')
parser.add_argument('--data-path', type=str, required=True,
    help='Location of top-level directory for the Unified WSD Framework')

parser.add_argument('--train-m', type=int, required=True,
    help='train metaphor')
parser.add_argument('--train-w', type=int, required=True,
    help='train wsd')

#sets which parts of the model to freeze ❄️ during training for ablation 
parser.add_argument('--freeze-context', action='store_true')
parser.add_argument('--freeze-gloss', action='store_true')
parser.add_argument('--tie-encoders', action='store_true')

#other training settings flags
parser.add_argument('--kshot', type=int, default=-1,
    help='if set to k (1+), will filter training data to only have up to k examples per sense')
parser.add_argument('--balanced', action='store_true',
    help='flag for whether or not to reweight sense losses to be balanced wrt the target word')

#evaluation arguments
parser.add_argument('--eval', action='store_true',
    help='Flag to set script to evaluate probe (rather than train)')
parser.add_argument('--split', type=str, default='semeval2007',
    choices=['semeval2007', 'senseval2', 'senseval3', 'semeval2013', 'semeval2015', 'ALL', 'all-test'],
    help='Which evaluation split on which to evaluate probe')

#uses these two gpus if training in multi-gpu
context_device = "cuda:1"
gloss_device = "cuda:0"

def tokenize_glosses(gloss_arr, tokenizer, max_len):
    glosses = []
    masks = []
    for gloss_text in gloss_arr:
        g_ids = [torch.tensor([[x]]) for x in tokenizer.encode(tokenizer.cls_token)+tokenizer.encode(gloss_text)+tokenizer.encode(tokenizer.sep_token)]
        g_attn_mask = [1]*len(g_ids)
        g_fake_mask = [-1]*len(g_ids)
        g_ids, g_attn_mask, _ = normalize_length(g_ids, g_attn_mask, g_fake_mask, max_len, pad_id=tokenizer.encode(tokenizer.pad_token)[0])
        g_ids = torch.cat(g_ids, dim=-1)
        g_attn_mask = torch.tensor(g_attn_mask)
        glosses.append(g_ids)
        masks.append(g_attn_mask)

    return glosses, masks

## Add by this comparing experiment
def read_glosses_from_file(gloss_file):
    gloss_dict = {}
    with open(gloss_file, 'r', encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            if line[0] in gloss_dict:
                gloss_dict[line[0]].append(line[1])
            else:
                gloss_dict[line[0]] = ['NULL',line[1]]
    return gloss_dict

def get_label_glosses(label_file):
    label_glosses = []
    with open(label_file, 'r') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            g_arr = ast.literal_eval(line[7])
            for g in g_arr:
                if g != 'NULL':
                    label_glosses.append(g.strip())
        return label_glosses
def update_gloss_dict(file, gloss_dict, tokenizer, max_len):

    with open(file, 'r') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            g_arr = ast.literal_eval(line[7])
            sent = line[2].split(' ')
            for w, g in zip(sent, g_arr):
                if g.strip() == 'NULL':
                    if not (w in gloss_dict):
                        gloss_dict[w] = ['NULL']
                        #preprocess glosses into tensors
                        gloss_ids, gloss_masks = tokenize_glosses(['NULL'], tokenizer, max_len)
                        gloss_ids = torch.cat(gloss_ids, dim=0)
                        gloss_masks = torch.stack(gloss_masks, dim=0)
                        gloss_dict[w] = (gloss_ids, gloss_masks, ['NULL'])
        return gloss_dict

def get_metaphor_label(label_file):
    labels = []
    marks = []
    nums = []
    with open(label_file, 'r') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            para_id = line[0]
            sent_idx = para_id + '#'+line[1]
            label_seq = ast.literal_eval(line[3])
            pos_seq = ast.literal_eval(line[4])
            assert (len(pos_seq) == len(label_seq))
            assert (len(line[2].split()) == len(pos_seq))
            sent = line[2].split(' ')
            num = [sent_idx+'#'+str(idx) for idx in range(len(sent))]
            label_int_seq = [int(l) for l in label_seq]
            labels.extend(label_int_seq)
            marks.extend(pos_seq)
            nums.extend(num)
    return labels, marks, nums

def load_and_preprocess_webster_glosses(gloss_file, tokenizer, label_glosses, max_len=-1):
    gloss_word_dict = read_glosses_from_file(gloss_file)

    sense_glosses = {}
    for word, gloss_arr in gloss_word_dict.items():
        #preprocess glosses into tensors
        gloss_ids, gloss_masks = tokenize_glosses(gloss_arr, tokenizer, max_len)
        gloss_ids = torch.cat(gloss_ids, dim=0)
        gloss_masks = torch.stack(gloss_masks, dim=0)
        sense_glosses[word] = (gloss_ids, gloss_masks, gloss_arr)

    sense_weights = {}    
    #intialize weights for balancing senses
    for sense, sense_tuple in sense_glosses.items():
        gloss_arr = sense_tuple[2]
        sense_weights[sense] = [0]*len(gloss_arr)
        for s_idx, sense_word in enumerate(gloss_arr):
            sense_weights[sense][s_idx] = label_glosses.count(sense_word)
    
    return sense_glosses, sense_weights



#creates a sense label/ gloss dictionary for training/using the gloss encoder
def load_and_preprocess_glosses(data, tokenizer, wn_senses, max_len=-1):
    sense_glosses = {}
    sense_weights = {}

    gloss_lengths = []

    for sent in data:
        for _, lemma, pos, _, label in sent:
            if label == -1:
                continue #ignore unlabeled words
            else:
                key = generate_key(lemma, pos)
                if key not in sense_glosses:
                    #get all sensekeys for the lemma/pos pair
                    sensekey_arr = wn_senses[key]
                    #get glosses for all candidate senses
                    gloss_arr = [wn.lemma_from_key(s).synset().definition() for s in sensekey_arr]

                    #preprocess glosses into tensors
                    gloss_ids, gloss_masks = tokenize_glosses(gloss_arr, tokenizer, max_len)
                    gloss_ids = torch.cat(gloss_ids, dim=0)
                    gloss_masks = torch.stack(gloss_masks, dim=0)
                    sense_glosses[key] = (gloss_ids, gloss_masks, sensekey_arr)

                    #intialize weights for balancing senses
                    sense_weights[key] = [0]*len(gloss_arr)
                    w_idx = sensekey_arr.index(label)
                    sense_weights[key][w_idx] += 1
                else:
                    #update sense weight counts
                    w_idx = sense_glosses[key][2].index(label)
                    sense_weights[key][w_idx] += 1
                
                #make sure that gold label is retrieved synset
                assert label in sense_glosses[key][2]

    #normalize weights
    for key in sense_weights:
        total_w = sum(sense_weights[key])
        sense_weights[key] = torch.FloatTensor([total_w/x if x !=0 else 0 for x in sense_weights[key]])

    return sense_glosses, sense_weights

def load_VUA_data(VUA_file):
    loaded_data = []


    with open(VUA_file, 'r') as f:
        lines = csv.reader(f)
        next(lines)
        count = 1
        for line in lines:

            sentence_detail = []

            
            label_seq = ast.literal_eval(line[3])
            pos_seq = ast.literal_eval(line[4])

            sent = line[2].split(' ')
            gloss_labels = ast.literal_eval(line[7])


            assert (len(line[2].split()) == len(pos_seq))
            assert (len(line[2].split()) == len(label_seq))
            
            for w_idx, w in enumerate(sent):
                if gloss_labels[w_idx] != 'NULL':
                    sentence_detail.append((w, gloss_labels[w_idx].strip(), int(label_seq[w_idx])))
                else:
                    sentence_detail.append((w, -1, int(label_seq[w_idx])))
            loaded_data.append(sentence_detail)

    return loaded_data

def compute_acc(true_label, pred_label):
    sizes = len(true_label)
    corr = sum([1 for pt, pp in zip(true_label, pred_label) if pt == pp])
    return corr / sizes
def preprocess_context(tokenizer, text_data, bsz=1, max_len=-1):
    if max_len == -1: assert bsz==1 #otherwise need max_length for padding

    context_ids = []
    context_attn_masks = []

    example_keys = []

    context_output_masks = []
    instances = []
    labels = []
    metaphor_labels = []
    metaphor_output_masks = []
    gloss_key_arr = []

    length_arr = []

    #tensorize data
    for sent in text_data:
        c_ids = [torch.tensor([tokenizer.encode(tokenizer.cls_token)])] #cls token aka sos token, returns a list with index
        o_masks = [-1]
        sent_labels = []
        sent_keys = []
        sent_m_labels = []
        m_o_masks = [-1]
        m_m_o_masks = [-1]
        m_l_o_masks = [-1]
        gloss_key = []


        #For each word in sentence...
        for idx, (word, label, m_label) in enumerate(sent):
            #tensorize word for context ids
            word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word.lower())]
            c_ids.extend(word_ids)

            #if word is labeled with WSD sense...
            if label != -1:
                #add word to bert output mask to be labeled
                o_masks.extend([idx]*len(word_ids))

                sent_labels.append(label)
                gloss_key.append(word)
            else:
                #mask out output of context encoder for WSD task (not labeled)
                o_masks.extend([-1]*len(word_ids))
            sent_keys.append(word)
            

            sent_m_labels.append(m_label)
            m_o_masks.extend([idx]*len(word_ids))


            #break if we reach max len
            if max_len != -1 and len(c_ids) >= (max_len-1):
                break

        c_ids.append(torch.tensor([tokenizer.encode(tokenizer.sep_token)])) #aka eos token
        c_attn_mask = [1]*len(c_ids)
        m_o_masks.append(-1)
        o_masks.append(-1)
        c_ids, c_attn_masks, m_o_masks, o_masks = normalize_length_for_context(c_ids, c_attn_mask, m_o_masks, o_masks, max_len, pad_id=tokenizer.encode(tokenizer.pad_token)[0])



        if len(sent_labels) > 0 or len(sent_m_labels) > 0:
            context_ids.append(torch.cat(c_ids, dim=-1))
            context_attn_masks.append(torch.tensor(c_attn_masks).unsqueeze(dim=0))
            context_output_masks.append(torch.tensor(o_masks).unsqueeze(dim=0))
            example_keys.append(sent_keys)
            labels.append(sent_labels)
            metaphor_labels.append(sent_m_labels)
            metaphor_output_masks.append(torch.tensor(m_o_masks).unsqueeze(dim=0))
            gloss_key_arr.append(gloss_key)


    #package data
    data = list(zip(context_ids, context_attn_masks, context_output_masks, example_keys, labels, metaphor_labels, metaphor_output_masks, gloss_key_arr))

    #batch data if bsz > 1
    if bsz > 1:
        print('Batching data with bsz={}...'.format(bsz))
        batched_data = []
        for idx in range(0, len(data), bsz):
            if idx+bsz <=len(data): b = data[idx:idx+bsz]
            else: b = data[idx:]
            context_ids = torch.cat([x for x,_,_,_,_,_,_,_ in b], dim=0)
            context_attn_mask = torch.cat([x for _,x,_,_,_,_,_,_ in b], dim=0)
            context_output_mask = []
            for _,_,x,_,_,_,_,_ in b: context_output_mask.extend(x)
            example_keys = []
            for _,_,_,x,_,_,_,_ in b: example_keys.extend(x)
            labels = []
            for _,_,_,_,x,_,_,_ in b: labels.extend(x)
            m_labels_arr = []
            for _,_,_,_,_,x,_,_ in b: m_labels_arr.extend(x)
            metaphor_output_mask = torch.cat([x for _,_,_,_,_,_,x,_ in b], dim=0)
            g_key_arr = []
            for _,_,_,_,_,_,_,x in b: g_key_arr.extend(x)
            

            batched_data.append((context_ids, context_attn_mask, context_output_mask, example_keys, labels, m_labels_arr, metaphor_output_mask, g_key_arr))
        return batched_data
    else:  
        return data

def _train(train_data, model, gloss_dict, optim, schedule, criterion, metaphor_criterion, gloss_bsz=-1, max_grad_norm=1.0, multigpu=False, silent=False, train_steps=-1, _train_m=True, _train_w=False):
    model.train()
    total_loss = 0.

    start_time = time.time()

    train_data = enumerate(train_data)
    if not silent: train_data = tqdm(list(train_data))
    total_gloss_size = 0

    for i, (context_ids, context_attn_mask, context_output_mask, example_keys, labels, m_labels_arr, metaphor_output_mask, gloss_key_arr) in train_data:

        gloss_sz = 0
        wsd_loss = 0.
        #reset model
        model.zero_grad()
        #run example sentence(s) through context encoder
        if multigpu:
            context_ids = context_ids.to(context_device)
            context_attn_mask = context_attn_mask.to(context_device)
        else:
            context_ids = context_ids.cuda()
            context_attn_mask = context_attn_mask.cuda()
        
        

        t = sum([1 for output_mask in context_output_mask if torch.max(output_mask).item() == -1])

        if t != len(context_output_mask) and _train_w:
                
            context_output = model.context_forward(context_ids, context_attn_mask, context_output_mask)


            wsd_loss = 0.
            gloss_sz = 0
            

            context_sz = len(labels)

            for j, (key, label) in enumerate(zip(gloss_key_arr, labels)):
                output = context_output.split(1,dim=0)[j]


                #run example's glosses through gloss encoder
                gloss_ids, gloss_attn_mask, sense_keys = gloss_dict[key]
                if multigpu:
                    gloss_ids = gloss_ids.to(gloss_device)
                    gloss_attn_mask = gloss_attn_mask.to(gloss_device)
                else:
                    gloss_ids = gloss_ids.cuda()
                    gloss_attn_mask = gloss_attn_mask.cuda()
                gloss_output = model.gloss_forward(gloss_ids, gloss_attn_mask)
                gloss_output = gloss_output.transpose(0,1)
            
                #get cosine sim of example from context encoder with gloss embeddings
                if multigpu:
                    output = output.cpu()
                    gloss_output = gloss_output.cpu()
                
                output = torch.mm(output, gloss_output)

                #get label and calculate loss
                idx = sense_keys.index(label)
                label_tensor = torch.tensor([idx])
                if not multigpu: 
                    label_tensor = label_tensor.cuda()
                else:
                    label_tensor = label_tensor.to(gloss_device)



                #looks up correct candidate senses criterion
                #needed if balancing classes within the candidate senses of a target word
                wsd_loss += criterion[key](output, label_tensor)

                gloss_sz += gloss_output.size(-1)

                if gloss_bsz != -1 and gloss_sz >= gloss_bsz:
                    #update model
                    

                    wsd_loss=wsd_loss/gloss_sz
                    total_loss += (wsd_loss.item())

                    loss = wsd_loss

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optim.step()
                    schedule.step() # Update learning rate schedule

                    #reset loss and gloss_sz
                    wsd_loss = 0.
                    gloss_sz = 0

                    #reset model
                    model.zero_grad()

                    #rerun context through model
                    context_output = model.context_forward(context_ids, context_attn_mask, context_output_mask)

            #stop epoch early if number of training steps is reached
            if train_steps > 0 and i+1 == train_steps: break

                    #update model after finishing context batch
            if gloss_bsz != -1: loss_sz = gloss_sz
            else: loss_sz = context_sz
            
            if loss_sz > 0:
                
                wsd_loss=wsd_loss/loss_sz
                loss = wsd_loss
                total_loss += (wsd_loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optim.step()
                schedule.step() # Update learning rate schedule


        if _train_m:
            metaphor_context_output = model.metaphor_context_forward(context_ids, context_attn_mask, metaphor_output_mask, _train_w)
            # m_idx = [idx for idx, m_label in enumerate(m_labels_arr) if m_label == 1]
            # l_idx = [idx for idx, m_label in enumerate(m_labels_arr) if m_label == 0]
            # random.shuffle(l_idx)
            # selected = l_idx[:len(m_idx)]
            # if len(m_idx) == 0:
            #     selected = l_idx[:2]
            # m_idx = m_idx + selected
            # random.shuffle(m_idx)
            # m_idx = m_idx[:int(len(m_idx)/2)]
            # m_loss = 0.
            # m_size = 0
            m_loss = 0.
            m_size = 0
            for j in range(len(m_labels_arr)):
                key = example_keys[j]
                m_label = m_labels_arr[j]
                metaphor_output = metaphor_context_output.split(1,dim=0)[j]


                #run example's glosses through gloss encoder
                gloss_ids, gloss_attn_mask, sense_keys = gloss_dict[key]


                if multigpu:
                    gloss_ids = gloss_ids.to(gloss_device)
                    gloss_attn_mask = gloss_attn_mask.to(gloss_device)
                else:
                    gloss_ids = gloss_ids.cuda()
                    gloss_attn_mask = gloss_attn_mask.cuda()

                gloss_output = model.metaphor_gloss_forward(gloss_ids, gloss_attn_mask, _train_w)
                gloss_output_T = gloss_output.transpose(0,1)
                
                #get cosine sim of example from context encoder with gloss embeddings
                if multigpu:
                    metaphor_output = metaphor_output.cpu()
                    gloss_output_T = gloss_output_T.cpu()
                
                output = torch.mm(metaphor_output, gloss_output_T)

                attn_score = torch.softmax(output, dim=-1)
                context_vector = torch.mm(attn_score, gloss_output)

                if multigpu:
                    metaphor_output = metaphor_output.to(context_device)
                    context_vector = context_vector.to(context_device)
                predicted_metaphor = model.fc_forward(torch.cat((metaphor_output, context_vector), dim=-1))
                m_label_tensor = torch.tensor(m_label)
                m_label_tensor = m_label_tensor.unsqueeze(-1)
                if not multigpu: m_label_tensor = m_label_tensor.cuda()
                m_loss += metaphor_criterion(predicted_metaphor.view(-1, 2), m_label_tensor.view(-1))
                m_size += 1
                if m_size > gloss_bsz:
                    m_loss = m_loss / m_size
                    m_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optim.step()
                    schedule.step() # Update learning rate schedule
                    m_loss = 0.
                    m_size = 0
                    #reset model
                    model.zero_grad()
                    metaphor_context_output = model.metaphor_context_forward(context_ids, context_attn_mask, metaphor_output_mask, _train_w)
            if m_size != 0:
                m_loss = m_loss / m_size
                m_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optim.step()
                schedule.step() # Update learning rate schedule
                m_loss = 0.
                m_size = 0



            
    print(total_loss)

    return model, optim, schedule, total_loss

def _eval(eval_data, model, gloss_dict, multigpu=False):
    model.eval()
    eval_preds = []
    eval_m_pred = []

    for context_ids, context_attn_mask, context_output_mask, example_keys, labels, m_labels_arr, metaphor_output_mask, gloss_key in eval_data:
        with torch.no_grad(): 
            #run example through model
            if multigpu:
                context_ids = context_ids.to(context_device)
                context_attn_mask = context_attn_mask.to(context_device)
            else:
                context_ids = context_ids.cuda()
                context_attn_mask = context_attn_mask.cuda()
            

            metaphor_context_output = model.context_forward(context_ids, context_attn_mask, metaphor_output_mask)

            

            for metaphor_output, label, key in zip(metaphor_context_output.split(1,dim=0), m_labels_arr, example_keys):
                gloss_ids, gloss_attn_mask, sense_keys = gloss_dict[key]
                if multigpu:
                    gloss_ids = gloss_ids.to(gloss_device)
                    gloss_attn_mask = gloss_attn_mask.to(gloss_device)
                else:
                    gloss_ids = gloss_ids.cuda()
                    gloss_attn_mask = gloss_attn_mask.cuda()
                gloss_output = model.gloss_forward(gloss_ids, gloss_attn_mask)
                gloss_output_T = gloss_output.transpose(0,1)
                #get cosine sim of example from context encoder with gloss embeddings
                if multigpu:
                    metaphor_output = metaphor_output.cpu()
                    gloss_output = gloss_output.cpu()
                output = torch.mm(metaphor_output, gloss_output_T)
                attn_score = torch.softmax(output, dim=-1)
                context_vector = torch.mm(attn_score, gloss_output)

                if multigpu:
                    metaphor_output = metaphor_output.to(context_device)
                    context_vector = context_vector.to(context_device)
                predicted_metaphor = model.fc_forward(torch.cat((metaphor_output, context_vector), dim=-1))
                pred_idx = predicted_metaphor.topk(1, dim=-1)[1].squeeze().item()
                eval_m_pred.append(pred_idx)

            t = sum([1 for output_mask in context_output_mask if torch.max(output_mask).item() == -1])
            if t == len(context_output_mask):
                continue

            context_output = model.context_forward(context_ids, context_attn_mask, context_output_mask)
            for output, key in zip(context_output.split(1,dim=0), gloss_key):
                #run example's glosses through gloss encoder

                gloss_ids, gloss_attn_mask, sense_keys = gloss_dict[key]
                if multigpu:
                    gloss_ids = gloss_ids.to(gloss_device)
                    gloss_attn_mask = gloss_attn_mask.to(gloss_device)
                else:
                    gloss_ids = gloss_ids.cuda()
                    gloss_attn_mask = gloss_attn_mask.cuda()
                gloss_output = model.gloss_forward(gloss_ids, gloss_attn_mask)
                gloss_output = gloss_output.transpose(0,1)

                #get cosine sim of example from context encoder with gloss embeddings
                if multigpu:
                    output = output.cpu()
                    gloss_output = gloss_output.cpu()
                output = torch.mm(output, gloss_output)
                pred_idx = output.topk(1, dim=-1)[1].squeeze().item()
                pred_label = sense_keys[pred_idx]
                eval_preds.append(pred_label)

    return eval_preds, eval_m_pred

def train_model(args):
    print('Training WSD bi-encoder model...')
    if args.freeze_gloss: assert args.gloss_bsz == -1 #no gloss bsz if not training gloss encoder, memory concerns

    #create passed in ckpt dir if doesn't exist
    if not os.path.exists(args.ckpt): os.mkdir(args.ckpt)

    '''
    LOAD PRETRAINED TOKENIZER, TRAIN AND DEV DATA
    '''
    print('Loading data + preprocessing...')
    sys.stdout.flush()

    tokenizer = load_tokenizer(args.encoder_name)

    #loading WSD (semcor) data
    train_path = os.path.join(args.data_path, 'VUA_seq_train.csv')
    train_data = load_VUA_data(train_path)
    gloss_file = os.path.join(args.data_path, 'gloss/vua_all_meaning.csv')

    #filter train data for k-shot learning
    if args.kshot > 0: train_data = filter_k_examples(train_data, args.kshot)


    label_glosses = get_label_glosses(train_path)
    #load gloss dictionary (all senses from webster dictionary)
    train_gloss_dict, train_gloss_weights = load_and_preprocess_webster_glosses(gloss_file, tokenizer, label_glosses, max_len=32)
    train_gloss_dict = update_gloss_dict(train_path, train_gloss_dict, tokenizer, max_len=args.gloss_max_length)

    #preprocess and batch data (context + glosses)  
    train_data = preprocess_context(tokenizer, train_data, bsz=args.context_bsz, max_len=args.context_max_length)
    

    epochs = args.epochs
    overflow_steps = -1
    t_total = len(train_data)*epochs

    #if few-shot training, override epochs to calculate num. epochs + steps for equal training signal
    if args.kshot > 0:
        #hard-coded num. of steps of fair kshot evaluation against full model on default numer of epochs
        NUM_STEPS = 181500 #num batches in full train data (9075) * 20 epochs 
        num_batches = len(train_data)
        epochs = NUM_STEPS//num_batches #recalculate number of epochs
        overflow_steps = NUM_STEPS%num_batches #num steps in last overflow epoch (if there is one, otherwise 0)
        t_total = NUM_STEPS #manually set number of steps for lr schedule
        if overflow_steps > 0: epochs+=1 #add extra epoch for overflow steps
        print('Overriding args.epochs and training for {} epochs...'.format(epochs))

    ''' 
    SET UP FINETUNING MODEL, OPTIMIZER, AND LR SCHEDULE
    '''
    model = JointBiEncoderModel(args.encoder_name, freeze_gloss=args.freeze_gloss, freeze_context=args.freeze_context, tie_encoders=args.tie_encoders)
    if not args.train_w:
        model_path = os.path.join(args.ckpt, 'VUA_na_best_model_context.ckpt')
        model.load_state_dict(torch.load(model_path))
    #speeding up training by putting two encoders on seperate gpus (instead of data parallel)
    if args.multigpu: 
        model.gloss_encoder = model.gloss_encoder.to(gloss_device)
        model.context_encoder = model.context_encoder.to(context_device)
        model.linearfc = model.linearfc.to(context_device)
    else:
        model = model.cuda()

    criterion = {}
    if args.balanced:
        for key in train_gloss_dict:
            criterion[key] = torch.nn.CrossEntropyLoss(reduction='none', weight=train_gloss_weights[key])
    else:
        for key in train_gloss_dict:
            criterion[key] = torch.nn.CrossEntropyLoss(reduction='none')
    metaphor_criterion = torch.nn.NLLLoss()

    #optimize + scheduler from pytorch_transformers package
    #this taken from pytorch_transformers finetuning code
    weight_decay = 0.0 #this could be a parameter
    no_decay = ['bias', 'LayerNorm.weight']
    diff_lr = 'linearfc'
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    adam_epsilon = 1e-8
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=adam_epsilon)
    schedule = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup, t_total=t_total)


    #loading WSD (semcor) data
    dev_path = os.path.join(args.data_path, 'VUA_seq_val.csv')
    dev_data = load_VUA_data(dev_path)
    gloss_file = os.path.join(args.data_path, 'gloss/vua_all_meaning.csv')



    dev_label_glosses = get_label_glosses(dev_path)
    dev_metaphor_labels, dev_marks, dev_nums = get_metaphor_label(dev_path)
    #load gloss dictionary (all senses from webster dictionary)
    dev_gloss_dict, dev_gloss_weights = load_and_preprocess_webster_glosses(gloss_file, tokenizer, dev_label_glosses, max_len=32)
    dev_gloss_dict = update_gloss_dict(dev_path, dev_gloss_dict, tokenizer, max_len=args.gloss_max_length)
    

    #preprocess and batch data (context + glosses)  
    dev_data = preprocess_context(tokenizer, dev_data, bsz=args.context_bsz, max_len=args.context_max_length)




    '''

    '''

    dev_preds, dev_m_preds = _eval(dev_data, model, dev_gloss_dict, multigpu=args.multigpu)


    '''
    TRAIN MODEL ON SEMCOR DATA
    '''

    best_dev_f1 = 0.
    print('Training probe...')
    sys.stdout.flush()

    for epoch in range(1, epochs+1):
        #if last epoch, pass in overflow steps to stop epoch early
        train_steps = -1
        if epoch == epochs and overflow_steps > 0: train_steps = overflow_steps

        #train model for one epoch or given number of training steps
        model, optimizer, schedule, train_loss = _train(train_data, model, train_gloss_dict, optimizer, schedule, criterion, metaphor_criterion, gloss_bsz=args.gloss_bsz, max_grad_norm=args.grad_norm, silent=args.silent, multigpu=args.multigpu, train_steps=train_steps, _train_m=args.train_m, _train_w=args.train_w)

        #eval model on dev set (semeval2007)
        dev_preds, dev_m_preds = _eval(dev_data, model, dev_gloss_dict, multigpu=args.multigpu)

        #generate predictions file
        pred_filepath = os.path.join(args.ckpt, 'VUA_tmp_predictions.txt')
        with open(pred_filepath, 'w') as f:
            for prediction in dev_preds:
                f.write('{}\n'.format(prediction))

        f_m_label, f_m_pred = dev_metaphor_labels, dev_m_preds
        

        print("test all pos accuracy: {:.4f}".format(accuracy_score(f_m_label, f_m_pred)))
        print("test all pos precision: {:.4f}".format(precision_score(f_m_label, f_m_pred)))
        print("test all pos recall: {:.4f}".format(recall_score(f_m_label, f_m_pred)))
        dev_acc = compute_acc(dev_label_glosses, dev_preds)
        print("test all pos f1: {:.4f}".format(f1_score(f_m_label, f_m_pred)), '*'*50)
        print('Dev acc after {} epochs = {}'.format(epoch, dev_acc))
        sys.stdout.flush()
        dev_f1 = f1_score(f_m_label, f_m_pred)
        if dev_f1 >= best_dev_f1:
            print('updating best model at epoch {}...'.format(epoch))
            sys.stdout.flush() 
            best_dev_f1 = dev_f1
            #save to file if best probe so far on dev set
            model_fname = os.path.join(args.ckpt, 'VUA_na_best_model_context.ckpt')
            with open(model_fname, 'wb') as f:
                torch.save(model.state_dict(), f)
            sys.stdout.flush()

        #shuffle train set ordering after every epoch
        random.shuffle(train_data)
    return

    

def evaluate_model(args):
    print('Evaluating WSD model on {}...'.format(args.split))

    '''
    LOAD TRAINED MODEL
    '''
    model = JointBiEncoderModel(args.encoder_name, freeze_gloss=args.freeze_gloss, freeze_context=args.freeze_context)
    model_path = os.path.join(args.ckpt, 'VUA_na_best_model_context.ckpt')
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    

    '''
    LOAD TOKENIZER
    '''
    tokenizer = load_tokenizer(args.encoder_name)

    '''
    LOAD EVAL SET
    '''



    #loading WSD (semcor) data
    eval_path = os.path.join(args.data_path, 'vua_seq_test.csv')
    eval_data = load_VUA_data(eval_path)
    gloss_file = os.path.join(args.data_path, 'gloss/vua_all_meaning.csv')



    eval_label_glosses = get_label_glosses(eval_path)
    eval_metaphor_labels, eval_marks, eval_nums = get_metaphor_label(eval_path)
    #load gloss dictionary (all senses from webster dictionary)
    eval_gloss_dict, eval_gloss_weights = load_and_preprocess_webster_glosses(gloss_file, tokenizer, eval_label_glosses, max_len=32)
    eval_gloss_dict = update_gloss_dict(eval_path, eval_gloss_dict, tokenizer, max_len=args.gloss_max_length)

    #preprocess and batch data (context + glosses)  
    eval_data = preprocess_context(tokenizer, eval_data, bsz=args.context_bsz, max_len=args.context_max_length)

    '''
    EVALUATE MODEL
    '''
    eval_preds, eval_m_preds = _eval(eval_data, model, eval_gloss_dict, multigpu=False)

    f_m_label, f_m_pred = eval_metaphor_labels, eval_m_preds


    print("test all pos accuracy: {:.4f}".format(accuracy_score(f_m_label, f_m_pred)))
    print("test all pos precision: {:.4f}".format(precision_score(f_m_label, f_m_pred)))
    print("test all pos recall: {:.4f}".format(recall_score(f_m_label, f_m_pred)))
    print("test all pos f1: {:.4f}".format(f1_score(f_m_label, f_m_pred)), '*'*50)

    #generate predictions file
    pred_filepath = os.path.join(args.ckpt, 'VUA_test_predictions.txt')
    with open(pred_filepath, 'w') as f:
        for m_num, m_l, prediction, mark in zip(eval_nums, eval_metaphor_labels, eval_m_preds, eval_marks):
            f.write('{} {} {} {}\n'.format(m_num, m_l, prediction, mark))

    #generate predictions file
    pred_filepath = os.path.join(args.ckpt, 'VUA_test_predictions_para.txt')
    with open(pred_filepath, 'w') as f:
        for m_num, m_l, prediction in zip(eval_nums, eval_label_glosses, eval_preds):
            f.write('{} {} {}\n'.format(m_num, m_l, prediction))


    # #run predictions through scorer
    # gold_filepath = os.path.join(eval_path, '{}.gold.key.txt'.format(args.split))
    # scorer_path = os.path.join(args.data_path, 'Evaluation_Datasets')
    # p, r, f1 = evaluate_output(scorer_path, gold_filepath, pred_filepath)
    # for e_l, ev in zip(eval_label_glosses, eval_preds):
    #     print(e_l, ev)
    # print(len(eval_label_glosses), len(eval_preds))
    print('acc of BERT probe on test set = {}'.format(compute_acc(eval_label_glosses, eval_preds)))

    return

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Need available GPU(s) to run this model...")
        quit()

    #parse args
    args = parser.parse_args()
    print(args)

    #set random seeds
    torch.manual_seed(args.rand_seed)
    os.environ['PYTHONHASHSEED'] = str(args.rand_seed)
    torch.cuda.manual_seed(args.rand_seed)
    torch.cuda.manual_seed_all(args.rand_seed)   
    np.random.seed(args.rand_seed)
    random.seed(args.rand_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic=True

    #evaluate model saved at checkpoint or...
    if args.eval: evaluate_model(args)
    #train model
    else: train_model(args)

#EOF
