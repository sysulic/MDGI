'''
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
'''

import torch
from torch.nn import functional as F
import math
import os
import sys
from pytorch_transformers import *
import torch.nn as nn

from wsd_models.util import *

def load_projection(path):
    proj_path = os.path.join(path, 'best_probe.ckpt')
    with open(proj_path, 'rb') as f: proj_layer = torch.load(f)
    return proj_layer

class PretrainedClassifier(torch.nn.Module):
    def __init__(self, num_labels, encoder_name, proj_ckpt_path):
        super(PretrainedClassifier, self).__init__()

        self.encoder, self.encoder_hdim = load_pretrained_model(encoder_name)

        if proj_ckpt_path and len(proj_ckpt_path) > 0:
            self.proj_layer = load_projection(proj_ckpt_path)
            #assert to make sure correct dims
            assert self.proj_layer.in_features == self.encoder_hdim
            assert self.proj_layer.out_features == num_labels
        else:
            self.proj_layer = torch.nn.Linear(self.encoder_hdim, num_labels)

    def forward(self, input_ids, input_mask, example_mask):
        output = self.encoder(input_ids, attention_mask=input_mask)[0]

        example_arr = []        
        for i in range(output.size(0)): 
            example_arr.append(process_encoder_outputs(output[i], example_mask[i], as_tensor=True))
        output = torch.cat(example_arr, dim=0)
        output = self.proj_layer(output)
        return output

class GlossEncoder(torch.nn.Module):
    def __init__(self, encoder_name, freeze_gloss, tied_encoder=None):
        super(GlossEncoder, self).__init__()

        #load pretrained model as base for context encoder and gloss encoder
        if tied_encoder:
            self.gloss_encoder = tied_encoder
            _, self.gloss_hdim = load_pretrained_model(encoder_name)
        else:
            self.gloss_encoder, self.gloss_hdim = load_pretrained_model(encoder_name)
        self.is_frozen = freeze_gloss

    def forward(self, input_ids, attn_mask):
        #encode gloss text
        if self.is_frozen:
            with torch.no_grad(): 
                gloss_output = self.gloss_encoder(input_ids, attention_mask=attn_mask)[0]
        else:
            gloss_output = self.gloss_encoder(input_ids, attention_mask=attn_mask)[0]
        #training model to put all sense information on CLS token 
        gloss_output = gloss_output[:,0,:].squeeze(dim=1) #now bsz*gloss_hdim
        return gloss_output
    def activate(self):
        self.is_frozen = False

class ContextEncoder(torch.nn.Module):
    def __init__(self, encoder_name, freeze_context):
        super(ContextEncoder, self).__init__()

        #load pretrained model as base for context encoder and gloss encoder
        self.context_encoder, self.context_hdim = load_pretrained_model(encoder_name)
        self.is_frozen = freeze_context

    def forward(self, input_ids, attn_mask, output_mask):
        #encode context
        if self.is_frozen:
            with torch.no_grad(): 
                context_output = self.context_encoder(input_ids, attention_mask=attn_mask)[0]
        else:
            context_output = self.context_encoder(input_ids, attention_mask=attn_mask)[0]

        #average representations over target word(s)
        example_arr = []        
        for i in range(context_output.size(0)): 
            if torch.max(output_mask[i]).item() == -1:
                continue
            example_arr.append(process_encoder_outputs(context_output[i], output_mask[i], as_tensor=True))
        context_output = torch.cat(example_arr, dim=0)

        return context_output

    def activate(self):
        self.is_frozen = False

class LinearFC(nn.Module):
    def __init__(self, num_classes, encoded_embedding_dim, dropout1=0.2):
        '''
        :param num_classes: The number of classes in the classification problem.
        :param encoded_embedding_dim: The dimension of the encoded word
        :param context_dim: the dimension of the context vector of the paraphrase
        :param dropout1: dropout on input to FC
        '''

        super(LinearFC, self).__init__()


        self.output_to_label = nn.Linear(encoded_embedding_dim, num_classes)

        self.dropout_on_input_to_FC = nn.Dropout(dropout1)


    def forward(self, encoded_state):

        #inputs = torch.cat((encoded_state, context_vector),dim=-1)

        embedded_input = self.dropout_on_input_to_FC(encoded_state)

        output = self.output_to_label(embedded_input)

        normalized_output = torch.log_softmax(output, dim=-1)

        return normalized_output








class AttentionModel(nn.Module):
    # encoder_input_dim: The encoder input dimension, outside ex_dim
    # decoder_input_dim: The decoder input dimension
    # output_dim: The output dimension
    # dropout1: dropout on input(encoder output) to linear layer
    # dropout2: dropout on input(decoder output) to linear layer
    def __init__(self, para_encoder_input_dim, query_dim, output_dim, dropout1=0.1, dropout2=0.1):

        super(AttentionModel, self).__init__()

        self.dropout_on_para_encode_state = nn.Dropout(dropout1)

        self.dropout_on_query = nn.Dropout(dropout2)

        self.output_dim = output_dim

        self.para_weight = nn.Linear(para_encoder_input_dim, output_dim, bias=False)

        self.query_weight = nn.Linear(query_dim, output_dim, bias=True)

        self.attention_vec = nn.Parameter(torch.randn(1, output_dim), requires_grad=True)




    def forward(self, para_encode_state, query):

        # expand the dim into [atten_len, atten_size]
        #para_encode_state = para_encode_state.unsqueeze(2)

        # batch_size = para_encode_state_size[0]
        # attn_size = para_encode_state_size[-1]
        # query_repeat = query.repeat(para_encode_state.size()[0], 1)
        # inputs = torch.cat((para_encode_state, query_repeat), dim=-1)
        # inputs = self.dropout_on_para_encode_state(inputs)
        # e = self.para_weight(inputs).squeeze(-1)
        # e = torch.log_softmax(e, dim=-1)
        para_encode_state = self.dropout_on_para_encode_state(para_encode_state)
        para_linear = self.para_weight(para_encode_state)


        

        # # expand the dim into [batch_size, 1, 1, attn_size]
        query = self.dropout_on_query(query)
        query_linear = self.query_weight(query)

        # # query_linear = query_linear.unsqueeze(-2)


        e = torch.matmul(self.attention_vec, torch.transpose(torch.tanh(para_linear + query_linear), 1, 0))



        e = torch.log_softmax(e, dim=-1)


        # e = para_linear.squeeze(-1).unsqueeze(0)





        # take softmax. shape (batch_size, attn_length)



        return e  # re-normalize

class BiEncoderModel(torch.nn.Module):
    def __init__(self, encoder_name, freeze_gloss=False, freeze_context=False, tie_encoders=False):
        super(BiEncoderModel, self).__init__()

        #tying encoders for ablation
        self.tie_encoders = tie_encoders

        #load pretrained model as base for context encoder and gloss encoder
        self.context_encoder = ContextEncoder(encoder_name, freeze_context)
        if self.tie_encoders:
            self.gloss_encoder = GlossEncoder(encoder_name, freeze_gloss, tied_encoder=self.context_encoder.context_encoder)
        else:
            self.gloss_encoder = GlossEncoder(encoder_name, freeze_gloss)
        assert self.context_encoder.context_hdim == self.gloss_encoder.gloss_hdim
        self.linearfc = LinearFC(num_classes=2, encoded_embedding_dim=self.context_encoder.context_hdim)
        self.attn_model = AttentionModel(768, 768, 128)

    def context_forward(self, context_input, context_input_mask, context_example_mask):
        return self.context_encoder.forward(context_input, context_input_mask, context_example_mask)

    def gloss_forward(self, gloss_input, gloss_mask):
        return self.gloss_encoder.forward(gloss_input, gloss_mask)

    def fc_forward(self, fc_input):
        return self.linearfc.forward(fc_input)

    def compute_attention(self, para_input, query):
        return self.attn_model.forward(para_input, query)



class JointBiEncoderModel(torch.nn.Module):
    def __init__(self, encoder_name, freeze_gloss=False, freeze_context=False, tie_encoders=False):
        super(JointBiEncoderModel, self).__init__()

        #tying encoders for ablation
        self.tie_encoders = tie_encoders

        #load pretrained model as base for context encoder and gloss encoder
        self.context_encoder = ContextEncoder(encoder_name, freeze_context)
        if self.tie_encoders:
            self.gloss_encoder = GlossEncoder(encoder_name, freeze_gloss, tied_encoder=self.context_encoder.context_encoder)
        else:
            self.gloss_encoder = GlossEncoder(encoder_name, freeze_gloss)
        assert self.context_encoder.context_hdim == self.gloss_encoder.gloss_hdim
        self.linearfc = LinearFC(num_classes=2, encoded_embedding_dim=self.context_encoder.context_hdim+self.gloss_encoder.gloss_hdim)

    def context_forward(self, context_input, context_input_mask, context_example_mask):
        return self.context_encoder.forward(context_input, context_input_mask, context_example_mask)

    def metaphor_context_forward(self, context_input, context_input_mask, context_example_mask, train_w):
        if train_w:
            output = self.context_encoder.forward(context_input, context_input_mask, context_example_mask)
        else:
            with torch.no_grad():
                output = self.context_encoder.forward(context_input, context_input_mask, context_example_mask)
        return output

    def gloss_forward(self, gloss_input, gloss_mask):
        return self.gloss_encoder.forward(gloss_input, gloss_mask)
    def metaphor_gloss_forward(self, gloss_input, gloss_mask, train_w):
        if train_w:
            gloss = self.gloss_encoder.forward(gloss_input, gloss_mask)
        else:
            with torch.no_grad():
                gloss = self.gloss_encoder.forward(gloss_input, gloss_mask)
        return gloss 

    def fc_forward(self, fc_input):
        return self.linearfc.forward(fc_input)

    def compute_attention(self, para_input, query):
        return self.attn_model.forward(para_input, query)


    #EOF