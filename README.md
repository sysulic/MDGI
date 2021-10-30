# MDGI
The source code and data for ACL2021 Findings paper: Enhancing Metaphor Detection by Gloss-based Interpretations


The code is modified based on [Moving Down the Long Tail of Word Sense Disambiguation with Gloss Informed Bi-encoders] https://github.com/facebookresearch/wsd-biencoders

## Dependencies 
To run this code, you'll need the following libraries:
* [Python 3](https://www.python.org/)
* [Pytorch 1.2.0](https://pytorch.org/)
* [Pytorch Transformers 1.1.0](https://github.com/huggingface/transformers)
* [Numpy 1.17.2](https://numpy.org/)
* [NLTK 3.4.5](https://www.nltk.org/)
* [tqdm](https://tqdm.github.io/)
* [xlrd 1.2.0](https://github.com/python-excel/xlrd)



## How to Run 
To train a MDGI-Joint model, run `python biencoder_PSUCMC_MDGI-Joint.py --data-path $path_to_data --ckpt $path_to_checkpoint --encoder-name $path_to_BERT(or other PLM)`. The required arguments are: `--data-path`, which is the access to the data path; and `--ckpt`, which is the filepath of the directory to which to save the trained model checkpoints and prediction files; `--encoder-name`, which is the name of the pretrained language model(if there exists problems, please look into the models/utils.py). 



To evaluate an existing biencoder, run `python biencoder_PSUCMC_MDGI-Joint.py --data-path $path_to_data --ckpt $path_to_checkpoint --eval `.



## Notice
1. MDGI-Joint-S is implemented by adding the argument `--tie-encoders` when running `biencoder_*_MDGI-Joint.py`.

2. When evaluating the model on VUA, please add the argument `--context-max-length 256`.

3. The current version does not support multigpu.
