#!/usr/bin/env python
# coding: utf-8
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
import operator, functools
from sacremoses import MosesPunctNormalizer, MosesTokenizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
# from accelerate import Accelerator
import nltk
import sys
import pandas as pd
import wandb
from random import randrange
import random
import json
import os
import random
import tqdm
import sys
import time
import glob
import numpy as np
import torch
# import utils
import logging
import gc
import argparse
import evaluate

import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler

from tokenizers import Tokenizer
import datasets
from sacremoses import MosesPunctNormalizer, MosesTokenizer
# from data_set import *
from transformers import GPT2Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM

import torchtext
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab,vocab
from torchtext.utils import download_from_url, extract_archive
import io


from models.GPT import *
from models.mBART import *
from models.RNN_sp import *
from models.attention_params import *

from models.hyperparams import *
from models.Transformer_sp import *

from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.configs import Config, EngineConfig

import sentencepiece as spm
nltk.download('punkt')

parser = argparse.ArgumentParser("Machine Translation")
parser.add_argument('--dataset_name', type=str, default='wmt14_gigafren', help='dataset_name')
parser.add_argument('--num_beams', type=int, default=2, help='Beam Size')
parser.add_argument('--max_source_length', type=int, default=50, help='Max Source length')
parser.add_argument('--max_target_length', type=int, default=50, help='Max Target length')
parser.add_argument('--pad_to_max_length', type=bool, default=False, help='Pad to max length')
parser.add_argument('--ignore_pad_token_for_loss', type=bool, default=True, help='Ignore pad token for loss')
parser.add_argument('--per_device_train_batch_size', type=int, default=32, help='Train Batch Size')
parser.add_argument('--per_device_eval_batch_size', type=int, default=32, help='Eval Batch Size')

parser.add_argument('--mbart_learning_rate', type=float, default=1e-3, help='mBART learning rate')
parser.add_argument('--mbart_learning_rate_min', type=float, default=5e-4, help='mBART min learning rate')
parser.add_argument('--mbart_momentum', type=float, default=0.9, help='mBART momentum')
parser.add_argument('--mbart_weight_decay', type=float, default=0, help='mBART weigth decay')

parser.add_argument('--gpt_learning_rate', type=float, default=1e-3, help='GPT learning rate')
parser.add_argument('--gpt_learning_rate_min', type=float, default=5e-4, help='GPT min learning rate')
parser.add_argument('--gpt_momentum', type=float, default=0.9, help='GPT momentum')
parser.add_argument('--gpt_weight_decay', type=float, default=0, help='GPT weigth decay')

parser.add_argument('--rnn_learning_rate_adam', type=float, default=1e-3, help='RNN learning rate')
parser.add_argument('--rnn_learning_rate_min_adam', type=float, default=5e-4, help='RNN min learning rate')
parser.add_argument('--rnn_momentum', type=float, default=0.0, help='RNN momentum')
parser.add_argument('--rnn_weight_decay', type=float, default=5e-7, help='RNN weigth decay')
parser.add_argument('--beta1', type=float, default=0.9, help='Beta 1')
parser.add_argument('--beta2', type=float, default=0.999, help='Beta 2')
parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon')
parser.add_argument('--rnn_learning_rate_sgd', type=float, default=1, help='RNN learning rate sgd')
parser.add_argument('--rnn_learning_rate_min_sgd', type=float, default=0.1, help='RNN min learning rate sgd')
parser.add_argument('--rnn_optimizer', type=str, default='adam', help='RNN Optimizer')

parser.add_argument('--begin_epoch', type=int, default=0, help='Begin Epoch')
parser.add_argument('--stop_epoch', type=int, default=5, help='Stop Epoch')
parser.add_argument('--report_freq', type=int, default=50, help='Report Frequency')
parser.add_argument('--gpu', type=int, default=0, help='GPU')
parser.add_argument('--num_train_epochs', type=int, default=15, help='Train Epochs')
parser.add_argument('--final_training', type=int, default=15, help='Train Epochs Final')

parser.add_argument('--seed', type=int, default=seed_, help='Seed')

parser.add_argument('--grad_clip', type=int, default=0.25, help='Grad clip')
parser.add_argument('--A_momentum', type=float, default=0.9, help='A momentum')
parser.add_argument('--A_learning_rate', type=float, default=1e-3, help='A learning rate')
parser.add_argument('--A_weight_decay', type=float, default=1e-3, help='A weight decay')

parser.add_argument('--B_momentum', type=float, default=0.9, help='B momentum')
parser.add_argument('--B_learning_rate', type=float, default=1e-3, help='B learning rate')
parser.add_argument('--B_weight_decay', type=float, default=1e-3, help='B weight decay')

parser.add_argument('--lambda_par', type=float, default=0.5, help='Lambda parameter')
parser.add_argument('--train_num_points', type=int, default=5000, help='Train number points')
parser.add_argument('--train_num_points_back', type=int, default=10000, help='Train number points extra')
parser.add_argument('--val_num_points', type=int, default=None, help='Validation number points')
parser.add_argument('--save', type=str, default='EXP_aug', help='Save dir')
parser.add_argument('--start_from_checkpoint', action='store_true', help='Use checkpoint')
parser.add_argument('--logging', action='store_true', help='Use logging')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
parser.add_argument('--save_models', action='store_true', help='Save models')
parser.add_argument('--update_arch', action='store_true', help='Save models')
parser.add_argument('--only_aug', action='store_true', help='Augmentation only loss')
parser.add_argument('--update_gpt_bart', action='store_true', help='Save models')
# parser.add_argument('--save_alpha', help='Save a and b parameters', action='store_true')
parser.add_argument('--save_alpha', action='store_true',help='Save Alpha')
parser.add_argument('--save_freq', type=int, default=1, help='Frequency of saving A and B')
parser.add_argument('--log_file_name', type=str, default='log.txt', help='Log file name')
parser.add_argument('--log_name', type=str, default='wandb', help='Log file name')
parser.add_argument('--validation_freq', type=int, default=5, help='Validation Frequency')
parser.add_argument('--full', type=str, default='', help='Full prefix')
parser.add_argument('--fp16', action='store_true',help='Low Precision')
parser.add_argument('--gradient_accumulation_steps', type = int, default = 1,help='gradient accumulation steps')
parser.add_argument('--max_train_steps', type = int, default = None,help='Max train steps')
parser.add_argument('--max_train_steps_free', type = int, default = None,help='Final Max train steps')
parser.add_argument('--use_engine', action='store_true', help='Use betty engine')
parser.add_argument('--sequential_train', action='store_true', help='Train Coplete model sequentially')
parser.add_argument('--transformer_model', action='store_true', help='To use Transformer model')
parser.add_argument('--no_aug', action='store_true', help='No augmentation')
parser.add_argument('--translation_language', type=str, default='fr', help='Translation language')
parser.add_argument('--baseline', type=str, default=None, help='Translation language')
parser.add_argument('--exp_name', type=str, default=None, help='Experiment Name')
parser.add_argument('--baseline_par', type=float, default=None, help='Baseline parameters')
parser.add_argument('--gpt_optimizer', type=str, default=None, help='GPT Optimizer Name')
parser.add_argument('--mbart_optimizer', type=str, default='adam', help='Mbart Optimizer Name')
parser.add_argument('--mbart_scheduler', type=str, default='cosine', help='Mbart scheduler Name')
parser.add_argument('--gpt_scheduler', type=str, default='cosine', help='gpt scheduler Name')
parser.add_argument('--summary_length', type=int, default=128, help='SOurce length')
parser.add_argument('--article_length', type=int, default=128, help='SOurce length')
parser.add_argument('--log_steps', type = int, default = 100,help='MLO log steps')
parser.add_argument('--dataset_type', type=str, default='iwslt', help='Dataset Name')
parser.add_argument('--A_optimizer', type=str, default=None, help='A, B Optimizer Name')
parser.add_argument('--src_lang', type=str, default='cor', help='lang name')
parser.add_argument('--trg_lang', type=str, default='eng', help='lang name')
parser.add_argument('--clip', action='store_false', help='Gradient clip false')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
parser.add_argument('--combine', action='store_true', help='Combine Data')
parser.add_argument('--data_name', type=str, default='flores', help='Data name')
parser.add_argument('--transformer', action='store_true', help='Transformer Model')
parser.add_argument('--darts_adam', action='store_true', help='Combine Data')
args = parser.parse_args()

args.device = "cuda" if torch.cuda.is_available() else "cpu"
from models.losses import *
def read_data(split='train'):
    with open('upload_datasets/'+args.src_lang+'_'+args.trg_lang+'/'+split+'.src', 'r') as file:
        # Read the file line by line
        src_lines = file.readlines()
    with open('upload_datasets/'+args.src_lang+'_'+args.trg_lang+'/'+split+'.trg', 'r') as file:
        # Read the file line by line
        trg_lines = file.readlines()
        
    for i in range(len(src_lines)):
        src_lines[i] = src_lines[i].strip()
        trg_lines[i] = trg_lines[i].strip()
    return src_lines,trg_lines


def read_bt_data():
    if args.data_name=='multi30':
            with open('upload_datasets/'+args.src_lang+'_'+args.trg_lang+'/wiki.aa.'+args.src_lang+'-'+args.trg_lang+'.'+args.src_lang+'.'+str(args.train_num_points), 'r') as file:
            # Read the file line by line
                src_lines = file.readlines()
            with open('upload_datasets/'+args.src_lang+'_'+args.trg_lang+'/wiki.aa.'+args.src_lang+'-'+args.trg_lang+'.'+args.trg_lang+'.'+str(args.train_num_points), 'r') as file:
                # Read the file line by line
                trg_lines = file.readlines()
    else:
        with open('upload_datasets/'+args.src_lang+'_'+args.trg_lang+'/wiki.aa.'+args.src_lang+'-'+args.trg_lang+'.'+args.src_lang, 'r') as file:
            # Read the file line by line
            src_lines = file.readlines()
        with open('upload_datasets/'+args.src_lang+'_'+args.trg_lang+'/wiki.aa.'+args.src_lang+'-'+args.trg_lang+'.'+args.trg_lang, 'r') as file:
            # Read the file line by line
            trg_lines = file.readlines()

    paired_lines = list(zip(src_lines, trg_lines))

    # Set the random seed for reproducibility
    random.seed(42)

    # Shuffle the paired lines
    random.shuffle(paired_lines)

    # Unpack the shuffled pairs
    src_lines, trg_lines = zip(*paired_lines)

    # Select top train_num_points_back
    src_lines = src_lines[:train_num_points_back]
    trg_lines = trg_lines[:train_num_points_back]
    return src_lines,trg_lines
#     return src_lines[:args.train_num_points_back],trg_lines[:args.train_num_points_back]
def data_process(filepaths):
    raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
    raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
    data = []
    i=0
    src = []
    trg = []
    for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
        src.append(raw_de.strip())
        trg.append(raw_en.strip())
    return src,trg
if args.data_name=='flores':
    dev_data = read_data('dev')
    test_data = read_data('test')
    dev_size = len(dev_data[0])
    dev_indices = list(range(dev_size))
    train_indices = random.sample(dev_indices, int(0.8 * dev_size))
    dev_indices = list(set(dev_indices) - set(train_indices))

    # Extract train and dev data based on the indices
    train_src_lines = [dev_data[0][i] for i in train_indices]
    train_trg_lines = [dev_data[1][i] for i in train_indices]
    dev_src_lines = [dev_data[0][i] for i in dev_indices]
    dev_trg_lines = [dev_data[1][i] for i in dev_indices]
    train_data = train_src_lines, train_trg_lines
    dev_data = dev_src_lines, dev_trg_lines
    
elif args.data_name=='multi30':
    url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
    train_urls = ('train.'+args.src_lang+'.gz', 'train.en.gz')
    val_urls = ('val.'+args.src_lang+'.gz', 'val.en.gz')
    test_urls = ('test_2016_flickr.'+args.src_lang+'.gz', 'test_2016_flickr.en.gz')

    train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
    val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
    test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]
    train_data = data_process(train_filepaths)
    random.shuffle(train_data)
    train_data = train_data[0][:args.train_num_points],train_data[1][:args.train_num_points]
#     print(train_data)
    if args.val_num_points is not None:
        val_data = data_process(val_filepaths)
        dev_data = val_data[0][:args.val_num_points],val_data[1][:args.val_num_points]
        test_data = data_process(test_filepaths)
        test_data = test_data[0][:args.val_num_points],test_data[1][:args.val_num_points]
    else:
        dev_data = data_process(val_filepaths)
        test_data = data_process(test_filepaths)
else:
    train_data = read_data('train')
    dev_data = read_data('dev')
    test_data = read_data('test')
if args.combine:

    # Combine train and dev data
    combined_data = list(zip(train_data[0], train_data[1])) + list(zip(dev_data[0], dev_data[1]))

    # Shuffle the combined data (optional)
    import random
    random.shuffle(combined_data)

    # Calculate the split index
    split_index = int(0.8 * len(combined_data))

    # Split the data into train and dev based on the split index
    train_data_split = combined_data[:split_index]
    dev_data_split = combined_data[split_index:]

    # Unzip the split data into separate lists
    train_src_lines, train_trg_lines = zip(*train_data_split)
    dev_src_lines, dev_trg_lines = zip(*dev_data_split)

    # Convert the lines to lists (optional, if needed)
    train_src_lines = list(train_src_lines)
    train_trg_lines = list(train_trg_lines)
    dev_src_lines = list(dev_src_lines)
    dev_trg_lines = list(dev_trg_lines)
    train_data = train_src_lines,train_trg_lines
    dev_data = dev_src_lines,dev_trg_lines
bt_data = read_bt_data()

class attention_params(torch.nn.Module):
    def __init__(self, N):
        super(attention_params, self).__init__()
#         self.alpha = torch.nn.Parameter(torch.ones(N))
        self.alpha = torch.nn.Parameter(torch.zeros(N))
#         self.alpha = torch.nn.Parameter(torch.rand(N))
        self.softmax = torch.nn.Softmax(dim=-1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        
    def forward(self, idx):
        
        probs = self.sigmoid(self.alpha[idx])

        
        return probs
    
def load_tokenizer(src_lang=args.src_lang,trg_lang=args.trg_lang,n_src= '4k',n_trg='4k'):
    sp_src = spm.SentencePieceProcessor(model_file='upload_datasets/'+src_lang+'_'+trg_lang+'/'+'opusTC.'+src_lang+'.'+n_src+'.spm')
    sp_trg = spm.SentencePieceProcessor(model_file='upload_datasets/'+src_lang+'_'+trg_lang+'/'+'opusTC.'+trg_lang+'.'+n_trg+'.spm')
    sp_src.SetEncodeExtraOptions("bos:eos")
    sp_trg.SetEncodeExtraOptions("bos:eos")
    return sp_src,sp_trg

sp_src, sp_trg = load_tokenizer()


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(seed_)

def data_process_train(data):
    data_process = []
    i=0
    for  raw_src,raw_trg in zip(data[0],data[1]):
        rnn_src = sp_src.encode(raw_src)
        rnn_trg = sp_trg.encode(raw_trg)
        data_process.append((rnn_src,rnn_trg,i))
        i+=1
    return data_process
train_tokens = data_process_train(train_data)
dev_tokens = data_process_train(dev_data)
test_tokens = data_process_train(test_data)

def generate_batch(data_batch):
    rnn_src_ids, rnn_trg_ids = [],[]
    attn_idxs = []
    for (src_id,trg_id,attn_idx) in data_batch:
        rnn_src_ids.append(torch.tensor(src_id,dtype=torch.int64))
        rnn_trg_ids.append(torch.tensor(trg_id,dtype=torch.int64))
        attn_idxs.append(torch.tensor(attn_idx,dtype = torch.int64))
    rnn_src_ids = pad_sequence(rnn_src_ids, batch_first=True, padding_value=0)
    rnn_trg_ids = pad_sequence(rnn_trg_ids, batch_first=True, padding_value=0)
    attn_idxs = torch.tensor(attn_idxs)
    return rnn_src_ids,rnn_trg_ids,attn_idxs

rnn_dataloader_train = DataLoader(train_tokens,batch_size=args.per_device_train_batch_size,
                        shuffle=False, collate_fn=generate_batch)
rnn_dataloader_val = DataLoader(dev_tokens,batch_size=args.per_device_train_batch_size,
                        shuffle=False, collate_fn=generate_batch)
rnn_dataloader_test = DataLoader(test_tokens,batch_size=args.per_device_train_batch_size,
                        shuffle=False, collate_fn=generate_batch)
bt_tokens = data_process_train(bt_data)

rnn_dataloader_bt = DataLoader(bt_tokens,batch_size=args.per_device_train_batch_size,
                        shuffle=False, collate_fn=generate_batch)

rnn_dataloader_target_source = DataLoader(train_tokens,batch_size=args.per_device_train_batch_size,
                        shuffle=False, collate_fn=generate_batch)

class Inner_1(ImplicitProblem):
    def forward(self, x):
        return self.module(x)

    def training_step(self, batch):
        rnn_src_ids,rnn_trg_ids,attn_idxs = batch
        
        attention_weights = self.outer1.module[1](attn_idxs)
        
        loss_vec = self.module.get_loss_vec(rnn_trg_ids,rnn_src_ids)
        loss_vec = loss_vec.mean(dim=-1)
#         print(loss_vec.shape)
        loss = torch.dot(attention_weights, loss_vec)

        scaling_factor = 1/torch.sum(attention_weights)
        l = scaling_factor*loss

        return l

    def configure_train_data_loader(self):
        return rnn_dataloader_target_source

    def configure_module(self):
        PAD_IDX = 0
        rnn_criterion=torch.nn.CrossEntropyLoss(ignore_index =PAD_IDX,reduction='none')# reduction='mean'
        rnn_criterion=rnn_criterion.cuda()
        if args.transformer:
            rnn_model = TRANSFORMER_MODEL(rnn_criterion,sp_trg,sp_src)
        else:
            rnn_model = RNN_MODEL(rnn_criterion,sp_trg,sp_src) 
        
        rnn_model.apply(init_weights)
        rnn_model = rnn_model.cuda()

        return rnn_model

    def configure_optimizer(self):
        # Optimizers
        mbart_params = group_parameters(self.module)
#         mbart_params = create_group_params(self.module, 4, args)
        if args.mbart_optimizer=='adam':
            mbart_optimizer = torch.optim.Adam(mbart_params,args.mbart_learning_rate)
        elif args.mbart_optimizer=='adamw':
            mbart_optimizer = torch.optim.AdamW(mbart_params)
        else:
            mbart_optimizer = torch.optim.SGD(mbart_params,args.mbart_learning_rate,momentum=args.mbart_momentum,weight_decay=args.mbart_weight_decay)
        return mbart_optimizer

    def configure_scheduler(self):
        pass
#         if args.mbart_scheduler=='cosine':
#             scheduler_mbart = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, float(args.num_train_epochs), eta_min=args.mbart_learning_rate_min)
#             return scheduler_mbart
#         elif args.mbart_scheduler=='linear':
#             scheduler_mbart = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=args.mbart_warmup_steps, num_training_steps=args.max_train_steps)
#             return scheduler_mbart
#         else:
#             pass
class Inner_2(ImplicitProblem):
    def forward(self, x,y):
        return self.module(x,y)

    def training_step(self, batch):
        rnn_src_ids,rnn_trg_ids,attn_idxs = batch
        rnn_src_ids_1,rnn_trg_ids_1,attn_idxs= self.get_batch_1() 
        loss_tr = self.module.loss(rnn_src_ids_1.to(device), rnn_trg_ids_1.to(device))
        loss_aug = calc_loss_aug_tri(rnn_trg_ids.to(device), self.inner1.module, self.module)
        if args.only_aug:
            rnn_loss = loss_aug
        else:
            rnn_loss = loss_tr + (args.lambda_par*loss_aug)
        return rnn_loss

    def configure_train_data_loader(self):
        return rnn_dataloader_bt

    def configure_module(self):
        self.train_iterator_1 = iter(rnn_dataloader_train)
        PAD_IDX = 0
        rnn_criterion=torch.nn.CrossEntropyLoss(ignore_index =PAD_IDX,reduction='mean')# reduction='mean'
        rnn_criterion=rnn_criterion.cuda()

        if args.transformer:
            rnn_model = TRANSFORMER_MODEL(rnn_criterion,sp_src,sp_trg)
        else:
            rnn_model = RNN_MODEL(rnn_criterion,sp_src,sp_trg)  

        rnn_model.apply(init_weights)
        rnn_model = rnn_model.cuda()

        return rnn_model

    def configure_optimizer(self):
        # Optimizers
        if args.rnn_optimizer=='adam':
            rnn_optimizer = optim.Adam(self.module.model.parameters())
        elif args.rnn_optimizer=='adamw':
            rnn_optimizer = optim.AdamW(self.module.model.parameters())
            
        elif args.rnn_optimizer=='sgd':
            rnn_optimizer = torch.optim.SGD(self.module.model.parameters(),args.rnn_learning_rate_sgd,momentum=args.rnn_momentum,weight_decay=args.rnn_weight_decay)
            
        return rnn_optimizer
    def configure_scheduler(self):
        pass
    def get_batch_1(self):
            """
            Load training batch from the user-provided data loader

            :return: New training batch
            :rtype: Any
            """
            
            try:
                batch_1 = next(self.train_iterator_1)
            except StopIteration:
                train_data_loader = rnn_dataloader_train
                self.train_iterator_1 = iter(train_data_loader)
                batch_1 = next(self.train_iterator_1)
            return batch_1

        
class Parent_1(ImplicitProblem):
    def forward(self, *args):
        return self.module(args)

    def training_step(self, batch, *args, **kwargs):
        rnn_src_ids,rnn_trg_ids,attn_idxs = batch
        loss = self.inner2(rnn_src_ids,rnn_trg_ids).loss
        return loss

    def configure_train_data_loader(self):
        return rnn_dataloader_val

    def configure_module(self):
        return torch.nn.ModuleList(modules=[attention_params(len(train_data[0])).to(device),attention_params(len(train_data[0])).to(device)])
#         return attention_params(len(train_data)).to(device)

    def configure_optimizer(self):
#         return torch.optim.SGD(self.module.parameters(), lr=1, momentum=0.9)
#         return optim.AdamW(self.module.parameters(),lr = args.A_learning_rate)
        if args.A_optimizer == "adam":
            return optim.Adam(self.module.parameters(),lr = args.A_learning_rate)
        else:
            return optim.AdamW(self.module.parameters(),lr = args.A_learning_rate)

        
def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def create_rnn_model():
    PAD_IDX = 0
    rnn_criterion=torch.nn.CrossEntropyLoss(ignore_index =PAD_IDX,reduction='mean')# reduction='mean'
    rnn_criterion=rnn_criterion.cuda()
#     if args.transformer_model:
#         rnn_model = TRANSFORMER_MODEL(rnn_criterion,rnn_tokenizer) 
#     else:
    rnn_model = RNN_MODEL(rnn_criterion,sp_src) 

    rnn_model.apply(init_weights)
    rnn_model = rnn_model.cuda()

    return rnn_model

def group_parameters(model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters
device = "cuda" if torch.cuda.is_available() else "cpu"


beam_width=1
temperature = 1.0
do_sample=False
repetition_penalty = 1.5
top_p = 0.9
top_k = 100
def evaluation(rnn_model, iterator):
    
    rnn_model.model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch[0].to(device)
            trg = batch[1].to(device)

            loss = rnn_model(src, trg, 0).loss #turn off teacher forcing
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

beam_width=1
temperature = 1.0
do_sample=False
repetition_penalty = 1.5
top_p = 0.9
top_k = 100
val_steps = 2
chrf_metric_1 = evaluate.load("chrf")
bleu_metric = evaluate.load("bleu")
meteor_metric = evaluate.load('meteor')
chrf_metric_2 = evaluate.load("chrf")

def valid_generate(rnn_model, iterator, beam_width=1,temperature = 1.0, do_sample=False, repetition_penalty = 1.0,top_p = 0.9, top_k = 100):

    rnn_model.model.eval()

    epoch_loss = 0


    with torch.no_grad():

        for i, batch in enumerate(iterator):
            trans_sentences = []
            labels_sentences = []
            src = batch[0].to(device)
            trg = batch[1].to(device)

            pred_sents = rnn_model.generate(src,beam_width=beam_width,temperature=temperature , do_sample=do_sample, repetition_penalty=repetition_penalty ,top_p=top_p, top_k=top_k)
#             pred_sents = rnn_model.tokenizer_english.decode_batch(pred_sents.detach().numpy(),skip_special_tokens=True)

#             print(trg)
            if type(pred_sents)==list:
                for i in range(len(pred_sents)):
                    pred_sents[i] = pred_sents[i].detach().cpu().numpy().tolist()
                pred_sents = rnn_model.tokenizer.decode(pred_sents)
            else:
                pred_sents = rnn_model.tokenizer.decode(pred_sents.detach().cpu().numpy().tolist())
            given_sents = rnn_model.tokenizer.decode(trg.detach().cpu().numpy().tolist())
            source_sents = sp_src.decode(src.detach().cpu().numpy().tolist())


            pred_sents = [ti.replace(" ⁇ ", "") for ti  in pred_sents]
            given_sents_rest = [ti.replace(" ⁇ ", "") for ti  in given_sents]
            source_sents = [ti.replace(" ⁇ ", "") for ti  in source_sents]
            given_sents_bleu = [[ti.replace(" ⁇ ", "")] for ti  in given_sents]
            
            bleu_metric.add_batch(predictions=pred_sents, references=given_sents_bleu)
            meteor_metric.add_batch(predictions=pred_sents, references=given_sents_rest)
            chrf_metric_1.add_batch(predictions=pred_sents, references=given_sents_rest)
            chrf_metric_2.add_batch(predictions=pred_sents, references=given_sents_rest)
#             metric_bleurt.add_batch(predictions=pred_sents, references=meteor_given)

    print(given_sents_rest[:5])
    print(pred_sents[:5])



#     rouge = metric_valid_rouge.compute(use_stemmer=True)
    meteor = meteor_metric.compute()['meteor']
#     bleu = bleu_metric.compute()['bleu']
    chrf_1 = chrf_metric_1.compute(word_order=1,lowercase=True)['score']
    chrf_2 = chrf_metric_2.compute(word_order=2,lowercase=True)['score']
#     bleurt_score = np.mean(metric_bleurt.compute()['scores'])
    return meteor,chrf_1,chrf_2

#initiate best accuracy
best_bleu = -1

#when we have to define a validation level then we make a subclass of Engine to do so
#if a validation level is not required we do not need this class
min_loss = float('inf')
max_bleu = 0
max_rouge = 0
max_meteor = 0
class ReweightingEngine(Engine):
    @torch.no_grad()

    #defines the validation level
    def validation(self):

        #initiate correct number of predictions and total predictions
        correct = 0
        total = 0
        global min_loss,max_bleu,max_rouge,max_meteor,best_bleu
        test_loss = evaluation(self.inner2.module,rnn_dataloader_test)
        meteor_test,chrf_1_test,chrf_2_test = valid_generate(self.inner2.module,rnn_dataloader_test, beam_width=beam_width,temperature = temperature, do_sample=do_sample, repetition_penalty = repetition_penalty,top_p = top_p, top_k = top_k)
        
        valid_loss = evaluation(self.inner2.module,rnn_dataloader_val)
        meteor_val,chrf_1_val,chrf_2_val = valid_generate(self.inner2.module,rnn_dataloader_val, beam_width=beam_width,temperature = temperature, do_sample=do_sample, repetition_penalty = repetition_penalty,top_p = top_p, top_k = top_k)
        print("valid loss")
        print(valid_loss)

        #update best accuracy if the new accuracy is greater than the previous accuracy
#         if best_bleu < bleu_test:
#             best_bleu = bleu_test
            
        d = {"loss": test_loss,"meteor":meteor_test,"chrf_1":chrf_1_test,"chrf_2":chrf_2_test}
        print(d)
        engine.logger.log(d,tag='test')
        print(self.outer1.module[1].alpha[:100])
        rnn_src_ids,rnn_trg_ids,attn_idxs = engine.inner2.get_batch_1()
#         generate_aug_data_tri(rnn_trg_ids.to(device), engine.inner1.module, engine.inner1.module)
        #return accuracy and best accuracy as a dictionary
        return {"loss": valid_loss,"meteor":meteor_val,"chrf_1":chrf_1_val,"chrf_2":chrf_2_val}



if args.clip and args.darts_adam:
    inner1_config = Config(type="darts_adam", fp16=args.fp16, log_step=100,unroll_steps = 1,gradient_clipping=5.0)#,gradient_clipping=5.0
    inner_1 = Inner_1(name="inner1", config=inner1_config)
    inner2_config = Config(type="darts_adam", fp16=args.fp16, log_step=100,unroll_steps = 1,gradient_clipping=5.0)#,gradient_clipping=5.0
    inner_2 = Inner_2(name="inner2", config=inner2_config)
    outer_config = Config(type="darts_adam", fp16=args.fp16, log_step=100,gradient_clipping=5.0)

    outer_1 = Parent_1(name="outer1", config=outer_config)
elif args.clip and not args.darts_adam:
    inner1_config = Config(type="darts", fp16=args.fp16, log_step=100,unroll_steps = 1,gradient_clipping=5.0)#,gradient_clipping=5.0
    inner_1 = Inner_1(name="inner1", config=inner1_config)
    inner2_config = Config(type="darts", fp16=args.fp16, log_step=100,unroll_steps = 1,gradient_clipping=5.0)#,gradient_clipping=5.0
    inner_2 = Inner_2(name="inner2", config=inner2_config)

    outer_config = Config(type="darts", fp16=args.fp16, log_step=100,gradient_clipping=5.0)
    outer_1 = Parent_1(name="outer1", config=outer_config)
    
else:
    inner1_config = Config(type="darts", fp16=args.fp16, log_step=100,unroll_steps = 1)#,gradient_clipping=5.0
    inner_1 = Inner_1(name="inner1", config=inner1_config)
    inner2_config = Config(type="darts", fp16=args.fp16, log_step=100,unroll_steps = 1)#,gradient_clipping=5.0
    inner_2 = Inner_2(name="inner2", config=inner2_config)
    outer_config = Config(type="darts", fp16=args.fp16, log_step=100)
    outer_1 = Parent_1(name="outer1", config=outer_config)



# outer_2 = Parent_2(name="outer2", config=outer_config, device=args.device)

n1 = math.ceil(len(rnn_dataloader_bt) / args.gradient_accumulation_steps)
n2 = math.ceil(len(rnn_dataloader_train) / args.gradient_accumulation_steps)
num_update_steps_per_epoch = n1+n2
if args.max_train_steps is None:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True
# problems = [inner_1,inner_2,inner_3,outer_1]
problems = [inner_1,inner_2,outer_1]

#setup engine configuration using EngineConfig Library
engine_config = EngineConfig(train_iters=args.max_train_steps, valid_step=num_update_steps_per_epoch,logger_type=args.log_name)#, distributed=args.distributed, roll_back=args.rollback

#set dependencies as dictionaries
#level 1(inner) accesses level 2(outer)
u2l = {outer_1:[inner_1]} # Green/optimized ,outer_:[inner_2]

#level 2(outer) accesses level 1(inner)

l2u = {inner_1:[inner_2],inner_2:[outer_1]} #red/unoptimized ,outer_2 ,outer_2:[outer_1],outer_1:[outer_2]

dependencies = {"l2u": l2u, "u2l": u2l}

engine = ReweightingEngine(config=engine_config, problems=problems, dependencies=dependencies)
print(args.max_train_steps)
engine.run()

attn_weight_list = []
source_examples = []
target_examples = []
rnn_model = engine.inner2.module
for step, batch in enumerate(rnn_dataloader_train):
    rnn_src_ids_1,rnn_trg_ids_1,attn_idxs = batch 
    attention_weights = engine.outer1.module[1](attn_idxs)
    source_sents = sp_src.decode(rnn_src_ids_1.detach().cpu().numpy().tolist())
    target_sents = sp_trg.decode(rnn_trg_ids_1.detach().cpu().numpy().tolist())
    source_sents = [ti.replace(" ⁇ ", "") for ti  in source_sents]
    target_sents = [ti.replace(" ⁇ ", "") for ti  in target_sents]
    source_examples.extend(source_sents)
    target_examples.extend(target_sents)
    attn_weight_list.extend(attention_weights.detach().cpu().numpy().tolist())

    
combined_list = list(zip(attn_weight_list, source_examples,target_examples))

# Sort the combined list in descending order
sorted_combined = sorted(combined_list, reverse=True)

# Now you can split this back into two lists using zip again
attn_weight_list, source_examples, target_examples = zip(*sorted_combined)

# save to the text file
with open('data_weights_'+args.data_name+'_'+args.src_lang+'.txt', 'w') as f:
    for i in range(len(attn_weight_list)):
        f.write("Attention weight:" + str(attn_weight_list[i]) + '     \t      ' + "Examples:" + source_examples[i] + '     \t      '+target_examples[i]+ '\n')

    
    
    
