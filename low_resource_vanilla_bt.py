import math
import io
import pickle
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
import operator, functools
from sacremoses import MosesPunctNormalizer, MosesTokenizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import evaluate
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
from datasets import concatenate_datasets
from tokenizers import Tokenizer
import datasets
from sacremoses import MosesPunctNormalizer, MosesTokenizer
# from data_set import *
from transformers import GPT2Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import get_linear_schedule_with_warmup

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

from tqdm import tqdm
import socket

import torch
import os
from betty.logging.logger_base import LoggerBase
import random
import sentencepiece as spm
try:
    import wandb
except ImportError:
    wandb = None
    
nltk.download('punkt')

try:
    import wandb
except ImportError:
    wandb = None
    
nltk.download('punkt')
parser = argparse.ArgumentParser("Machine Translation")
parser.add_argument('--src_lang', type=str, default='cor', help='lang name')
parser.add_argument('--trg_lang', type=str, default='eng', help='lang name')
parser.add_argument('--train_num_points', type=int, default=-1, help='train data size')
parser.add_argument('--train_num_points_back', type=int, default=10000, help='train data size back')
parser.add_argument('--val_num_points', type=int, default=-1, help='val data')
parser.add_argument('--transformer_model', action='store_true', help='Use transformer model')
parser.add_argument('--weight_decay', type=int, default=0, help='Weight decay')
parser.add_argument('--epochs', type=int, default=30, help='n epochs')
parser.add_argument('--epochs_back', type=int, default=20, help='back epochs')
parser.add_argument('--optimizer_back', type=str, default='adam', help='back translation optimizer')
parser.add_argument('--lr_back', type=float, default=2e-5, help='back translation lr')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation')
parser.add_argument('--back_model', type=str, default='opus', help='back translation model')
parser.add_argument('--no_augmentation', action='store_true', help='No augmantation')
parser.add_argument('--only_aug', action='store_true', help='Only augmentation')
parser.add_argument('--train_seq', action='store_true', help='Sequential training')
parser.add_argument('--only_generate', action='store_true', help='Only generation')
parser.add_argument('--preprocessing_num_workers', type=int, default=1, help='Workers')
parser.add_argument('--epochs_on_mono', type=int, default=5, help='back epochs training rnn')
parser.add_argument('--bt', action='store_true', help='Add Back translated data')
parser.add_argument('--baseline', type=str, default=None, help='Baseline')
parser.add_argument('--baseline_par', type=float, default=None, help='Baseline parameters')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
parser.add_argument('--max_len', type=int, default=128, help='Max length')
# parser.add_argument('--train_num_points_back', type=int, default=10000, help='BT size')
parser.add_argument('--create_bt_data', action='store_true', help='Create Data')
parser.add_argument('--tagged', action='store_true', help='Tagged')
parser.add_argument('--combine', action='store_true', help='Combine Data')
parser.add_argument('--run_all_baselines', action='store_true', help='Combine all baselines')
parser.add_argument('--create_path', type=str, default=None, help='Data create Source file name')
parser.add_argument('--data_name', type=str, default='flores', help='Data name')
parser.add_argument('--max_gen', type=int, default=50000, help='Max gen data')
# epochs_on_mono

class Args():
    def __init__(self):
        self.translation_language = 'fr'
        self.train_num_points = -1
        self.val_num_points = -1
        self.transformer_model = False
        self.weight_decay=0
        self.epochs = 18
        self.epochs_back = 18
        self.gradient_accumulation_steps = 1
args = parser.parse_args()
args.exp_name = 'Low resource'+'_'+args.src_lang+'_'+args.trg_lang+'_'+str(args.train_num_points)
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
random.seed(42)
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
    # Set the random seed for reproducibility
#     random.shuffle(train_data)
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


if args.bt:
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
        src_lines = src_lines[:args.train_num_points_back]
        trg_lines = trg_lines[:args.train_num_points_back]
            
        return src_lines,trg_lines
    bt_data = read_bt_data()
    src_train_data = train_data[0]
    trg_train_data = train_data[1]
    src_train_data.extend(bt_data[0][:args.train_num_points_back])
    trg_train_data.extend(bt_data[1][:args.train_num_points_back])
    # Combine the lists into pairs using zip()
    combined_lists = list(zip(src_train_data, trg_train_data))

    # Shuffle the combined lists
    random.shuffle(combined_lists)

    # Unpack the shuffled pairs into separate lists
    src_train_data, trg_train_data = zip(*combined_lists)
    train_data = (src_train_data,trg_train_data)

def load_tokenizer(src_lang=args.src_lang,trg_lang=args.trg_lang,n_src= '4k',n_trg='4k'):
    sp_src = spm.SentencePieceProcessor(model_file='upload_datasets/'+args.src_lang+'_'+args.trg_lang+'/'+'opusTC.'+args.src_lang+'.'+n_src+'.spm')
    sp_trg = spm.SentencePieceProcessor(model_file='upload_datasets/'+args.src_lang+'_'+args.trg_lang+'/'+'opusTC.'+args.trg_lang+'.'+n_trg+'.spm')
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
    for  raw_src,raw_trg in zip(data[0],data[1]):
        rnn_src = sp_src.encode(raw_src)[:args.max_len]
        rnn_trg = sp_trg.encode(raw_trg)[:args.max_len]
        data_process.append((rnn_src,rnn_trg))
    return data_process

train_tokens = data_process_train(train_data)
dev_tokens = data_process_train(dev_data)
test_tokens = data_process_train(test_data)
if args.create_bt_data:
    train_tokens.extend(dev_tokens)
def generate_batch(data_batch):
    rnn_src_ids, rnn_trg_ids = [],[]
    for (src_id,trg_id) in data_batch:
        rnn_src_ids.append(torch.tensor(src_id,dtype=torch.int64))
        rnn_trg_ids.append(torch.tensor(trg_id,dtype=torch.int64))
    rnn_src_ids = pad_sequence(rnn_src_ids, batch_first=True, padding_value=0)
    rnn_trg_ids = pad_sequence(rnn_trg_ids, batch_first=True, padding_value=0)    
    return rnn_src_ids,rnn_trg_ids


batch_size = args.batch_size

rnn_dataloader_train = DataLoader(train_tokens,batch_size=batch_size,
                        shuffle=False, collate_fn=generate_batch)
rnn_dataloader_val = DataLoader(dev_tokens,batch_size=batch_size,
                        shuffle=False, collate_fn=generate_batch)
rnn_dataloader_test = DataLoader(test_tokens,batch_size=batch_size,
                        shuffle=False, collate_fn=generate_batch)

def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def create_rnn_model(src_tok,trg_tok):
    PAD_IDX = 0
    rnn_criterion=torch.nn.CrossEntropyLoss(ignore_index =PAD_IDX,reduction='mean')# reduction='mean'
    rnn_criterion=rnn_criterion.cuda()
    if args.transformer_model:
        rnn_model = TRANSFORMER_MODEL(rnn_criterion,src_tok,trg_tok) 
    else:
        rnn_model = RNN_MODEL(rnn_criterion,src_tok,trg_tok) 

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
from tqdm import tqdm
def train_target_to_source(model,data_loader,optimizer,rnn_scheduler,epochs):
    for t in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        for j,batch in enumerate(tqdm(data_loader)):
            optimizer.zero_grad()
            mbart_src_ids,mbart_trg_ids = batch
            mbart_src_ids,mbart_trg_ids = mbart_src_ids.to(device),mbart_trg_ids.to(device)
            
            loss = model.loss(mbart_trg_ids,mbart_src_ids)
            train_loss+=loss.item()
            loss.backward()
            optimizer.step()
            rnn_scheduler.step()
        d = {'loss':train_loss/j}
#         wand_log.log(d,tag = 'target_to_source')

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
if args.create_bt_data:
    rnn_model_target_to_source = create_rnn_model(sp_trg,sp_src).to(device)
    mbart_params = rnn_model_target_to_source.parameters()
    # mbart_params = group_parameters(rnn_model_target_to_source)
    # if args.optimizer_back == 'adam':
    rnn_optimizer_target_to_source = optim.AdamW(mbart_params)

    num_update_steps_per_epoch = math.ceil(len(rnn_dataloader_train))
    max_train_steps = num_update_steps_per_epoch*args.epochs
    warmup_steps = 0.1*max_train_steps
    rnn_scheduler_target_to_source = get_linear_schedule_with_warmup(
                    rnn_optimizer_target_to_source, num_warmup_steps=warmup_steps, num_training_steps=max_train_steps
                )
    train_target_to_source(rnn_model_target_to_source,rnn_dataloader_train,rnn_optimizer_target_to_source,rnn_scheduler_target_to_source,20)#model,data_loader,optimizer,rnn_scheduler,epochs
    with open('upload_datasets/'+args.src_lang+'_'+args.trg_lang+'/'+args.create_path, 'r') as file:
        # Read the file line by line
        trg_lines = file.readlines()
    if len(trg_lines)>args.max_gen:
        trg_lines = trg_lines[:args.max_gen]
    for i in range(len(trg_lines)):
        trg_lines[i] = trg_lines[i].strip()
    def data_process_gen(data):    
        data_process = []
        for  raw_trg in trg_lines:
            rnn_trg = sp_trg.encode(raw_trg)
            data_process.append((rnn_trg))
        return data_process
       
    gen_tokens = data_process_gen(trg_lines) 
    
    def generate_batch_gen(data_batch):
        rnn_trg_ids = []
        for trg_id in data_batch:
            rnn_trg_ids.append(torch.tensor(trg_id,dtype=torch.int64))
        rnn_trg_ids = pad_sequence(rnn_trg_ids, batch_first=True, padding_value=0)    
        return rnn_trg_ids
    
    rnn_dataloader_gen = DataLoader(gen_tokens,batch_size=batch_size,
                        shuffle=False, collate_fn=generate_batch_gen)
    source_sents_l = []
    target_sents_l = []
    for i, batch in enumerate(rnn_dataloader_gen):
        trans_sentences = []
        labels_sentences = []
        src = batch.to(device)
#         print(src.shape)
        pred_sents = rnn_model_target_to_source.generate(src,beam_width=beam_width,temperature=temperature , do_sample=do_sample, repetition_penalty=repetition_penalty ,top_p=top_p, top_k=top_k)
        pred_sents = sp_src.decode(pred_sents.detach().cpu().numpy().tolist()) #deu
        source_sents = sp_trg.decode(src.detach().cpu().numpy().tolist()) #ido
        if args.tagged:
            pred_sents = ['BT '+ ti.replace(" ⁇ ", "") for ti  in pred_sents] #deu
            source_sents = ['BT '+ ti.replace(" ⁇ ", "") for ti  in source_sents] #ido
        else:
            pred_sents = [ti.replace(" ⁇ ", "") for ti  in pred_sents] #deu
            source_sents = [ti.replace(" ⁇ ", "") for ti  in source_sents] #ido
        
        source_sents_l.extend(source_sents)
        target_sents_l.extend(pred_sents)
    print(source_sents_l[:5])
    print(target_sents_l[:5])
    # Open a file for writing
    if args.data_name=='multi30':
        with open('upload_datasets/'+args.src_lang+'_'+args.trg_lang+'/wiki.aa.'+args.src_lang+'-'+args.trg_lang+'.'+args.src_lang+'.'+str(args.train_num_points), 'w') as f:
        # Write each element of the list to the file
            for item in target_sents_l:
                f.write("%s\n" % item)
        with open('upload_datasets/'+args.src_lang+'_'+args.trg_lang+'/wiki.aa.'+args.src_lang+'-'+args.trg_lang+'.'+args.trg_lang+'.'+str(args.train_num_points), 'w') as f:
            # Write each element of the list to the file
            for item in source_sents_l:
                f.write("%s\n" % item)
                
    else:
        with open('upload_datasets/'+args.src_lang+'_'+args.trg_lang+'/wiki.aa.'+args.src_lang+'-'+args.trg_lang+'.'+args.src_lang, 'w') as f:
            # Write each element of the list to the file
            for item in target_sents_l:
                f.write("%s\n" % item)
        with open('upload_datasets/'+args.src_lang+'_'+args.trg_lang+'/wiki.aa.'+args.src_lang+'-'+args.trg_lang+'.'+args.trg_lang, 'w') as f:
            # Write each element of the list to the file
            for item in source_sents_l:
                f.write("%s\n" % item)
            
def find_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
        
rnn_model_source_to_target = create_rnn_model(sp_src,sp_trg).to(device)
mbart_params = rnn_model_source_to_target.parameters()
print(find_size(rnn_model_source_to_target))
# mbart_params = group_parameters(rnn_model_target_to_source)
# if args.optimizer_back == 'adam':
rnn_optimizer_source_to_target = optim.AdamW(mbart_params)

num_update_steps_per_epoch = math.ceil(len(rnn_dataloader_train))
max_train_steps = num_update_steps_per_epoch*args.epochs
warmup_steps = 0.1*max_train_steps
rnn_scheduler_source_to_target = get_linear_schedule_with_warmup(
                rnn_optimizer_source_to_target, num_warmup_steps=warmup_steps, num_training_steps=max_train_steps
            )


class WandBLogger(LoggerBase):
    def __init__(self,name=None):
#         try:
#             wandb.init(project="betty", entity=socket.gethostname())
#         except:
        os.environ['WANDB_API_KEY']='c0f0a59b4bd9dae783b5aa458996ebf0386e16de'
        if name==None:
            wandb.init(project="betty")
        else:
            wandb.init(project="betty")

    def log(self, stats, tag=None, step=None):
        """
        Log metrics/stats to Weight & Biases (wandb) logger

        :param stats: Dictoinary of values and their names to be recorded
        :type stats: dict
        :param tag:  Data identifier
        :type tag: str, optional
        :param step: step value associated with ``stats`` to record
        :type step: int, optional
        """
        if stats is None:
            return
        for key, value in stats.items():
            prefix = "" if tag is None else tag + "/"
            full_key = prefix + key
            if torch.is_tensor(value):
                value = value.item()
            wandb.log({full_key: value})
    def complete(self):
        wandb.finish()
wand_log = WandBLogger(name=args.exp_name)

val_steps = 2*num_update_steps_per_epoch



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
            
#             pred_sents = sp_trg.decode(pred_sents.detach().cpu().numpy().tolist())
#             given_sents = sp_trg.decode(trg.detach().cpu().numpy().tolist())
#             source_sents = sp_src.decode(src.detach().cpu().numpy().tolist())


            pred_sents = [ti.replace(" ⁇ ", "") for ti  in pred_sents]
            given_sents_rest = [ti.replace(" ⁇ ", "") for ti  in given_sents]
            source_sents = [ti.replace(" ⁇ ", "") for ti  in source_sents]
            given_sents_bleu = [[ti.replace(" ⁇ ", "")] for ti  in given_sents]
            
#             bleu_metric.add_batch(predictions=pred_sents, references=given_sents_bleu)
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

place_token = '<pad>'
class Baselines():
    def __init__(self):
        self.bls = ['swap', 'dropout','blank','smooth']
        
    def convert_source(self,de_ids,par=None,en_ids=None):
        if args.baseline == 'swap':
            if par!=None:
                return self.swap(de_ids,int(par))
            return self.swap(de_ids)
        if args.baseline == 'dropout':
            if par!=None:
                return self.dropout(de_ids)
            return self.dropout(de_ids)
        if args.baseline == 'blank':
            if par!=None:
                return self.blank(de_ids,par)
            return self.blank(de_ids)
        if args.baseline == 'smooth':
            dis_de = self.get_unigram_distribution(de_ids)
            dis_en = self.get_unigram_distribution(en_ids)
            if par!=None:
                dis_de = self.smooth(de_ids, dis_de, sample_p = par)
                dis_en = self.smooth(en_ids, dis_en, sample_p = par)
            else:
                dis_de = self.smooth(de_ids, dis_de, sample_p = 0.15)
                dis_en = self.smooth(en_ids, dis_en, sample_p = 0.15)
            return dis_de,dis_en
            
        
    def swap(self,de_ids, window_size = 10): # pass dataloader # Window size 3

#         for step, batch in enumerate(dataloader): 

            # batch dim (batch_size, seq_len)

        len_batch = de_ids.shape[-1] # length of a sequence

        start_idx = randrange(len_batch - window_size) # the start index of the window

        order = np.random.permutation(np.arange(start_idx, start_idx + window_size, 1, dtype=int)) # set the order randomly

#         de_ids = torch.cat((de_ids[:,:start_idx],de_ids[:,order],de_ids[:,start_idx+window_size:]),axis=-1) # swap the tokens as per the order
        de_ids[:,start_idx:start_idx+window_size] = de_ids[:,order]
        return de_ids
    
    # Dropout
    def dropout(self,de_ids, sample_p = 0.15): # the probability of a word dropped

#         for step, de in enumerate(de_ids): 

            # batch dim (batch_size, seq_len)

        N = de_ids.shape[-1] # length of a sequence

        # select indices
        indices = random.sample(range(N), int(np.ceil(sample_p*N))) # select only p% of words

        rem_indices = list(set(np.arange(N)) - set(indices))

        de_ids = de_ids[:,rem_indices] # drop indices and just keep the remaining indices

        return de_ids # return the altered dataloader
    
    # smooth
    def get_unigram_distribution(self,ids, sample_p = 0.15): # the probability of a word replaced with a sample from the unigram frequence distribution

        # estimate the unigram distribution

        source_language_list = []

#         target_language_list = []

#         for step, [source_batch, target_batch] in enumerate(dataloader):

        source_language_list = source_language_list + ((ids.view(-1)).tolist())

#             target_language_list = target_language_list + ((target_batch.view(-1)).tolist())

        source_distribution, _ = np.histogram(source_language_list, bins = np.arange(max(set(source_language_list))+2), density = True)

#         target_distribution, _ = np.histogram(target_language_list, bins = np.arange(max(set(target_language_list))+2), density = True)

        return source_distribution

    def smooth(self,ids, distribution, sample_p = 0.15): #pass the distribution and the corresponding dataloader
        # for source or target
#         for step, batch in enumerate(dataloader): 

            # batch dim (batch_size, seq_len)

        N = ids.shape[-1] # length of a sequence

        # select indices
        indices = random.sample(range(N), int(np.ceil(sample_p*N))) # select only p% of words

        sample_tokens = np.random.choice(np.arange(len(distribution)), (ids[:,indices]).shape, p=distribution)

        ids[:,indices] = torch.tensor(sample_tokens).to(device) # replace the tokens with the place holder token

        return ids # return the altered dataloader 
    
        # Blank
    def blank(self,ids, place_holder_token = place_token, sample_p = 0.15): # the probability of a word replaced with a place holder

#         for step, batch in enumerate(dataloader): 

            # batch dim (batch_size, seq_len)

        N = ids.shape[-1] # length of a sequence

        # select indices
        indices = random.sample(range(N), int(np.ceil(sample_p*N))) # select only p% of words

        ids[:,indices] = place_holder_token # replace the tokens with the place holder token

        return ids # return the altered dataloader  
    # 


baselines = Baselines()
def train_source_to_target(model,data_loader,data_loader_val,optimizer,rnn_scheduler,epochs):
    i=0
    l1,l2,l3 = [],[],[]
    for t in tqdm(range(epochs)):
        
        model.train()
        train_loss = 0.0
        for j,batch in enumerate(tqdm(data_loader)):
            i+=1
            optimizer.zero_grad()
            mbart_src_ids,mbart_trg_ids = batch
            mbart_src_ids,mbart_trg_ids = mbart_src_ids.to(device),mbart_trg_ids.to(device)
            if args.run_all_baselines:
                mbart_src_ids_1,mbart_trg_ids_1 = copy.deepcopy(mbart_src_ids), copy.deepcopy(mbart_trg_ids)
                mbart_src_ids_2,mbart_trg_ids_2 = copy.deepcopy(mbart_src_ids), copy.deepcopy(mbart_trg_ids)
                mbart_src_ids_3,mbart_trg_ids_3 = copy.deepcopy(mbart_src_ids), copy.deepcopy(mbart_trg_ids)
                loss_tr = model.loss(mbart_src_ids_1, mbart_trg_ids_1)
                args.baseline = 'smooth'
                mbart_src_ids,mbart_trg_ids = baselines.convert_source(mbart_src_ids,0.3,mbart_trg_ids)
                l4 = model.loss(mbart_src_ids, mbart_trg_ids)
                
                args.baseline = 'swap'
                mbart_src_ids_2 = baselines.convert_source(mbart_src_ids_2,6)
                l5 = model.loss(mbart_src_ids_2, mbart_trg_ids_2)
                
                args.baseline = 'dropout'
                mbart_src_ids_3 = baselines.convert_source(mbart_src_ids_3,0.3)
                l6 = model.loss(mbart_src_ids_3, mbart_trg_ids_3)
                
                

                l = l4+ l5+ l6 +loss_tr
                loss = l
            elif args.baseline in baselines.bls:
                mbart_src_ids_1,mbart_trg_ids_1 = copy.deepcopy(mbart_src_ids), copy.deepcopy(mbart_trg_ids)
                loss_tr = model.loss(mbart_src_ids_1, mbart_trg_ids_1)
                if args.baseline=='smooth':
                    mbart_src_ids,mbart_trg_ids = baselines.convert_source(mbart_src_ids,args.baseline_par,mbart_trg_ids)
                else:
                    mbart_src_ids = baselines.convert_source(mbart_src_ids,args.baseline_par)

                l = model.loss(mbart_src_ids, mbart_trg_ids)+loss_tr
                loss = l
#             elif args.back_model=='rnn':
            else:
                loss = model.loss(mbart_src_ids,mbart_trg_ids)
            train_loss+=loss.item()
            loss.backward()
            optimizer.step()
            rnn_scheduler.step()
        d = {'loss':train_loss/j}
        wand_log.log(d,tag = 'source_to_target')
        l1.append(d)
        
        if i%val_steps==0:
            model.eval()
            val_losses = 0.0
            for j,batch in enumerate(data_loader_val):
                mbart_src_ids,mbart_trg_ids = batch
                mbart_src_ids,mbart_trg_ids = mbart_src_ids.to(device),mbart_trg_ids.to(device)
                loss = model.loss(mbart_src_ids,mbart_trg_ids)

                val_losses+=loss
            print(val_losses/(j+1))
            d = {'loss':val_losses/(j+1)}
            l2.append(d)
            test_meteor,chrf_1,chrf_2 = valid_generate(rnn_model_source_to_target,rnn_dataloader_test, beam_width=beam_width,temperature = temperature, do_sample=do_sample, repetition_penalty = repetition_penalty,top_p = top_p, top_k = top_k)
            test_d = {'meteor':test_meteor,'chrf_1':chrf_1,'chrf_2':chrf_2}
            
            val_meteor,chrf_1_val,chrf_2_val = valid_generate(rnn_model_source_to_target,data_loader_val, beam_width=beam_width,temperature = temperature, do_sample=do_sample, repetition_penalty = repetition_penalty,top_p = top_p, top_k = top_k)
            val_d = {'meteor':val_meteor,'chrf_1':chrf_1_val,'chrf_2':chrf_2_val}
            
            wand_log.log(test_d,tag = 'test')
            wand_log.log(val_d,tag = 'val')
            l3.append(test_d)
            model.train()
        
    print(l1)
    print(l2)
    print(l3)


train_source_to_target(rnn_model_source_to_target,rnn_dataloader_train,rnn_dataloader_val,rnn_optimizer_source_to_target,rnn_scheduler_source_to_target,20)



