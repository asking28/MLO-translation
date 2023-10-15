# Produce synthetic summaries to further produce synthetic articles

import os
import random
import numpy as np
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2LMHeadModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from .hyperparams import *

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(seed_)


class Embedding_(torch.nn.Module):
    def __init__(self, embedding_layer):
        super(Embedding_, self).__init__()
        
        self.embedding = embedding_layer.cuda()

    def forward(self, mask):
        if mask.ndim == 2:
            assert mask.dtype == torch.long
            return self.embedding(mask)
        
        assert mask.dtype == torch.float
        # here the mask is the one-hot encoding
        return torch.matmul(mask, self.embedding.weight)


class GPT(nn.Module):
    
    def __init__(self, criterion, tokenizer, max_length=summary_length, min_length=50, MODEL='distilgpt2'):
        super(GPT, self).__init__()
        # few parameters needed to define the loss function and generate function
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id
        # loss type CrossEntropyLoss(ignore_index = self.pad_token_id)
        self._criterion = criterion
        self.model_name=MODEL
        # maximum and minimum length for generation
        self.min_length=min_length
        self.max_length=max_length
        # GPT model definition with encoder and the decoder
        if MODEL=='distilgpt2':
            self.gpt_model = GPT2LMHeadModel.from_pretrained(MODEL)
        else:
            self.gpt_model = AutoModelForCausalLM.from_pretrained(MODEL)
            
        self.gpt_model.config.max_length=15

        self.gpt_model.config.pad_token_id = self.gpt_model.config.eos_token_id

        # embedding layer for both encoder and decoder since it is shared   
        self.embedding = Embedding_(self.gpt_model.transformer.wte).requires_grad_()

    def forward(self, input_ids, input_attn, target_ids = None, target_attn = None):

        # get the input embeddings of the encoder
        inp_emb = self.embedding(input_ids)

        out = self.gpt_model(inputs_embeds = inp_emb, attention_mask = input_attn, labels = input_ids, return_dict=True)

        return out

    def get_loss_vec(self, input_ids, input_attn, target_ids = None, target_attn = None):

        # batch size
        batch_size = target_ids.shape[0]
        
        # target_sequence_length of the model
        target_sequence_length = target_ids.shape[1]

        logits = (self(input_ids, input_attn, target_ids = target_ids, target_attn = target_attn)).logits

        shift_logits = logits[..., :-1, :].contiguous()
        
        shift_labels = target_ids[..., 1:].contiguous()

        loss_vec = self._criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))#flatten complete batch...[batch_size,seq_len,vocab_size]-->[batch_size*seq_len,vocab_size]...loss output-->(batch_size*seq_len)

        loss_vec = loss_vec.view(batch_size, -1).mean(dim = 1)  #convert loss to [batch_size,seq_len] then take mean across seq_len and return loss as a vector batch wise

        return loss_vec

    # used for generation of summaries from articles
    def generate(self, input_ids, num_beams = 2):
        
        # # beam search
        # summary_ids = self.gpt_model.generate( input_ids = input_ids, num_beams = num_beams, max_length = self.max_length, no_repeat_ngram_size = 2, repetition_penalty = 1.2)
        
        # # beam search
        # summary_ids = self.gpt_model.generate( input_ids = input_ids, num_beams = num_beams, early_stopping = True, max_length = self.max_length, no_repeat_ngram_size = 2, repetition_penalty = 1.2)
        
        # sampling with top_p
        summary_ids = self.gpt_model.generate( input_ids = input_ids, num_beams = 1, early_stopping = True, max_length = self.max_length, top_p = 0.95, top_k = 50, no_repeat_ngram_size = 2, repetition_penalty = 1.2, min_length=self.min_length)

        return summary_ids

    # new model for the definitions of gradients in architec.py 
    def new(self):

        # there is embedding layer and the summarization head that we will not train on 
        # we just train on the encoder and the decoder weights 
        model_new = GPT(self._criterion, self.tokenizer,MODEL=self.model_name).cuda()
        
        # hence we deep copy all the weights and update the required ones
        # use the first order approximation for the summarization head
        # i.e, do not use unrolled model w.r.t to the summarization head parameters
        model_new.gpt_model.load_state_dict(self.gpt_model.state_dict())
        
        return model_new

