# BART as ds model and to produce articles given summaries

import os
import random
import numpy as np
import copy
# from transformers import MBartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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


class MT(nn.Module):

    def __init__(self, criterion, tokenizer, MODEL = 'Helsinki-NLP/opus-mt-de-en'):
        super(MT, self).__init__()
        # few parameters needed to define the loss function and generate function
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id
        # loss type CrossEntropyLoss(ignore_index = self.pad_token_id)
        self._criterion = criterion
        self.model_name=MODEL
        # mBART model definition with encoder and the decoder
        self.mbart_model =  AutoModelForSeq2SeqLM.from_pretrained(MODEL).requires_grad_() #"./abhisingh-volume/french_english/helsinki_en_fr_model",local_files_only=True

        # print('Loading the pretrained model ....')
        # Load the pre-trained model trained for
        # self.mbart_model.load_state_dict(torch.load('pretrained_BART.pt')) #need to check requirement


        # print('Done! Loaded the pretrained model ....')

        self.encoder = self.mbart_model.model.encoder
        self.decoder = self.mbart_model.model.decoder

        # embedding layer for both encoder and decoder since it is shared
        self.embedding = Embedding_(self.encoder.embed_tokens).requires_grad_()
        self.enc_emb_scale = self.encoder.embed_scale

    def forward(self, input_ids, input_attn, target_ids = None, target_attn = None):

        # get the input embeddings of the encoder
        inp_emb = self.embedding(input_ids)/self.enc_emb_scale

        out = self.mbart_model(inputs_embeds=inp_emb, attention_mask = input_attn, labels = target_ids, decoder_attention_mask = target_attn, return_dict=True)

        return out

    def loss(self, input_ids, input_attn, target_ids, target_attn):

        output = self(input_ids, input_attn, target_ids, target_attn)

        return output.logits, output.loss

    def get_loss_vec(self, input_ids, input_attn, target_ids = None, target_attn = None):

        # batch size
        batch_size = target_ids.shape[0]

        # target_sequence_length of the model
        target_sequence_length = target_ids.shape[1]

        logits = (self(input_ids, input_attn, target_ids = target_ids, target_attn = target_attn)).logits

        loss_vec = self._criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        loss_vec = loss_vec.view(batch_size, -1).mean(dim = 1)

        return loss_vec

    # used for generation of summaries from articles
    def generate(self, input_ids, num_beams = 2, max_length=article_length):

        # beam search
        summary_ids = self.mbart_model.generate( input_ids = input_ids, num_beams = num_beams, early_stopping = True, max_length = max_length, no_repeat_ngram_size = 2, repetition_penalty = 1.2)

        ## sampling with top_p
        #summary_ids = self.bart_model.generate( input_ids = input_ids, num_beams = 1, max_length = max_length, top_p = 0.95, top_k = 50, no_repeat_ngram_size = 2, repetition_penalty = 1.2)

        return summary_ids

    # new model for the definitions of gradients in architec.py
    def new(self):

        # there is embedding layer and the summarization head that we will not train on
        # we just train on the encoder and the decoder weights
        model_new = mBART(self._criterion, self.tokenizer,MODEL=self.model_name).cuda()

        # hence we deep copy all the weights and update the required ones
        # use the first order approximation for the summarization head
        # i.e, do not use unrolled model w.r.t to the summarization head parameters
        model_new.mbart_model.load_state_dict(self.mbart_model.state_dict())

        return model_new
