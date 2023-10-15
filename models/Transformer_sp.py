from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)
from torch import nn
from torch import Tensor
import math
import operator, functools
from queue import PriorityQueue
import numpy as np
import random
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import Tensor
import random
from dataclasses import dataclass, field
from typing import Any
from typing import Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class fun1():
    def __init__(self,loss,outputs):
        self.loss= loss
        self.output = outputs

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(0),:])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, mask: Tensor):
#         return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
        
        if mask.ndim == 2:
            assert mask.dtype == torch.long
            return self.embedding(mask.long()) * math.sqrt(self.emb_size)
        
        assert mask.dtype == torch.float
#         here the mask is the one-hot encoding
#         print("REACHED_EMBEDDING")
        return torch.matmul(mask, self.embedding.weight) * math.sqrt(self.emb_size)



# https://torchtutorialstaging.z5.web.core.windows.net/beginner/translation_transformer.html
# https://pytorch.org/tutorials/beginner/translation_transformer.html

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                 emb_size: int, src_vocab_size: int, tgt_vocab_size: int,
                 dim_feedforward:int = 512,NHEAD = 3, dropout:float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor,
                tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                        tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer_encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer_decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)



class TRANSFORMER_MODEL(nn.Module):
    def __init__(self,criterion,src_tokenizer,trg_tokenizer,max_length=128,MODEL=None):
        super(TRANSFORMER_MODEL, self).__init__()
        SRC_VOCAB_SIZE = len(src_tokenizer)
        TGT_VOCAB_SIZE = len(trg_tokenizer)
        self.MAX_LENGTH = max_length
        self.PAD_IDX = 0
        self.BOS_IDX = src_tokenizer.bos_id()
        self.EOS_IDX = src_tokenizer.eos_id()
        self.vocab_size = len(trg_tokenizer)
        self._criterion = criterion
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        INPUT_DIM = len(src_tokenizer)
        self.input_dim = INPUT_DIM
#         EMB_SIZE = 256
#         NHEAD = 4
#         FFN_HID_DIM = 256
#         NUM_ENCODER_LAYERS = 2
#         NUM_DECODER_LAYERS = 2
        
        EMB_SIZE = 256
        NHEAD = 4
        FFN_HID_DIM = 256
        BATCH_SIZE = 128
        NUM_ENCODER_LAYERS = 3
        NUM_DECODER_LAYERS = 3
        self.tokenizer = trg_tokenizer
#         EMB_SIZE = 512
#         NHEAD = 8
#         FFN_HID_DIM = 512
#         NUM_ENCODER_LAYERS = 3
#         NUM_DECODER_LAYERS = 3

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                                         EMB_SIZE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                                         FFN_HID_DIM,NHEAD)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.model = self.model.to(device)
        
    def forward(self,src,trg,teacher_forcing_ratio=0.5):
        src = src.to(device)
        trg = trg.to(device)
        try:
            src = src.permute(1,0)
        except:
            src = src.permute(1,0,2)
        trg = trg.permute(1,0)
        tgt_input = trg[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt_input)
        
        logits = self.model(src, tgt_input, src_mask, tgt_mask,
                                src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = trg[1:,:]
        loss = self._criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        return fun1(loss,logits)
    

    def loss(self,src,trg,teacher_forcing_ratio=0.5):
        loss_vec = self(src,trg,teacher_forcing_ratio).loss
        return loss_vec
    
    def get_loss_vec(self,src,trg,teacher_forcing_ratio=0.5):

        t = self(src,trg,teacher_forcing_ratio)
#         trg = trg.permute(1,0)
        loss_vec = self._criterion(t.output.permute(1,2,0),trg[:,:-1])
        return loss_vec
    
    def generate_square_subsequent_mask(self,sz):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self,src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
        if src.ndim == 2:
            src_padding_mask = (src == self.PAD_IDX).transpose(0, 1)
        else:
            _,indx = torch.max(src,dim=2)
            src_padding_mask = (indx == self.PAD_IDX).transpose(0, 1)
            
        tgt_padding_mask = (tgt == self.PAD_IDX).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
    
    def generate(self, source,beam_width=1,temperature = 1.0, do_sample=False, repetition_penalty = 1.0,top_p = 0.9, top_k = 100):
        start_symbol = self.BOS_IDX
        source = source.to(device)
        num_tokens = source.shape[1]
        batch_size = source.shape[0]
        source_mask = (torch.zeros(batch_size, num_tokens, num_tokens)).type(torch.bool)
        source_mask = source_mask.to(device)
        batch_size = source.shape[0]
        max_len = self.MAX_LENGTH
        ys_list = []
        for j in range(batch_size):
            src = source[j,:].unsqueeze(1)
            src_mask = source_mask[j,:,:]
            memory = self.model.encode(src, src_mask)
            ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)

            for i in range(max_len-1):
                memory = memory.to(device)
                memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
                tgt_mask = (self.generate_square_subsequent_mask(ys.size(0))
                                            .type(torch.bool)).to(device)
                out = self.model.decode(ys, memory, tgt_mask)
                out = out.transpose(0, 1)
                prob = self.model.generator(out[:, -1])
                _, next_word = torch.max(prob, dim = 1)
                next_word = next_word.item()

                ys = torch.cat([ys,
                                torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
                if next_word == self.EOS_IDX:
                    break
            ys_list.append(ys.flatten())
        return ys_list
    