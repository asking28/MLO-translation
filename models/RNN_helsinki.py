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
        


@dataclass(order=True)
class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        priority: int
        item: Any=field(compare=False)
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward
    
    
class Embedding_(torch.nn.Module):
    def __init__(self, embedding_layer):
        super(Embedding_, self).__init__()

        self.embedding = embedding_layer.cuda()

    def forward(self, mask):
#         print(mask)
        if mask.ndim == 2:
            assert mask.dtype == torch.long
            return self.embedding(mask)
        
        assert mask.dtype == torch.float
#         here the mask is the one-hot encoding
#         print("REACHED_EMBEDDING")
        return torch.matmul(mask, self.embedding.weight)
    
    
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        
        embedding = nn.Embedding(input_dim, emb_dim)
        self.embedding = Embedding_(embedding).requires_grad_()

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                src: Tensor) -> Tuple[Tensor]:

        embedded = self.dropout(self.embedding(src))

        outputs, hidden = self.rnn(embedded)

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 attn_dim: int):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim

        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tensor:

        src_len = encoder_outputs.shape[0]

        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((
            repeated_decoder_hidden,
            encoder_outputs),
            dim = 2)))

        attention = torch.sum(energy, dim=2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: int,
                 attention: nn.Module):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)


    def _weighted_encoder_rep(self,
                              decoder_hidden: Tensor,
                              encoder_outputs: Tensor) -> Tensor:

        a = self.attention(decoder_hidden, encoder_outputs)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted_encoder_rep = torch.bmm(a, encoder_outputs)

        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

        return weighted_encoder_rep


    def forward(self,
                input: Tensor,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tuple[Tensor]:

        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden,
                                                          encoder_outputs)
#         print(embedded.shape,weighted_encoder_rep.shape)
        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim = 2)

        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)

        output = self.out(torch.cat((output,
                                     weighted_encoder_rep,
                                     embedded), dim = 1))

        return output, decoder_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 device: torch.device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self,
                src: Tensor,
                trg: Tensor,
                teacher_forcing_ratio: float = 0.5) -> Tensor:

        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> token
        output = trg[0,:]

        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        return outputs
    
class RNN_MODEL(nn.Module):
    def __init__(self,criterion,rnn_tokenizer,MODEL=None):
        
        super(RNN_MODEL, self).__init__()
        INPUT_DIM = len(rnn_tokenizer)
        OUTPUT_DIM = len(rnn_tokenizer)
        ENC_EMB_DIM = 256
        DEC_EMB_DIM = 256
        ENC_HID_DIM = 512
        DEC_HID_DIM = 512
        ATTN_DIM = 128
        ENC_DROPOUT = 0.5
        DEC_DROPOUT = 0.5
        
        self.topk = 1  # how many sentence do you want to generate
        self.EOS_token = rnn_tokenizer.eos_token_id
        self.SOS_token = rnn_tokenizer.bos_token_id
        self.MAX_LENGTH = 50
        self.vocab_size = len(rnn_tokenizer)
        
        self.tokenizer = rnn_tokenizer
        self._criterion = criterion
#         self.tokenizer_english = en_tokenizer
#         self.tokenizer_german = de_tokenizer
#         self.vocab_english = en_vocab
#         self.vocab_german = de_vocab
        attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)

        enc  = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
        dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
        self.model = Seq2Seq(enc, dec, device).to(device)
        self.model.apply(init_weights)
        
        if MODEL is not None:
            self.model.load_state_dict(torch.load(MODEL))
        
#         self.enc = self.model.encoder
#         self.dec = self.model.decoder
        
    def forward(self,src,trg,teacher_forcing_ratio=0.5):
        try:
            src = src.permute(1,0)
        except:
            src = src.permute(1,0,2)
        trg = trg.permute(1,0)
#         print(src.shape)
#         print(trg.shape)
        output = self.model(src, trg, teacher_forcing_ratio)
        batch_size = trg.shape[1]
        output_dim = output.shape[-1]
        
        output_1 = output[1:].view(-1, output_dim)
        
        trg = trg[1:].reshape(-1)
#         print(output.shape)
#         print(trg.shape)
        loss_vec = self._criterion(output_1,trg)
        
        
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        return fun1(loss_vec,output)
    
    def loss(self,src,trg,teacher_forcing_ratio=0.5):
        loss_vec = self(src,trg,teacher_forcing_ratio).loss
        return loss_vec
    
    def get_loss_vec(self,src,trg,teacher_forcing_ratio=0.5):
#         output = self(src,trg,teacher_forcing_ratio).output
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
#         batch_size = trg.shape[1]
#         output_dim = output.shape[-1]
        
#         output = output[1:].view(-1, output_dim)
        
#         trg = trg[1:].reshape(-1)
# #         print(output.shape)
# #         print(trg.shape)
#         loss_vec = self._criterion(output,trg)
        
        
#         loss_vec = loss_vec.view(batch_size, -1).mean(dim = 1)
        loss_vec = self(src,trg,teacher_forcing_ratio).loss
        return loss_vec
        
        
    def generate(self,source,beam_width=1,temperature = 1.0, do_sample=False, repetition_penalty = 1.0,top_p = 0.9, top_k = 100):
        src = source
#         trg = batch[1]
        src = src.permute(1,0)
#         trg = trg.permute(1,0)
        batch_size = src.shape[1]
#         trg_len = trg.shape[0]
        trg_vocab_size = self.vocab_size # change to change target language

        #tensor to store decoder outputs
        outputs = torch.zeros(self.MAX_LENGTH, batch_size, trg_vocab_size).to(device)

        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.model.encoder(src)
        if beam_width>1:
            decoded_batch = []
            #     :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
            #     :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
            #     :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
            #     :return: decoded_batch
            # decoding goes sentence by sentence
            activation = torch.nn.LogSoftmax(dim=-1)
            for idx in range(batch_size):
                decoder_hidden = hidden[idx, :].unsqueeze(0)
                encoder_output = encoder_outputs[:,idx, :].unsqueeze(1)
                # Start with the start of the sentence token
                decoder_input = torch.LongTensor([[self.tokenizer.bos_token_id]]).cuda()
                endnodes = []
                number_required = min((self.topk + 1), self.topk - len(endnodes))
                # starting node -  hidden vector, previous node, word id, logp, length
                node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
                nodes = PriorityQueue()
                # start the queue
                nodes.put((-node.eval(), node))
                qsize = 1
                # start beam search
                while True:
                    # give up when decoding takes too long
                    if qsize > 2000: break
                    # fetch the best node
                    score, n = nodes.get()
                    decoder_input = n.wordid
                    decoder_input = decoder_input.squeeze(0)
                    decoder_hidden = n.h
                    if n.wordid.item() == self.tokenizer.eos_token_id and n.prevNode != None:
                        endnodes.append((score, n))
                        # if we reached maximum # of sentences required
                        if len(endnodes) >= number_required:
                            break
                        else:
                            continue
                    # decode for one step using decoder
                    decoder_output, decoder_hidden = self.model.decoder(decoder_input, decoder_hidden, encoder_output)
                    decoder_output = activation(decoder_output)
                    # PUT HERE REAL BEAM SEARCH OF TOP
                    log_prob, indexes = torch.topk(decoder_output, beam_width)
                    nextnodes = []
                    for new_k in range(beam_width):
                        decoded_t = indexes[0][new_k].view(1, -1)
                        log_p = log_prob[0][new_k].item()

                        node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                        score = -node.eval()
                        nextnodes.append((score, node))


                    # put them into queue
                    for i in range(len(nextnodes)):
                        score, nn = nextnodes[i]
                        nodes.put((score, nn))
                        # increase qsize
                    qsize += len(nextnodes) - 1
                # choose nbest paths, back trace them
                if len(endnodes) == 0:
                    endnodes = [nodes.get() for _ in range(self.topk)]
                utterances = []
                for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                    utterance = []
                    utterance.append(n.wordid)
                    # back trace
                    while n.prevNode != None:
                        n = n.prevNode
                        utterance.append(n.wordid)

                    utterance = utterance[::-1]
                    utterances.append(utterance)

                decoded_batch.append(utterances)

            l= torch.full((batch_size,self.MAX_LENGTH),self.tokenizer.pad_token_id)
            for i in range(len(decoded_batch)):
                l[i,:len(decoded_batch[i][0])]=torch.tensor([r[0].cpu().numpy()[0] for r in decoded_batch[i][0]][:self.MAX_LENGTH])
            return l
        else:
            if repetition_penalty == 1.0:
#                 decoder_hidden = hidden
#                 decoded_batch = torch.zeros((batch_size, self.MAX_LENGTH),dtype=torch.int32)
#                 decoder_input = torch.cuda.LongTensor([self.tokenizer.bos_token_id for _ in range(batch_size)])
    #             print(decoder_input.shape)
                decoder_hidden = hidden
                decoded_batch = torch.zeros((batch_size, self.MAX_LENGTH),dtype=torch.int32)
                decoder_input = torch.cuda.LongTensor([self.SOS_token for _ in range(batch_size)])
    #             print(decoder_input.shape)
                for t in range(self.MAX_LENGTH):
                    decoder_output, decoder_hidden = self.model.decoder(decoder_input, decoder_hidden, encoder_outputs)

                    topv, topi = decoder_output.data.topk(1)  # get candidates
                    topi = topi.view(-1)
                    decoded_batch[:, t] = topi
    #                 print(topi)

                    decoder_input = topi.detach().view(-1)
                            
#                 for t in range(self.MAX_LENGTH):
#                     decoder_output, decoder_hidden = self.model.decoder(decoder_input, decoder_hidden, encoder_outputs)

#                     topv, topi = decoder_output.data.topk(1)  # get candidates
#                     topi = topi.view(-1)
#                     decoded_batch[:, t] = topi
#     #                 print(topi)

#                     decoder_input = topi.detach().view(-1)
    #                 print(decoder_input.shape)
    #         print(decoded_batch)
            
            else:
                cur_len = 0
                # max_len = 10
                # repetition_penalty = 2.5
                # do_sample = True
                # temperature = 2
                # top_p = 0.9
                # top_k = 100
                unfinished_sents = src.new(batch_size).fill_(1).to(device)
                pad_token_id = self.tokenizer.pad_token_id
                eos_token_ids = [self.tokenizer.eos_token_id]

                decoder_hidden = hidden
                decoded_batch = torch.zeros((batch_size, self.MAX_LENGTH),dtype=torch.int32).to(device)
                decoder_input = torch.cuda.LongTensor([self.tokenizer.bos_token_id for _ in range(batch_size)])
                while cur_len < self.MAX_LENGTH:
                #     model_inputs = self.prepare_inputs_for_generation(input_ids, pasts=pasts)
                #     outputs = self(**model_inputs)
                    decoder_output, decoder_hidden = self.model.decoder(decoder_input, decoder_hidden, encoder_outputs)
                #     next_token_logits = outputs[:, -1, :]
                    next_token_logits = decoder_output
                    if repetition_penalty != 1.0:
                            for i in range(batch_size):
                                for previous_tokens in set(decoded_batch[i].tolist()):
                                    if next_token_logits[i, previous_tokens] < 0:
                                        next_token_logits[i, previous_tokens] *= repetition_penalty
                                    else:
                                        next_token_logits[i, previous_tokens] /= repetition_penalty
                    if do_sample:
                            # Temperature (higher temperature => more likely to sample low probability tokens)
                        if temperature > 0 and temperature != 1.0:
                            next_token_logits = next_token_logits / temperature
                        # Top-p/top-k filtering
                        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                        # Sample
                        next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).squeeze(1)
                    else:
                        # Greedy decoding
                        next_token = torch.argmax(next_token_logits, dim=-1)

                # update generations and finished sentences
                    tokens_to_add = next_token * unfinished_sents + pad_token_id * (1 - unfinished_sents)
                    tokens_to_add = tokens_to_add.to(device)
                #     decoded_batch = torch.cat([decoded_batch, tokens_to_add.unsqueeze(-1)], dim=-1)
                    decoded_batch[:,cur_len] = tokens_to_add #.unsqueeze(-1)
                    for eos_token_id in eos_token_ids:
                        unfinished_sents.mul_(tokens_to_add.ne(eos_token_id).long())
                    cur_len = cur_len + 1
                    if unfinished_sents.max() == 0:
#                         print("HERE")
                        break
                #     print(tokens_to_add)
                #     break
                    decoder_input = tokens_to_add
        return decoded_batch
    
    
    # new model for the definitions of gradients in architec.py
    def new(self):
        model_new = RNN_MODEL(self._criterion,self.tokenizer)
        model_new.model.load_state_dict(self.model.state_dict())
        return model_new
    
def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
            
def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits