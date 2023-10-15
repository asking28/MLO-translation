import os
import gc
import random
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from .utils import *
from .hyperparams import *
import logging
import torch.nn.functional as F
print(summary_length,article_length)
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(seed_)
NORM_FACTOR = 10000 # the normalization factor for the GPT/BART loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# the loss for the encoder and the decoder model 
# this takes into account the attention for all the datapoints for the encoder-decoder model
def CTG_loss(input_ids, input_attn, target_ids, target_attn, attn_idx, attention_parameters, model):
    
    attention_weights = attention_parameters(attn_idx)
    
    # similar to the loss defined in the BART model hugging face conditional text generation
#     print(attention_weights)
    # probability predictions
    loss_vec = model.get_loss_vec(input_ids, input_attn, target_ids = target_ids, target_attn = target_attn)
    
    loss = torch.dot(attention_weights, loss_vec)#.div(input_ids.shape[0])
    
    scaling_factor = 1/torch.sum(attention_weights)
    l = scaling_factor*loss
#     print('Yaha hoon')
#     print(loss_vec)
#     print(loss)
#     print(l)
#     print(attn_idx)
#     print(attention_weights)
#     if loss.isnan():
#         print(input_ids)
#         print(target_ids)
#         print(model)
    
    return l#torch.mean(loss_vec)
def normalize_gen(gen_sentences):
    lines=[]
    for line in gen_sentences:
        line = line.replace("▁"," ")
        lines.append(line)
    return lines
#RNN LOSS
def tokenize_rnn_en(sentences,rnn_model):
    en_tokenizer = rnn_model.tokenizer_english
    en_vocab = rnn_model.vocab_english
    pad_id = en_vocab['<pad>']
#     print(pad_id)
    tokenized = []
    for sent in sentences:
#         print(en_tokenizer(sent))
        en_tensor_ = torch.tensor([en_vocab['<bos>']] + [en_vocab[token] for token in en_tokenizer(sent)] + [en_vocab['<eos>']],dtype=torch.long)
        tokenized.append(en_tensor_)
    en_ids = pad_sequence(tokenized, batch_first = True, padding_value = pad_id )

    return en_ids

def tokenize_rnn_de(sentences,rnn_model):
    de_tokenizer = rnn_model.tokenizer_german
    de_vocab = rnn_model.vocab_german
    pad_id = de_vocab['<pad>']
#     print(pad_id)
    tokenized = []
    for sent in sentences:
#         print(de_tokenizer(sent))
        de_tensor_ = torch.tensor([de_vocab['<bos>']] + [de_vocab[token] for token in de_tokenizer(sent)] + [de_vocab['<eos>']],dtype=torch.long)
        tokenized.append(de_tensor_)
    de_ids = pad_sequence(tokenized, batch_first = True, padding_value = pad_id )

    return de_ids

def tokenize_rnn(sentences,rnn_model):
    tokenizer = rnn_model.tokenizer
    pad_id = tokenizer.pad_token_id
#     print(pad_id)
    un_tokenized = []
    sos = tokenizer.bos_token
    for sent in sentences:
        sent = sos+" "+sent
        un_tokenized.append(sent)
#         print(de_tokenizer(sent))
#         de_tensor_ = torch.tensor([de_vocab['<bos>']] + [de_vocab[token] for token in de_tokenizer(sent)] + [de_vocab['<eos>']],dtype=torch.long)
#         tokenized.append(de_tensor_)
    tokenized = tokenizer(un_tokenized,padding=False,truncation=True,max_length=50)['input_ids']
    tokenized = [torch.tensor(t) for t in tokenized]
    
    ids = pad_sequence(tokenized, batch_first = True, padding_value = pad_id )

    return ids

def tokenize_transformer(sentences,rnn_model):
    tokenizer = rnn_model.tokenizer
    tokenized = []
    for sent in sentences:
#         print(de_tokenizer(sent))
        de_tensor_ = torch.tensor(tokenizer.encode(sent,add_special_tokens=True).ids,dtype=torch.long)
        tokenized.append(de_tensor_)
    de_ids = pad_sequence(tokenized, batch_first = True, padding_value = tokenizer.token_to_id('<pad>') )

    return de_ids

def generate_aug_data(target_ids,summary_ids, summary_attn, gpt_model, bart_model, DS_model):
        #######################################################################################################
    eng_sents = gpt_model.tokenizer.batch_decode(summary_ids,skip_special_tokens=True)
    
    bart_summary_ids, bart_summary_attn = tokenize(eng_sents, bart_model.tokenizer, max_length = summary_length)
    bart_summary_ids, bart_summary_attn = bart_summary_ids.cuda(), bart_summary_attn.cuda()
    article_ids = bart_model.generate(bart_summary_ids)
    

    sents = DS_model.tokenizer.batch_decode(target_ids,skip_special_tokens=True)
    print("Original Target Sentences")
    print(normalize_gen(sents[:10]))
    print("Original English Sentences")
    print(normalize_gen(eng_sents[:10]))
    #######################################################################################################
    ### GPT convert
    # convert input to the bart encodings
    # # use the generate approach
    gpt_summary_ids = gpt_model.generate(summary_ids[:,:5])
    
    gpt_logits = gpt_model(gpt_summary_ids, torch.ones_like(gpt_summary_ids).long(), target_ids = gpt_summary_ids , target_attn = torch.ones_like(gpt_summary_ids).long()).logits

    # find the decoded vector from probabilities
    
    gpt_soft_idx, gpt_idx = torch.max(gpt_logits, dim=-1, keepdims= True)

    #######################################################################################################
    
    # convert to the bart ids
    
    gpt2bart_summary_ids, gpt2bart_summary_attn = tokenize(gpt_model.tokenizer.batch_decode(gpt_summary_ids,skip_special_tokens=True), bart_model.tokenizer, max_length = summary_length)
    ### T-5 specific
#     sents = gpt_model.tokenizer.batch_decode(gpt_summary_ids,skip_special_tokens=True)
#     for i in range(len(sents)):
#         sents[i] = 'translate English to German: '+sents[i]
        
#     gpt2bart_summary_ids, gpt2bart_summary_attn = tokenize(sents,bart_model.tokenizer, max_length = summary_length)
    
    
    gpt2bart_summary_ids, gpt2bart_summary_attn = gpt2bart_summary_ids.cuda(), gpt2bart_summary_attn.cuda()



    # BART
    bart_article_ids = bart_model.generate(gpt2bart_summary_ids)
    

    sents = bart_model.tokenizer.batch_decode(bart_article_ids,skip_special_tokens=True)
#     sents = ["<bos> "+t+ " <eos>" for t in sents]
    print("Augmented Target language Sentence")
    print(normalize_gen(sents[:10]))


    


    
    sents = gpt_model.tokenizer.batch_decode(gpt_summary_ids,skip_special_tokens=True)
    print("Augmented English Sentences")
    print(sents[:10])
    
def generate_aug_data_2(summary_ids, summary_attn, gpt_model, bart_model, DS_model):

    #######################################################################################################
    ### GPT convert
    # convert input to the bart encodings
    # # use the generate approach
    gpt_summary_ids = gpt_model.generate(summary_ids[:,:5])
    
    gpt_logits = gpt_model(gpt_summary_ids, torch.ones_like(gpt_summary_ids).long(), target_ids = gpt_summary_ids , target_attn = torch.ones_like(gpt_summary_ids).long()).logits

    # find the decoded vector from probabilities
    
    gpt_soft_idx, gpt_idx = torch.max(gpt_logits, dim=-1, keepdims= True)

    #######################################################################################################
    
    # convert to the bart ids
    
    gpt2bart_summary_ids, gpt2bart_summary_attn = tokenize(gpt_model.tokenizer.batch_decode(gpt_summary_ids,skip_special_tokens=True), bart_model.tokenizer, max_length = summary_length)
    ### T-5 specific
#     sents = gpt_model.tokenizer.batch_decode(gpt_summary_ids,skip_special_tokens=True)
#     for i in range(len(sents)):
#         sents[i] = 'translate English to German: '+sents[i]
        
#     gpt2bart_summary_ids, gpt2bart_summary_attn = tokenize(sents,bart_model.tokenizer, max_length = summary_length)
    
    
    gpt2bart_summary_ids, gpt2bart_summary_attn = gpt2bart_summary_ids.cuda(), gpt2bart_summary_attn.cuda()


    # BART
    bart_article_ids = bart_model.generate(gpt2bart_summary_ids)
    
    sents = bart_model.tokenizer.batch_decode(bart_article_ids,skip_special_tokens=True)
    sents_trans = normalize_gen(sents) # generated Translated sentences

    
    sents_root = gpt_model.tokenizer.batch_decode(gpt_summary_ids,skip_special_tokens=True) # generated root sentences
    return sents_root, sents_trans


### Helsinki loss
# def calc_loss_aug(summary_ids, summary_attn, gpt_model, bart_model, DS_model):

#     #######################################################################################################
#     ### GPT convert
#     # convert input to the bart encodings
#     # # use the generate approach
#     gpt_summary_ids = gpt_model.generate(summary_ids[:,:5])
    
#     gpt_logits = gpt_model(gpt_summary_ids, torch.ones_like(gpt_summary_ids).long(), target_ids = gpt_summary_ids , target_attn = torch.ones_like(gpt_summary_ids).long()).logits

#     # find the decoded vector from probabilities
    
#     gpt_soft_idx, gpt_idx = torch.max(gpt_logits, dim=-1, keepdims= True)

#     #######################################################################################################
    
#     # convert to the bart ids
    
#     gpt2bart_summary_ids, gpt2bart_summary_attn = tokenize(gpt_model.tokenizer.batch_decode(gpt_summary_ids,skip_special_tokens=True), bart_model.tokenizer, max_length = summary_length)
#     ### T-5 specific
# #     sents = gpt_model.tokenizer.batch_decode(gpt_summary_ids,skip_special_tokens=True)
# #     for i in range(len(sents)):
# #         sents[i] = 'translate English to German: '+sents[i]
        
# #     gpt2bart_summary_ids, gpt2bart_summary_attn = tokenize(sents,bart_model.tokenizer, max_length = summary_length)
    
    
#     gpt2bart_summary_ids, gpt2bart_summary_attn = gpt2bart_summary_ids.cuda(), gpt2bart_summary_attn.cuda()

#     #######################################################################################################

#     ## BART model articles generation
    
#     # the gumbel soft max trick
    
#     one_hot = torch.zeros(gpt2bart_summary_ids.shape[0], gpt2bart_summary_ids.shape[1], bart_model.vocab_size).cuda()
    
#     bart_summary_ids = one_hot.scatter_(-1, gpt2bart_summary_ids.unsqueeze(-1), 1.).float().detach() + gpt_soft_idx.sum() - gpt_soft_idx.sum().detach()

#     # BART
#     bart_article_ids = bart_model.generate(gpt2bart_summary_ids)
    
#     bart_logits = bart_model(bart_summary_ids, gpt2bart_summary_attn, target_ids = bart_article_ids, target_attn = torch.ones_like(bart_article_ids).long()).logits

    
    
# #     print(bart_logits.shape)
# #     return
#     # find the decoded vector from probabilities
    
#     bart_soft_idx, bart_idx = torch.max(bart_logits, dim=-1, keepdims= True) #Uncomment for previous version

#     #######################################################################################################

#     # convert to the DS ids
#     # "translate French to English: "+
#     sents = bart_model.tokenizer.batch_decode(bart_article_ids,skip_special_tokens=True)
# #     sents = ["<bos> "+t+ " <eos>" for t in sents]
# #     logging.info("German Sentence")
# #     logging.info(sents[-1])
#     # GERMAN TOKENIZING
# #     print("French Sentences: ",sents)
#     bart2DS_article_ids, bart2DS_article_attn = tokenize(sents, DS_model.tokenizer, max_length = article_length) #NEED Implementation
# #     bart2DS_article_ids = tokenize_rnn(sents,DS_model)
#     bart2DS_article_ids = bart2DS_article_ids.cuda()
#     bart2DS_article_attn = bart2DS_article_attn.cuda()
# #     print("Augmented German Sentences:",sents)
# #     print("Augmented German IDs:",bart2DS_article_ids)
# #     print("English Sentences :",gpt_model.tokenizer.batch_decode(gpt_summary_ids,skip_special_tokens=True))
    
#     sents = gpt_model.tokenizer.batch_decode(gpt_summary_ids,skip_special_tokens=True)
# #     logging.info("English Sentence")
# #     logging.info(sents[-1])
# #     sents = ["<bos> "+t+ " <eos>" for t in sents]
# #     print(sents)
#     bart2DS_summary_ids,bart2DS_summary_attn = tokenize(sents, DS_model.tokenizer, max_length = summary_length)
# #     print("Augmented English Sentences:",sents)
# #     print("Augmented English IDs:",bart2DS_summary_ids)
# #     bart2DS_summary_ids, bart2DS_summary_attn = tokenize(, DS_model.tokenizer, max_length = summary_length) #NEED IMPLEMENTATION
    
#     bart2DS_summary_ids,bart2DS_summary_attn = bart2DS_summary_ids.cuda(),bart2DS_summary_attn.cuda()

#     #######################################################################################################
    
#     # DS model

#     one_hot = torch.zeros(bart2DS_article_ids.shape[0], bart2DS_article_ids.shape[1], DS_model.vocab_size).cuda() #Uncomment for previous version
    
#     DS_article_ids = one_hot.scatter_(-1, bart2DS_article_ids.unsqueeze(-1), 1.).float().detach() + bart_soft_idx.sum() - bart_soft_idx.sum().detach()    #Uncomment for previous version
    
#     DS_summary_ids = bart2DS_summary_ids

#     loss = DS_model.loss(DS_article_ids,bart2DS_article_attn,DS_summary_ids,bart2DS_summary_attn)[1]

        
#     gc.collect()   
    
#     return loss
def generate_aug_data_tri(mbart_english_ids, bart_model, DS_model):
    bart_src_ids = bart_model.generate(mbart_english_ids)
    
    bart_logits = bart_model(mbart_english_ids, bart_article_ids).logits
    
    bart_soft_idx, bart_idx = torch.max(bart_logits, dim=-1, keepdims= True) #Uncomment for previous version

    #######################################################################################################
    sents = bart_model.tokenizer.decode(bart_src_ids.detach().cpu().numpy().tolist())
    sents = [ti.replace(" ⁇ ", "") for ti  in sents]
    print('Source Sentences')
    print(sents[:10])

    bart2DS_article_ids, bart2DS_article_attn = tokenize(sents, DS_model.tokenizer, max_length = article_length) #NEED Implementation
    bart2DS_article_ids = tokenize_rnn(sents,DS_model)
    bart2DS_article_ids = bart2DS_article_ids.cuda()
    
    sents = bart_model.tokenizer.batch_decode(mbart_english_ids,skip_special_tokens=True)
    sents = normalize_gen(sents)
    print('Target Sentences')
    print(sents[:10])
    
### RNN LOSS    

def calc_loss_aug_tri(mbart_english_ids, bart_model, DS_model):
    bart_src_ids = bart_model.generate(mbart_english_ids)
    if type(bart_src_ids)==list:
        bart_src_ids = pad_sequence(bart_src_ids, batch_first = True, padding_value = 0 )
        bart_src_ids = torch.tensor(bart_src_ids)
#     print(mbart_english_ids, bart_src_ids.long())
    bart_logits = bart_model(mbart_english_ids.cuda(), bart_src_ids.long().cuda()).output
    
    bart_soft_idx, bart_idx = torch.max(bart_logits, dim=-1, keepdims= True) #Uncomment for previous version

    #######################################################################################################
    sents = bart_model.trg_tokenizer.decode(bart_src_ids.detach().cpu().numpy().tolist())
    sents = [ti.replace(" ⁇ ", "") for ti  in sents]

    tokenized = DS_model.src_tokenizer.encode(sents)
    tokenized = [torch.tensor(t) for t in tokenized]
    
    ids = pad_sequence(tokenized, batch_first = True, padding_value = 0 )
    bart2DS_article_ids = torch.tensor(ids).cuda()
       
    bart2DS_summary_ids = mbart_english_ids.cuda()

    one_hot = torch.zeros(bart2DS_article_ids.shape[0], bart2DS_article_ids.shape[1], DS_model.input_dim).cuda() #Uncomment for previous version
    
    DS_article_ids = one_hot.scatter_(-1, bart2DS_article_ids.unsqueeze(-1), 1.).float().detach() + bart_soft_idx.sum() - bart_soft_idx.sum().detach()    #Uncomment for previous version

    DS_summary_ids = bart2DS_summary_ids

    loss = DS_model.loss(DS_article_ids,DS_summary_ids)

        
    gc.collect()   
    
    return loss

### RNN LOSS    

def calc_loss_aug(summary_ids, summary_attn, gpt_model, bart_model, DS_model):

    #######################################################################################################
    ### GPT convert
    # convert input to the bart encodings
    # # use the generate approach
    gpt_summary_ids = gpt_model.generate(summary_ids[:,:5])
    
    gpt_logits = gpt_model(gpt_summary_ids, torch.ones_like(gpt_summary_ids).long(), target_ids = gpt_summary_ids , target_attn = torch.ones_like(gpt_summary_ids).long()).logits

    # find the decoded vector from probabilities
    
    gpt_soft_idx, gpt_idx = torch.max(gpt_logits, dim=-1, keepdims= True)

    #######################################################################################################
    
    # convert to the bart ids
    
    gpt2bart_summary_ids, gpt2bart_summary_attn = tokenize(gpt_model.tokenizer.batch_decode(gpt_summary_ids,skip_special_tokens=True), bart_model.tokenizer, max_length = summary_length)
    ### T-5 specific
#     sents = gpt_model.tokenizer.batch_decode(gpt_summary_ids,skip_special_tokens=True)
#     for i in range(len(sents)):
#         sents[i] = 'translate English to German: '+sents[i]
        
#     gpt2bart_summary_ids, gpt2bart_summary_attn = tokenize(sents,bart_model.tokenizer, max_length = summary_length)
    
    
    gpt2bart_summary_ids, gpt2bart_summary_attn = gpt2bart_summary_ids.cuda(), gpt2bart_summary_attn.cuda()

    #######################################################################################################

    ## BART model articles generation
    
    # the gumbel soft max trick
    
    one_hot = torch.zeros(gpt2bart_summary_ids.shape[0], gpt2bart_summary_ids.shape[1], bart_model.vocab_size).cuda()
    
    bart_summary_ids = one_hot.scatter_(-1, gpt2bart_summary_ids.unsqueeze(-1), 1.).float().detach() + gpt_soft_idx.sum() - gpt_soft_idx.sum().detach()

    # BART
    bart_article_ids = bart_model.generate(gpt2bart_summary_ids)
    
    bart_logits = bart_model(bart_summary_ids, gpt2bart_summary_attn, target_ids = bart_article_ids, target_attn = torch.ones_like(bart_article_ids).long()).logits

    
    
#     print(bart_logits.shape)
#     return
    # find the decoded vector from probabilities
    
    bart_soft_idx, bart_idx = torch.max(bart_logits, dim=-1, keepdims= True) #Uncomment for previous version

    #######################################################################################################

    # convert to the DS ids
    # "translate French to English: "+
    sents = bart_model.tokenizer.batch_decode(bart_article_ids,skip_special_tokens=True)
    sents = normalize_gen(sents)
#     sents = ["<bos> "+t+ " <eos>" for t in sents]
#     logging.info("German Sentence")
#     logging.info(sents[-1])
    # GERMAN TOKENIZING
#     print("French Sentences: ",sents)
    bart2DS_article_ids, bart2DS_article_attn = tokenize(sents, DS_model.tokenizer, max_length = article_length) #NEED Implementation
    bart2DS_article_ids = tokenize_rnn(sents,DS_model)
    bart2DS_article_ids = bart2DS_article_ids.cuda()
#     print("Augmented German Sentences:",sents)
#     print("Augmented German IDs:",bart2DS_article_ids)
#     print("English Sentences :",gpt_model.tokenizer.batch_decode(gpt_summary_ids,skip_special_tokens=True))
    
    sents = gpt_model.tokenizer.batch_decode(gpt_summary_ids,skip_special_tokens=True)
#     logging.info("English Sentence")
#     logging.info(sents[-1])
#     sents = ["<bos> "+t+ " <eos>" for t in sents]
    bart2DS_summary_ids = tokenize_rnn(sents,DS_model)
#     print("Augmented English Sentences:",sents)
#     print("Augmented English IDs:",bart2DS_summary_ids)
#     bart2DS_summary_ids, bart2DS_summary_attn = tokenize(, DS_model.tokenizer, max_length = summary_length) #NEED IMPLEMENTATION
    
    bart2DS_summary_ids = bart2DS_summary_ids.cuda()

    #######################################################################################################
    
    # DS model

    one_hot = torch.zeros(bart2DS_article_ids.shape[0], bart2DS_article_ids.shape[1], DS_model.vocab_size).cuda() #Uncomment for previous version
    
    DS_article_ids = one_hot.scatter_(-1, bart2DS_article_ids.unsqueeze(-1), 1.).float().detach() + bart_soft_idx.sum() - bart_soft_idx.sum().detach()    #Uncomment for previous version

    
    DS_summary_ids = bart2DS_summary_ids
    
#     loss = DS_model(DS_article_ids, bart2DS_article_attn, target_ids = DS_summary_ids, target_attn = bart2DS_summary_attn).loss
    loss = DS_model.loss(DS_article_ids,DS_summary_ids)
#     loss = DS_model.loss(sampled_article_ids,DS_summary_ids)
        
#     del DS_article_ids, bart_summary_ids, one_hot
        
    gc.collect()   
    
    return loss

# for the calculation of classification loss on the augmented dataset
# define calc_loss_aug
def calc_loss_aug_2(summary_ids, summary_attn, gpt_model, bart_model, DS_model):

    #######################################################################################################
    ### GPT convert
    # convert input to the bart encodings
    # # use the generate approach
    gpt_summary_ids = gpt_model.generate(summary_ids[:,:5])
    
    gpt_logits = gpt_model(gpt_summary_ids, torch.ones_like(gpt_summary_ids).long(), target_ids = gpt_summary_ids , target_attn = torch.ones_like(gpt_summary_ids).long()).logits

    # find the decoded vector from probabilities
    
    gpt_soft_idx, gpt_idx = torch.max(gpt_logits, dim=-1, keepdims= True)

    #######################################################################################################
    
    # convert to the bart ids
    
    gpt2bart_summary_ids, gpt2bart_summary_attn = tokenize(gpt_model.tokenizer.batch_decode(gpt_summary_ids,skip_special_tokens=True), bart_model.tokenizer, max_length = summary_length)
    ### T-5 specific
#     sents = gpt_model.tokenizer.batch_decode(gpt_summary_ids,skip_special_tokens=True)
#     for i in range(len(sents)):
#         sents[i] = 'translate English to German: '+sents[i]
        
#     gpt2bart_summary_ids, gpt2bart_summary_attn = tokenize(sents,bart_model.tokenizer, max_length = summary_length)
    
    
    gpt2bart_summary_ids, gpt2bart_summary_attn = gpt2bart_summary_ids.cuda(), gpt2bart_summary_attn.cuda()

    #######################################################################################################

    ## BART model articles generation
    
    # the gumbel soft max trick
    
    one_hot = torch.zeros(gpt2bart_summary_ids.shape[0], gpt2bart_summary_ids.shape[1], bart_model.vocab_size).cuda()
    
    bart_summary_ids = one_hot.scatter_(-1, gpt2bart_summary_ids.unsqueeze(-1), 1.).float().detach() + gpt_soft_idx.sum() - gpt_soft_idx.sum().detach()

    # BART
    bart_article_ids = bart_model.generate(gpt2bart_summary_ids)
    
    bart_logits = bart_model(bart_summary_ids, gpt2bart_summary_attn, target_ids = bart_article_ids, target_attn = torch.ones_like(bart_article_ids).long()).logits

    
    
#     print(bart_logits.shape)
#     return
    # find the decoded vector from probabilities
    
    bart_soft_idx, bart_idx = torch.max(bart_logits, dim=-1, keepdims= True) #Uncomment for previous version

    #######################################################################################################

    # convert to the DS ids
    # "translate French to English: "+
    sents = bart_model.tokenizer.batch_decode(bart_article_ids,skip_special_tokens=True)
#     sents = ["<bos> "+t+ " <eos>" for t in sents]
#     logging.info("German Sentence")
#     logging.info(sents[-1])
    # GERMAN TOKENIZING
#     print("French Sentences: ",sents)
    bart2DS_article_ids = tokenize_transformer(sents, DS_model) #NEED Implementation

    bart2DS_article_ids = bart2DS_article_ids.cuda()
#     print("Augmented German Sentences:",sents)
#     print("Augmented German IDs:",bart2DS_article_ids)
#     print("English Sentences :",gpt_model.tokenizer.batch_decode(gpt_summary_ids,skip_special_tokens=True))
    
    sents = gpt_model.tokenizer.batch_decode(gpt_summary_ids,skip_special_tokens=True)

    bart2DS_summary_ids = tokenize_transformer(sents,DS_model)

    
    bart2DS_summary_ids = bart2DS_summary_ids.cuda()

    #######################################################################################################
    
    # DS model

    one_hot = torch.zeros(bart2DS_article_ids.shape[0], bart2DS_article_ids.shape[1], DS_model.vocab_size).cuda() #Uncomment for previous version
    
    DS_article_ids = one_hot.scatter_(-1, bart2DS_article_ids.unsqueeze(-1), 1.).float().detach() + bart_soft_idx.sum() - bart_soft_idx.sum().detach()    #Uncomment for previous version
    
    DS_summary_ids = bart2DS_summary_ids
    
#     loss = DS_model(DS_article_ids, bart2DS_article_attn, target_ids = DS_summary_ids, target_attn = bart2DS_summary_attn).loss
    loss = DS_model.loss(DS_article_ids,DS_summary_ids)
        
    gc.collect()   
    
    return loss

# def calc_loss_aug(summary_ids, summary_attn, gpt_model, bart_model, DS_model):

#     #######################################################################################################
#     ### GPT convert
#     # convert input to the bart encodings
#     # # use the generate approach
#     gpt_summary_ids = gpt_model.generate(summary_ids[:,:5])
    
#     gpt_logits = gpt_model(gpt_summary_ids, torch.ones_like(gpt_summary_ids).long(), target_ids = gpt_summary_ids , target_attn = torch.ones_like(gpt_summary_ids).long()).logits

#     # find the decoded vector from probabilities
    
#     gpt_soft_idx, gpt_idx = torch.max(gpt_logits, dim=-1, keepdims= True)

#     #######################################################################################################
    
#     # convert to the bart ids
    
#     gpt2bart_summary_ids, gpt2bart_summary_attn = tokenize(gpt_model.tokenizer.batch_decode(gpt_summary_ids,skip_special_tokens=True), bart_model.tokenizer, max_length = summary_length)
    
#     gpt2bart_summary_ids, gpt2bart_summary_attn = gpt2bart_summary_ids.cuda(), gpt2bart_summary_attn.cuda()

#     #######################################################################################################

#     ## BART model articles generation
    
#     # the gumbel soft max trick
    
#     one_hot = torch.zeros(gpt2bart_summary_ids.shape[0], gpt2bart_summary_ids.shape[1], bart_model.vocab_size).cuda()
    
#     bart_summary_ids = one_hot.scatter_(-1, gpt2bart_summary_ids.unsqueeze(-1), 1.).float().detach() + gpt_soft_idx.sum() - gpt_soft_idx.sum().detach()

#     # BART
#     bart_article_ids = bart_model.generate(gpt2bart_summary_ids)
    
#     bart_logits = bart_model(bart_summary_ids, gpt2bart_summary_attn, target_ids = bart_article_ids, target_attn = torch.ones_like(bart_article_ids).long()).logits

#     # find the decoded vector from probabilities
    
#     bart_soft_idx, bart_idx = torch.max(bart_logits, dim=-1, keepdims= True)

#     #######################################################################################################

#     # convert to the DS ids
#     # "translate French to English: "+
#     sents = bart_model.tokenizer.batch_decode(bart_article_ids,skip_special_tokens=True)
# #     sents = ["<bos> "+t+ " <eos>" for t in sents]
    
#     # GERMAN TOKENIZING
# #     print("French Sentences: ",sents)
# #     bart2DS_article_ids, bart2DS_article_attn = tokenize(sents, DS_model.tokenizer, max_length = article_length) #NEED Implementation
#     bart2DS_article_ids = tokenize_rnn_de(sents,DS_model)
#     bart2DS_article_ids = bart2DS_article_ids.cuda()
# #     print("Augmented German Sentences:",sents)
# #     print("Augmented German IDs:",bart2DS_article_ids)
# #     print("English Sentences :",gpt_model.tokenizer.batch_decode(gpt_summary_ids,skip_special_tokens=True))
    
#     sents = gpt_model.tokenizer.batch_decode(gpt_summary_ids,skip_special_tokens=True)
# #     sents = ["<bos> "+t+ " <eos>" for t in sents]
#     bart2DS_summary_ids = tokenize_rnn_en(sents,DS_model)
# #     print("Augmented English Sentences:",sents)
# #     print("Augmented English IDs:",bart2DS_summary_ids)
# #     bart2DS_summary_ids, bart2DS_summary_attn = tokenize(, DS_model.tokenizer, max_length = summary_length) #NEED IMPLEMENTATION
    
#     bart2DS_summary_ids = bart2DS_summary_ids.cuda()

#     #######################################################################################################
    
#     # DS model

#     one_hot = torch.zeros(bart2DS_article_ids.shape[0], bart2DS_article_ids.shape[1], len(DS_model.vocab_german)).cuda()
    
#     DS_article_ids = one_hot.scatter_(-1, bart2DS_article_ids.unsqueeze(-1), 1.).float().detach() + bart_soft_idx.sum() - bart_soft_idx.sum().detach()    
    
#     DS_summary_ids = bart2DS_summary_ids
    
# #     loss = DS_model(DS_article_ids, bart2DS_article_attn, target_ids = DS_summary_ids, target_attn = bart2DS_summary_attn).loss
#     loss = DS_model.loss(DS_article_ids,DS_summary_ids)
        
#     del DS_article_ids, bart_summary_ids, one_hot
        
#     gc.collect()   
    
#     return loss

#t5 loss

# def calc_loss_aug(summary_ids, summary_attn, gpt_model, bart_model, DS_model):

#     #######################################################################################################
#     ### GPT convert
#     # convert input to the bart encodings
#     # # use the generate approach
#     gpt_summary_ids = gpt_model.generate(summary_ids[:,:5])
    
#     gpt_logits = gpt_model(gpt_summary_ids, torch.ones_like(gpt_summary_ids).long(), target_ids = gpt_summary_ids , target_attn = torch.ones_like(gpt_summary_ids).long()).logits

#     # find the decoded vector from probabilities
    
#     gpt_soft_idx, gpt_idx = torch.max(gpt_logits, dim=-1, keepdims= True)

#     #######################################################################################################
    
#     # convert to the bart ids
    
#     gpt2bart_summary_ids, gpt2bart_summary_attn = tokenize(gpt_model.tokenizer.batch_decode(gpt_summary_ids,skip_special_tokens=True), bart_model.tokenizer, max_length = summary_length)
    
#     gpt2bart_summary_ids, gpt2bart_summary_attn = gpt2bart_summary_ids.cuda(), gpt2bart_summary_attn.cuda()

#     #######################################################################################################

#     ## BART model articles generation
    
#     # the gumbel soft max trick
    
#     one_hot = torch.zeros(gpt2bart_summary_ids.shape[0], gpt2bart_summary_ids.shape[1], bart_model.vocab_size).cuda()
    
#     bart_summary_ids = one_hot.scatter_(-1, gpt2bart_summary_ids.unsqueeze(-1), 1.).float().detach() + gpt_soft_idx.sum() - gpt_soft_idx.sum().detach()

#     # BART
#     bart_article_ids = bart_model.generate(gpt2bart_summary_ids)
    
#     bart_logits = bart_model(bart_summary_ids, gpt2bart_summary_attn, target_ids = bart_article_ids, target_attn = torch.ones_like(bart_article_ids).long()).logits

#     # find the decoded vector from probabilities
    
#     bart_soft_idx, bart_idx = torch.max(bart_logits, dim=-1, keepdims= True)

#     #######################################################################################################

#     # convert to the DS ids
#     # "translate French to English: "+
#     sents = bart_model.tokenizer.batch_decode(bart_article_ids,skip_special_tokens=True)
#     sents = ["translate French to English: "+t for t in sents]
# #     print("French Sentences: ",sents)
#     bart2DS_article_ids, bart2DS_article_attn = tokenize(sents, DS_model.tokenizer, max_length = article_length)
    
#     bart2DS_article_ids, bart2DS_article_attn = bart2DS_article_ids.cuda(), bart2DS_article_attn.cuda()
# #     print("English Sentences :",gpt_model.tokenizer.batch_decode(gpt_summary_ids,skip_special_tokens=True))
#     bart2DS_summary_ids, bart2DS_summary_attn = tokenize(gpt_model.tokenizer.batch_decode(gpt_summary_ids,skip_special_tokens=True), DS_model.tokenizer, max_length = summary_length)
    
#     bart2DS_summary_ids, bart2DS_summary_attn = bart2DS_summary_ids.cuda(), bart2DS_summary_attn.cuda()

#     #######################################################################################################
    
#     # DS model

#     one_hot = torch.zeros(bart2DS_article_ids.shape[0], bart2DS_article_ids.shape[1], DS_model.vocab_size).cuda()
    
#     DS_article_ids = one_hot.scatter_(-1, bart2DS_article_ids.unsqueeze(-1), 1.).float().detach() + bart_soft_idx.sum() - bart_soft_idx.sum().detach()    
    
#     DS_summary_ids = bart2DS_summary_ids
    
#     loss = DS_model(DS_article_ids, bart2DS_article_attn, target_ids = DS_summary_ids, target_attn = bart2DS_summary_attn).loss
        
#     del DS_article_ids, bart_summary_ids, one_hot
        
#     gc.collect()   
    
#     return loss


# def calc_loss_aug(summary_ids, summary_attn, gpt_model, bart_model, DS_model):

#     #######################################################################################################
#     ### GPT convert
#     # convert input to the bart encodings
#     # # use the generate approach
#     gpt_summary_ids = gpt_model.generate(summary_ids[:,:5])
    
#     gpt_logits = gpt_model(gpt_summary_ids, torch.ones_like(gpt_summary_ids).long(), target_ids = gpt_summary_ids , target_attn = torch.ones_like(gpt_summary_ids).long()).logits

#     # find the decoded vector from probabilities
    
#     gpt_soft_idx, gpt_idx = torch.max(gpt_logits, dim=-1, keepdims= True)

#     #######################################################################################################
    
#     # convert to the bart ids
    
#     gpt2bart_summary_ids, gpt2bart_summary_attn = tokenize(gpt_model.tokenizer.batch_decode(gpt_summary_ids,skip_special_tokens=True), bart_model.tokenizer, max_length = summary_length)
    
#     gpt2bart_summary_ids, gpt2bart_summary_attn = gpt2bart_summary_ids.cuda(), gpt2bart_summary_attn.cuda()

#     #######################################################################################################

#     ## BART model articles generation
    
#     # the gumbel soft max trick
    
#     one_hot = torch.zeros(gpt2bart_summary_ids.shape[0], gpt2bart_summary_ids.shape[1], bart_model.vocab_size).cuda()
    
#     bart_summary_ids = one_hot.scatter_(-1, gpt2bart_summary_ids.unsqueeze(-1), 1.).float().detach() + gpt_soft_idx.sum() - gpt_soft_idx.sum().detach()

#     # BART
#     bart_article_ids = bart_model.generate(gpt2bart_summary_ids)
    
#     bart_logits = bart_model(bart_summary_ids, gpt2bart_summary_attn, target_ids = bart_article_ids, target_attn = torch.ones_like(bart_article_ids).long()).logits

#     # find the decoded vector from probabilities
    
#     bart_soft_idx, bart_idx = torch.max(bart_logits, dim=-1, keepdims= True)

#     #######################################################################################################

#     # convert to the DS ids

# #     bart2DS_article_ids, bart2DS_article_attn = tokenize(bart_model.tokenizer.batch_decode(bart_article_ids,skip_special_tokens=True), DS_model.tokenizer, max_length = article_length) #need to change this
    
#     bart2DS_article_ids = DS_model.tokenizer_french.encode_batch(bart_model.tokenizer.batch_decode(bart_article_ids,skip_special_tokens=True),add_special_tokens=True)  #added
#     bart2DS_article_ids = pad_sequence(bart2DS_article_ids, batch_first=True, padding_value=DS_model.tokenizer_french.pad_token_id) # added
    
# #     bart2DS_article_ids, bart2DS_article_attn = bart2DS_article_ids.cuda(), bart2DS_article_attn.cuda()
#     bart2DS_article_ids = bart2DS_article_ids.cuda() # added
    
# #     bart2DS_summary_ids, bart2DS_summary_attn = tokenize(bart_model.tokenizer.batch_decode(gpt2bart_summary_ids,skip_special_tokens=True), DS_model.tokenizer, max_length = summary_length) ##need to change this
    
#     bart2DS_summary_ids = DS_model.tokenizer_english.encode_batch(bart_model.tokenizer.batch_decode(gpt2bart_summary_ids,skip_special_tokens=True), add_special_tokens=True)
#     bart2DS_summary_ids = pad_sequence(bart2DS_summary_ids, batch_first=True, padding_value=DS_model.tokenizer_english.pad_token_id) # added
    
    
# #     bart2DS_summary_ids, bart2DS_summary_attn = bart2DS_summary_ids.cuda(), bart2DS_summary_attn.cuda()
#     bart2DS_summary_ids = bart2DS_summary_ids.cuda()
#     #######################################################################################################
    
#     # DS model

# #     one_hot = torch.zeros(bart2DS_article_ids.shape[0], bart2DS_article_ids.shape[1], DS_model.vocab_size).cuda() # changed
#     one_hot = torch.zeros(bart2DS_article_ids.shape[0], bart2DS_article_ids.shape[1], DS_model.tokenizer_french.vocab_size).cuda()
                                                                  
#     DS_article_ids = one_hot.scatter_(-1, bart2DS_article_ids.unsqueeze(-1), 1.).float().detach() + bart_soft_idx.sum() - bart_soft_idx.sum().detach()    
    
#     DS_summary_ids = bart2DS_summary_ids
    
#     loss = DS_model(DS_article_ids, target_ids = DS_summary_ids).loss #need to figure this out
    
        
#     del DS_article_ids, bart_summary_ids, one_hot
        
#     gc.collect()   
    
#     return loss