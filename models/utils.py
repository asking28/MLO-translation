import os
import random
import torch
import numpy as np
import torch
import shutil
# import torchvision.transforms as transforms
from torch.autograd import Variable
from .hyperparams import *
import logging
from rouge import Rouge

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import sys
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed_)


def tokenize(text_data, tokenizer, max_length, padding = True):

    encoding = tokenizer(text_data, return_tensors='pt', padding=padding, truncation = True, max_length = max_length)

    input_ids = encoding['input_ids']

    attention_mask = encoding['attention_mask']

    return input_ids, attention_mask

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct.contiguous()

    res = []

    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))

    return res

def rouge(outputs,summary_DS,DS_model):
    _,preds=outputs.topk(1,2,True,True)
    preds=preds.view(-1,15)
    preds_sents=DS_model.tokenizer.batch_decode(preds)
    summary_sents=DS_model.tokenizer.batch_decode(summary_DS)
    rouge = Rouge()
    scores1 = rouge.get_scores(preds_sents, summary_sents)
    # or
    scores_avg = rouge.get_scores(preds_sents, summary_sents, avg=True)

    return scores_avg['rouge-l']['f'],scores_avg['rouge-1']['f']

def rouge_sentences(preds,labels):
    rouge = Rouge()
#     preds=[l.replace('_',' ') for l in preds if l not None]
    labels=[l.replace('▁',' ') for l in labels if l is not None]
    try:
        scores1=rouge.get_scores(preds,labels)
        scores_avg=rouge.get_scores(preds,labels,avg=True)
        return scores_avg['rouge-l']['f'],scores_avg['rouge-1']['f']
    except:
        new_preds=[]
        new_labels=[]
        for p,l in zip(preds,labels):
            p_s = [" ".join(_.split()) for _ in p.split(".") if len(_) > 0]
            l_s = [" ".join(_.split()) for _ in l.split(".") if len(_) > 0]
            if len(p_s)>0 and len(l_s)>0:
                new_preds.append(p)
                new_labels.append(l)
                
        scores1=rouge.get_scores(new_preds,new_labels)
        scores_avg=rouge.get_scores(new_preds,new_labels,avg=True)
        return scores_avg['rouge-l']['f'],scores_avg['rouge-1']['f']
def bleu_sentences(preds,labels):
    # we need sentence level average, not corpus level bleu (both are different)
    labels=[l.replace('▁',' ') for l in labels if l is not None]
    summ=0.0
    for i in range(len(preds)):
        summ+=nltk.translate.bleu_score.sentence_bleu([word_tokenize(preds[i])],word_tokenize(labels[i]),weights=[1])
    return summ/len(preds) 
class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1.-drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=True, delta=0, path='checkpoint.pt', trace_func=logging.info):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience

        self.verbose = verbose

        self.counter = 0

        self.best_score = None

        self.early_stop = False

        self.val_loss_min = np.Inf

        self.delta = delta

        self.path = path

        self.trace_func = trace_func

    def __call__(self, val_loss, gpt_model, bart_model, DS_model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, gpt_model, bart_model, DS_model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(str(f'EarlyStopping counter: {self.counter} out of {self.patience}'))
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, gpt_model, bart_model, DS_model)
            self.counter = 0

    def save_checkpoint(self, val_loss, gpt_model, bart_model, DS_model):
        '''Saves model when validation loss decrease.'''

        if self.verbose:
            self.trace_func(str(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'))

        torch.save(gpt_model.state_dict(), os.path.join(self.path, 'gpt_weights.pt'))

        torch.save(bart_model.state_dict(), os.path.join(self.path, 'bart_weights.pt'))

        torch.save(DS_model.state_dict(), os.path.join(self.path, 'DS_weights.pt'))

        self.val_loss_min = val_loss
