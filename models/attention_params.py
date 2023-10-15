import os
import random
import torch
import numpy as np
from .hyperparams import *

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(seed_)

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
        
#         probs = self.softmax(self.alpha[idx])
        probs = self.sigmoid(self.alpha[idx])
#         probs = self.alpha[idx]
#         probs = self.relu(self.alpha[idx])
        
        return probs