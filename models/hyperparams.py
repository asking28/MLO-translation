import numpy as np
import torch

seed_ = 42

summary_length = 128

article_length = 128

def update_params(args):
    global summary_length,article_length
    summary_length = args.summary_length
    article_length = args.article_length