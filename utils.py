import os
import torch
import random
import numpy as np
import yaml
from munch import Munch
from sklearn.metrics import accuracy_score, roc_auc_score

## When use AKT, uncomment these lines
# from AKT.akt import AKT
# from AKT.dkvmn import DKVMN

def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def tokenize_function(tokenizer, sentences):
    return tokenizer(sentences, padding=True, truncation=True,return_tensors="pt")

def read_configs(path):
    with open(path, "r") as f:
        mydict = yaml.safe_load(f)
    configs = Munch(mydict)
    return configs

def aggregate_metrics(log):
    results = {}
    for k in log[0].keys():
        if k == 'auc':
            logits = np.concatenate([x[k]['logits'].numpy().reshape(-1) for x in log])
            scores = np.concatenate([x[k]['scores'].numpy().reshape(-1) for x in log])
            results[k] = roc_auc_score(scores, logits)
        elif k == 'pred':
            res = np.concatenate([x[k].numpy().reshape(-1) for x in log])
            results[k] = res.sum()
        else:
            res = np.concatenate([x[k].numpy().reshape(-1) for x in log])
            results[k] = np.mean(res)
    return results


def prompt_proc_func(prompt, configs=None):
    return '    /**\n     * ' + prompt.replace('\n', ' ') + '\n     */\n'

def code_proc_func(code, tokenizer, configs=None):
    return tokenizer.eos_token + code + tokenizer.eos_token

## uncomment for AKT/DKVMN
## AKT utilities, copied from AKT/utils
# def load_model(configs, params, q_matrix, qa_matrix):
#     model_type = configs.kt_model

#     ## Parameters for this part is hard-coded for AKT/DKVMN model
#     if model_type in {'akt'}:
#         model = AKT(n_question=configs.n_questions, n_solutions=configs.n_solutions, 
#                     n_blocks=1, d_model=768,
#                     dropout=0.05, kq_same=1, model_type=model_type, 
#                     q_matrix=q_matrix, qa_matrix=qa_matrix).cuda()
#     elif model_type in {'dkvmn'}:
#         model = DKVMN(n_question=configs.n_questions, n_solutions=configs.n_solutions, 
#                       seqlen=configs.max_len, d_model=768, memory_size=50,
#                       q_matrix=q_matrix, qa_matrix=qa_matrix)
#     else:
#         model = None
#     return model