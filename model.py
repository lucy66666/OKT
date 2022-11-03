import torch
import torch.nn as nn
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelWithLMHead
# from utils import load_model

from pdb import set_trace

def create_lstm_model(configs):
    ## Create a LSTM with input dimension [968] (prompt=[768], ASTNN=[200]) and output dimension [768]
    lstm = nn.LSTM(configs.lstm_inp_dim, configs.lstm_hid_dim)
    if configs.pre_trained_lstm_path:
        lstm = torch.load(configs.pre_trained_lstm_path)
    lstm.cuda();
    
    if configs.use_classifier:
        classifier = nn.Sequential(
            nn.Linear(configs.lstm_hid_dim+768, configs.classifier_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.classifier_hid_dim, 1)
        )
        classifier.cuda();
    
        # orthogonal init for classifier
        for name, param in classifier.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
                
        return lstm, classifier
    
    return lstm


def create_tokenizer(configs):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if configs.pad_token == 'pad':
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    elif configs.pad_token == 'eos':
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def create_okt_model(configs):
    ## load the code generator model
    tokenizer = create_tokenizer(configs)
    
    if configs.okt_model == 'student':
        model = AutoModelWithLMHead.from_pretrained("model/gpt_code_v1_student")
    elif configs.okt_model == 'funcom':
        model = AutoModelWithLMHead.from_pretrained("/model/gpt_code_v1")
    else:
        model = AutoModelWithLMHead.from_pretrained('gpt2')
        
    if configs.pad_token == 'pad':
        model.resize_token_embeddings(len(tokenizer))
    model.cuda();
    
    weight, linear = None, None
    if configs.combine_method == 'weight':
        weight = nn.Parameter(torch.tensor(float(configs.combine_weight)).cuda(), requires_grad=True)
    elif configs.combine_method == 'linear':
        if configs.use_kc:
            linear = nn.Linear(768 + 18, 768).cuda() # No. KC = 18 is hardcoded
        else:
            linear = nn.Linear(768, 768).cuda()
    elif configs.combine_method in ['exp_decay', 'kc_sim_decay', 'exp_kc_decay', 'no_decay']:
        linear = nn.Linear(200, 768).cuda() # 200=code-embedding dimension, hardcoded
    
    ## load the student model
    classifier = None
    lstm = None
    if configs.use_lstm:
        if configs.kt_model == 'lstm':
            if configs.pre_trained_lstm_path:
                print('using pre-trained student model')
                lstm = torch.load(configs.pre_trained_lstm_path)
                if configs.use_classifier:
                    print('using pre-trained classifier')
                    classifier = torch.load(configs.pre_trained_classifier_path)
            else:
                student_model = create_lstm_model(configs)
                if configs.use_classifier:
                    lstm, classifier = student_model[0].cuda(), student_model[1].cuda()
                else:
                    lstm = student_model
                    lstm.cuda();
        # If model is 'akt' or 'dkvmn', uncomment this part
        # elif configs.kt_model in {'akt', 'dkvmn'}:
        #     if configs.pre_trained_lstm_path:
        #         params = load_baseline_parameters()
        #         checkpoint = torch.load(configs.pre_trained_lstm_path)
        #         lstm = load_model(configs, params, 
        #                           q_matrix=np.zeros((51,768)), qa_matrix=np.zeros((39797,200))) # 50 dimension is hard-coded
        #         lstm.load_state_dict(checkpoint['model_state_dict'])
        #         lstm.cuda();
    else:
        assert(configs.combine_method in ['exp_decay', 'kc_sim_decay', 'exp_kc_decay', 'no_decay'])
    
    return lstm, classifier, tokenizer, model, linear, weight