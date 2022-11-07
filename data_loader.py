import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from collections import defaultdict
from sklearn.model_selection import train_test_split

from utils import set_random_seed, tokenize_function, prompt_proc_func, code_proc_func

from pdb import set_trace


def read_data(configs):
    '''
    @param configs.label_type: whether to use binarized label, raw label, or ternery label
    @param configs.max_len: maximum allowed length for each student's answer sequence. longer
                    than this number will be truncated and set as new student(s)
    @param configs.seed: reproducibility
    '''
    # load dataset
    dataset = pd.read_pickle(configs.data_path + '/dataset.pkl')
    
    ## if only testing, subsample part of dataset
    if configs.testing:
        dataset = dataset.sample(n=500)
    
    # choose label format
    if configs.label_type == 'binary':
        scores_y = []
        for item in dataset['Score_y']:
            if item >= 2:
                scores_y.append(1)
            else:
                scores_y.append(0)
        dataset['Score'] = scores_y
    elif configs.label_type == 'ternery':
        dataset['Score'] = dataset['Score_y']
    elif configs.label_type == 'raw':
        dataset['Score'] = dataset['Score_x']
    dataset = dataset.drop(columns=['Score_x','Score_y'])
    
    # convert prompt embedding to tensor
    if configs.data_for == 'lstm':
        dataset['prompt-embedding'] = dataset['prompt-embedding'].apply(lambda x: torch.tensor(x))
    
    ## optionally keep only the first answer by the student
    if configs.first_ast_convertible:
        print('only using first ast-convertible code')
        dataset = dataset.drop_duplicates(
                        subset = ['SubjectID', 'ProblemID'],
                        keep = 'first').reset_index(drop = True)
    
    ## split a student's record into multiples 
    ## if it exceeds configs.max_len, change the subject ID to next one
    prev_subject_id = 0
    subjectid_appedix = []
    timesteps = []
    for i in tqdm(range(len(dataset)), desc="splitting students' records ..."):
        if prev_subject_id != dataset.iloc[i].SubjectID:
            # when encountering a new student ID
            prev_subject_id = dataset.iloc[i].SubjectID
            accumulated = 0
            id_appendix = 1
        else:
            accumulated += 1
            if accumulated >= configs.max_len:
                id_appendix += 1
                accumulated = 0
        timesteps.append(accumulated)
        subjectid_appedix.append(id_appendix)
    dataset['timestep'] = timesteps
    dataset['SubjectID_appendix'] = subjectid_appedix
    dataset['SubjectID'] = [dataset.iloc[i].SubjectID + \
                '_{}'.format(dataset.iloc[i].SubjectID_appendix) for i in range(len(dataset))]
        
    ## Each subject ID implies a student
    students = dataset['SubjectID'].unique()
    
    ## optionally load the knowledge component
    if configs.use_kc:
        kc_collections = []
        print('use knowledge components')
        kc_all = pd.read_excel(configs.data_path + '/prompt_concept.xlsx')
        exclude_cols = ['AssignmentID', 'ProblemID', 'Requirement'] # this part is hardcoded
        kc_len = len(kc_all.columns) - len(exclude_cols)
        for row in range(len(dataset)):
            kc_vec = np.zeros(kc_len)
            kc = kc_all[kc_all.ProblemID == dataset.iloc[row].ProblemID]
            for idx, col in enumerate(kc_all.columns):
                if col not in exclude_cols and kc[col].item() == 1:
                    kc_vec[idx-3] = 1
            kc_collections.append(kc_vec)
        dataset['kc_vec'] = kc_collections
    else:
        dataset['kc_vec'] = [0] * len(dataset)
    
    ## train-valid split 
    if configs.data_for == 'lstm':
        # for student model, split on the students
        train_students, test_students = train_test_split(students, test_size=configs.test_size, random_state=configs.seed)
        valid_students, test_students = train_test_split(test_students, test_size=0.5, random_state=configs.seed)
        return train_students, valid_students, test_students, dataset
    elif configs.data_for == 'okt':
        dataset = dataset.drop(dataset.index[dataset['timestep'] == 0]).reset_index(drop = True)
        # OKT: split on the entries instead of on the students
        trainset, testset = train_test_split(dataset, test_size=configs.test_size, random_state=configs.seed)
        validset, testset = train_test_split(testset, test_size=0.5, random_state=configs.seed)
        return trainset, validset, testset, dataset
        
        
def make_pytorch_dataset(dataset_split, dataset_full, students, configs, do_lstm_dataset=True):
    '''
    convert the pandas dataframe into dataset format that pytorch dataloader takes
    the resulting format is a list of dictionaries
    '''
    if configs.data_for == 'lstm':
        lstm_dataset = []
    
        for student in students:
            subset = dataset_full[dataset_full.SubjectID==student]
            if configs.data_for == 'lstm':
                lstm_dataset.append({
                    'SubjectID': student,
                    'ProblemID_seq': subset.ProblemID.tolist(),
                    'Score': subset.Score.tolist(),
                    'prompt-embedding': subset['prompt-embedding'].tolist(),
                    'input': subset.input.tolist(),
                })
        del dataset_full
        return lstm_dataset
    
    elif configs.data_for == 'okt':
        okt_dataset = []
        students = dataset_split['SubjectID'].unique()
        for student in students:
            subset = dataset_split[dataset_split.SubjectID==student]
            for t in range(len(subset)):
                # IMPORTANT: we want to predict the student's answer to the NEXT time step's question prompt
                okt_dataset.append({
                    'SubjectID': student,
                    'step': subset.iloc[t].timestep-1, 
                    'next_Score': subset.iloc[t].Score,
                    'next_prompt': subset.iloc[t].prompt,
                    'next_prompt_emb': subset.iloc[t]['prompt-embedding'],
                    'next_prompt_kc': subset.iloc[t].kc_vec, # only in use when use_kc=True
                    'next_code': subset.iloc[t].Code,
                })
                # set_trace()
        del dataset_split
        
        # dictionary, key=student id, value=list of lstm inputs at each time step
        if do_lstm_dataset:
            lstm_dataset = {}
            students = dataset_full['SubjectID'].unique()
            if configs.combine_method not in ['exp_decay', 'kc_sim_decay', 'exp_kc_decay', 'no_decay']:
                for student in students:
                    if configs.kt_model == 'lstm':
                        lstm_dataset[student]=dataset_full[dataset_full.SubjectID==student].input.tolist()
                    elif configs.kt_model in {'akt', 'dkvmn'}:
                        # get the q embedding matrix
                        q_data_unique = np.unique(dataset_full.ProblemID.tolist()) # sorted, so deterministic
                        q_data_unique = np.concatenate([[0], q_data_unique])
                        q_dict = dict(zip(q_data_unique, np.arange(0, len(q_data_unique)))) # reserve 0 for padding
                        # get the c embedding matrix
                        cs_data_unique = dataset_full.CodeStateID.unique().tolist() # follows the dataset order
                        cs_data_unique = ['0'] + cs_data_unique
                        c_dict = dict(zip(cs_data_unique, np.arange(0, len(cs_data_unique)))) # reserve 0 for padding
                        # get the student's question sequence, padding with 0
                        q_data = dataset_full[dataset_full.SubjectID==student].ProblemID.tolist()
                        q_data = q_data + (configs.max_len - len(q_data)) * [0]
                        q_data = [q_dict[q] for q in q_data]
                        # get the student's code state sequence, padding with '0'
                        c_data = dataset_full[dataset_full.SubjectID==student].CodeStateID.tolist()
                        c_data = c_data + (configs.max_len - len(c_data)) * ['0']
                        c_data = [c_dict[c] for c in c_data]
                        lstm_dataset[student] = {'q_data': q_data, 'c_data': c_data}
            else:
                for student in students:
                    assert(len(dataset_full[dataset_full.SubjectID==student]) <= configs.max_len)
                    lstm_dataset[student]={'code_emb': dataset_full[dataset_full.SubjectID==student].embedding.tolist(),
                                            'prompt_kc': dataset_full[dataset_full.SubjectID==student].kc_vec.tolist(),
                                        }
            del dataset_full
            return okt_dataset, lstm_dataset
        else:
            return okt_dataset


def make_dataloader(dataset_split, dataset_full, students, collate_fn, configs, n_workers=0, do_lstm_dataset=True, train=True):
    '''
    if lstm, make standard dataset with a list of dict
    if okt , make two datasets: one with a list of dict (for GPT), and another a dict with student_id as key (for LSTM to compute knowledge states)
    '''
    
    shuffle = True if train else False
    
    ## make these sets into pytorch dataset format (list of dicts)
    if configs.data_for == 'lstm':
        lstm_dataset = make_pytorch_dataset(dataset_split, dataset_full, students, configs)
        data_loader = torch.utils.data.DataLoader(
            lstm_dataset, collate_fn=collate_fn, shuffle=shuffle, batch_size=configs.batch_size, num_workers=n_workers)    
        return data_loader
     
    elif configs.data_for == 'okt':
        if do_lstm_dataset:
            okt_dataset, lstm_dataset = make_pytorch_dataset(dataset_split, dataset_full, None, configs, do_lstm_dataset)
            data_loader = torch.utils.data.DataLoader(
                okt_dataset, collate_fn=collate_fn, shuffle=shuffle, batch_size=configs.batch_size, num_workers=n_workers)   
            return okt_dataset, data_loader, lstm_dataset
        else:
            okt_dataset = make_pytorch_dataset(dataset_split, dataset_full, None, configs, do_lstm_dataset)
            data_loader = torch.utils.data.DataLoader(
                okt_dataset, collate_fn=collate_fn, shuffle=shuffle, batch_size=configs.batch_size, num_workers=n_workers)   
            return okt_dataset, data_loader


class CollateForLSTM(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        ## input padding for scores
        scores = [b['Score'] for b in batch]
        max_len = max(len(i) for i in scores)
        padded_scores = [i + [-100]*(max_len-len(i)) for i in scores]
        padded_scores = torch.tensor(padded_scores).float().t() # dim=T*B
        
        ## input padding for lstm input
        inputs = [b['input'] for b in batch]
        padded_inputs = [i + [torch.zeros(i[0].shape[0])]*(max_len - len(i)) for i in inputs]
        padded_inputs = torch.stack([torch.stack(x, dim=0) for x in padded_inputs], dim=1).float() # dim=T*B*D
        
        ## prompt embedding padding for output computation
        prompt_embs = [b['prompt-embedding'] for b in batch]
        padded_prompt_embs = [i + [torch.zeros(i[0].shape[0])]*(max_len - len(i)) for i in prompt_embs]
        padded_prompt_embs = torch.stack([torch.stack(x, dim=0) for x in padded_prompt_embs], dim=1).float() # dim=T*B*D
        
        return padded_inputs, padded_prompt_embs, padded_scores
    
    
class CollateForOKT(object):
    def __init__(self, tokenizer, configs):
        self.tokenizer = tokenizer
        self.configs = configs
        
    def __call__(self, batch):

        ## get input prompts and turn them into embeddings, with padding
        generator_inputs_raw = [prompt_proc_func(b['next_prompt']) + code_proc_func(b['next_code'], self.tokenizer) for b in batch]
        generator_inputs = self.tokenizer(generator_inputs_raw, return_tensors='pt', padding=True, truncation=True) # WARNING TODO for now we just use truncate the code if it's maximum length is beyond GPT's maximum length
        generator_inputs_ids, attention_mask = generator_inputs['input_ids'], generator_inputs['attention_mask']
        
        if self.configs.testing:
            for i in range(generator_inputs_ids.shape[0]):
                len_tmp = self.tokenizer(generator_inputs_raw[i], return_tensors='pt')['input_ids'].shape[1]
                assert(attention_mask[i, len_tmp:].sum() == 0)
                
        # find the length of the prompt; special attention paid to truncated code where no EOS token at the end.
        # In that case we just use the length of the sequence as the length of the code
        eos_id_locations = torch.where(generator_inputs_ids==self.tokenizer.eos_token_id)
        first_eos_id_location = torch.zeros(generator_inputs_ids.shape[0]).long()
        second_eos_id_location = torch.zeros(generator_inputs_ids.shape[0]).long()
        for idx in range(generator_inputs_ids.shape[0]):
            eos_idx = eos_id_locations[1][eos_id_locations[0]==idx]
            first_eos_id_location[idx] = eos_idx[0]
            second_eos_id_location[idx] = eos_idx[1] if len(eos_idx) > 1 else len(generator_inputs_ids[idx])-1
        prompt_id_lens = [len(generator_inputs_ids[i][:first_eos_id_location[i]]) for i in range(len(first_eos_id_location))]
        
        # labels; only for the code
        labels = torch.ones_like(generator_inputs_ids) * -100.
        for idx in range(labels.shape[0]):
            labels[idx, first_eos_id_location[idx]+1:second_eos_id_location[idx]+1] = generator_inputs_ids[idx, first_eos_id_location[idx]+1:second_eos_id_location[idx]+1]
        
        if self.configs.testing:
            try:
                assert(50257 not in labels)
            except:
                set_trace()
        
        ## students
        students = [b['SubjectID'] for b in batch]
        
        ## time steps
        timesteps = [b['step'] for b in batch]
        
        ## correctness labels
        correctness = [b['next_Score'] for b in batch]
        padded_correctness = torch.tensor(correctness).float()

        ## next prompt embeddings
        next_prompt_embs = torch.tensor([b['next_prompt_emb'] for b in batch]).float()
        
        ## optional knowledge components
        if self.configs.use_kc:
            kc_vecs = torch.FloatTensor(np.stack([b['next_prompt_kc'] for b in batch]))            
            return generator_inputs_ids, attention_mask, labels.long(), prompt_id_lens, students, timesteps, kc_vecs

        if self.configs.use_classifier:
            return generator_inputs_ids, attention_mask, labels.long(), prompt_id_lens, students, timesteps, padded_correctness, next_prompt_embs

        return generator_inputs_ids, attention_mask, labels.long(), prompt_id_lens, students, timesteps