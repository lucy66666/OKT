import torch
import torch.nn.functional as F

import nltk
from nltk import ngrams

import os
from multiprocessing import Pool
from nltk.translate.bleu_score import SmoothingFunction
import abc
from utils import prompt_proc_func, code_proc_func
from trainer import *

from evaluator.CodeBLEU import calc_code_bleu

# prevent an error:
#       "huggingface/tokenizers: The current process just got forked, after parallelism has already been used"
#       "Avoid using `tokenizers` before the fork if possible"
#       "Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from pdb import set_trace

def generate_code(test_set, lstm_inputs, tokenizer, 
                     idx, model, lstm, linear, weight, configs):
    
    # get the knowledge state
    student, step, prompt, code = test_set[idx]['SubjectID'], test_set[idx]['step'], test_set[idx]['next_prompt'], test_set[idx]['next_code']
    
    if configs.use_kc:
        kc_vecs = torch.FloatTensor(test_set[idx]['next_prompt_kc']).unsqueeze(0)
    
    ks, _ = get_knowledge_states_for_generator(lstm, lstm_inputs, [student], [step], configs)
    
    # assemble generator input
    tokenized_prompt = tokenizer(prompt_proc_func(prompt), return_tensors='pt')['input_ids']
    prompt_wte = model.transformer.wte(tokenized_prompt.cuda())
    
    # combine knowledge with generator input
    for i in range(len([student])):
        if configs.combine_method == 'add':
            prompt_wte += configs.combine_weight * ks[i]
        elif configs.combine_method == 'average':
            prompt_wte = (prompt_wte + configs.combine_weight * ks[i]) / 2
        elif configs.combine_method == 'weight':
            prompt_wte += weight * ks[i]
        elif configs.combine_method == 'linear':
            if configs.use_kc:
                assert(kc_vecs[i] is not None)
                prompt_wte += linear( torch.cat((ks[i], kc_vecs[i].cuda()), dim=-1) )
            else:
                prompt_wte += linear(ks[i])  
        elif configs.combine_method in ['exp_decay', 'kc_sim_decay', 'exp_kc_decay', 'no_decay']:
            prompt_wte += linear(ks[i]) 
    
    # this aligns with training: the EOS token is not added with knowledge state 
    generator_input_emb = torch.cat((prompt_wte.squeeze(0), model.transformer.wte(torch.tensor([tokenizer.eos_token_id]).cuda())), dim=0) 
    
    # generate
    with torch.no_grad():
        generated_ids, nll = generate(model, tokenizer, generator_input_emb, k=configs.k, p=configs.p)
        generated_code = tokenizer.decode(generated_ids).strip() # for codebleu computation
        
    return generated_code.strip(), nll, code.replace('<|endoftext|>', '').strip(), prompt


def compute_code_bleu(ground_truth_codes, generated_codes):
    params='0.25,0.25,0.25,0.25'
    lang='java'
    codebleu_score, detailed_codebleu_score = calc_code_bleu.get_codebleu(
                        pre_references=[ground_truth_codes], hypothesis=generated_codes, lang=lang, params=params)
    return codebleu_score, detailed_codebleu_score
    

# helper function for sampling
def top_k_top_p_filtering(logits, top_k=1, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def generate(model, tokenizer, generator_input_emb, k=1, p=0):
    count = 0
    past = None
    generations = []
    nll = 0
    
    for step in range(400):

        if step == 0:
            out = model(inputs_embeds=generator_input_emb, past_key_values=past)
        else:
            out = model(inp, past_key_values=past)

        past = out['past_key_values']
        logits = out['logits'][-1]
        probs = torch.softmax(logits, dim=-1)
        filtered_logits = top_k_top_p_filtering(logits, top_k=k, top_p=p)
        inp = torch.multinomial(torch.softmax(filtered_logits, dim=-1), 1)
        if inp.item() == tokenizer.eos_token_id:
            continue
        elif tokenizer.decode(inp.item()) == 'public':
            if count == 0:
                generations.append(inp.item())
                count += 1
            else:
                break
        else:
            generations.append(inp.item())
        nll += -torch.log(probs[inp.item()])
    return generations, nll.item() / len(generations)


class Metric():
    """
    Defines a text quality metric.
    """

    def get_name(self):
        return self.name

    @abc.abstractmethod
    def compute_metric(self, texts):
        pass


class Distinct_N(Metric):

    def __init__(self, n):
        """
        Distinct n-grams metrics. This is a sequence-level diversity metric.
        See https://www.aclweb.org/anthology/N16-1014 for more details.

        Args:
            n (int): n-grams 
        """

        self.n = n
        self.name = f'Distinct_{n}'

    def compute_metric(self, texts):
        return self._distinct_ngrams(texts, self.n)

    def _distinct_ngrams(self, texts, n):
        total = 0.0
        for t in texts:
            try:
                tokens = nltk.tokenize.word_tokenize(t)
                n_distinct = len(set(ngrams(tokens, n)))
                total += n_distinct/ len(tokens)
            except:
                continue

        return total / len(texts)