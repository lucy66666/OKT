import torch
import numpy as np
from sklearn.metrics import jaccard_score

from pdb import set_trace

def lstm_step(batch, lstm, classifier, hidden_dim, 
                    optimizers, loss_fn, train=True,
                    init='rand', use_scheduler=True, schedulers=None, configs=None):
    if train:
        if configs.lstm_train:
            lstm.train()
        else:
            lstm.eval()
        classifier.train()
    else:
        lstm.eval()
        classifier.eval()
    
    lstm_inputs, prompt_embs, scores = batch[0][:-1], batch[1][:-1], batch[2][1:]

    # lstm hidden state init
    if init == 'rand':
        hidden_h, hidden_c = torch.rand(1, lstm_inputs.shape[1], hidden_dim).cuda(), torch.rand(1, lstm_inputs.shape[1], hidden_dim).cuda()
    elif init == 'zero':
        hidden_h, hidden_c = torch.zeros(1, lstm_inputs.shape[1], hidden_dim).cuda(), torch.zeros(1, lstm_inputs.shape[1], hidden_dim).cuda()
    
    # pass through lstm
    if configs.lstm_train:
        out, hidden = lstm(lstm_inputs.cuda(), (hidden_h, hidden_c))
    else:
        with torch.no_grad():
            out, hidden = lstm(lstm_inputs.cuda(), (hidden_h, hidden_c))
    logits = classifier(torch.cat((out, prompt_embs.cuda()), dim=-1)).squeeze(-1)
    
    # compute stats
    loss = loss_fn(logits[scores!=-100], scores[scores!=-100].cuda()).sum()
    if train:
        ( loss / lstm_inputs.shape[1] ).backward()
    
    # optimization
    if train:
        for optimizer in optimizers:
            optimizer.step()
        if use_scheduler:
            for scheduler in schedulers:
                scheduler.step(loss / lstm_inputs.shape[1])
        for optimizer in optimizers:
            optimizer.zero_grad()
        lstm.zero_grad()
        classifier.zero_grad()
    
    predictions = (torch.sigmoid(logits) > 0.5) * 1.
    acc = predictions[scores!=-100].detach().cpu() == scores[scores!=-100]
    log = {'loss': (loss.detach().cpu().true_divide(len(scores[scores!=-100]))).repeat(len(scores[scores!=-100])), 
            'pred': predictions.detach().cpu(), 'acc': acc, 
            'auc':{'logits': logits[scores!=-100].detach().cpu(), 'scores': scores[scores!=-100]}}
    
    if train:
        return log, lstm, classifier
    else:
        return log
    
    



def generator_step(batch, lstm_inputs,
                   model, lstm, linear, weight,
                   optimizers=None, optimizers_lstm=None, 
                   configs=None, train=True, scheduler=None, classifier=None):
    
    assert(configs!=None)        
    
    if train:
        if configs.train_okt:
            assert(optimizers != None)
            model.train()
            if configs.combine_method == 'linear':
                linear.train()
            
        if configs.train_lstm and configs.use_lstm:
            assert(optimizers_lstm != None)
            lstm.train()
        if configs.use_classifier:
            classifier.train()
    else:
        model.eval()
        if configs.use_lstm:
            lstm.eval()
        if configs.combine_method == 'linear':
            linear.eval()
        if configs.use_classifier:
            classifier.eval()
        
    # assemble generator input
    generator_inputs_ids, attention_mask, labels, prompt_id_lens, \
        students, timesteps = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
    
    kc_vecs = batch[-1] if configs.use_kc else None
    padded_correctness = batch[-2] if configs.use_classifier else None
    next_prompt_embs = batch[-1] if configs.use_classifier else None
    
    generator_inputs_wte, ks = assemble_generator_input(model, lstm, linear, weight, configs,
                                                    generator_inputs_ids, prompt_id_lens, 
                                                    lstm_inputs, students, timesteps, kc_vecs=kc_vecs)
    
    # forward generator
    if train:
        outputs = model(inputs_embeds=generator_inputs_wte, attention_mask=attention_mask.cuda(), labels=labels.cuda())
    else:
        with torch.no_grad():
            outputs = model(inputs_embeds=generator_inputs_wte, attention_mask=attention_mask.cuda(), labels=labels.cuda())
    
    # compute stats
    loss = outputs[0]
    
    # compute correctness loss
    if configs.use_classifier:
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
        logits = classifier(torch.cat((ks, next_prompt_embs.cuda()), dim=-1)).squeeze(-1)
        loss_cls = loss_fn(logits[padded_correctness!=-100], padded_correctness[padded_correctness!=-100].cuda()).sum()
        loss_cls = loss
        loss += loss_cls
    
    if train:
        loss.backward()
    
    # optimization
    if train:
        for optimizer in optimizers:
            optimizer.step()
        if configs.use_scheduler:
            scheduler.step()
        for optimizer in optimizers:
            optimizer.zero_grad()
        model.zero_grad()
        if configs.combine_method == 'linear':
            linear.zero_grad()
        
        # training the lstm
        if configs.train_lstm and configs.use_lstm:
            assert(optimizers_lstm != None)
            for optimizer in optimizers_lstm:
                optimizer.step()
            for optimizer in optimizers_lstm:
                optimizer.zero_grad()
            lstm.zero_grad()
    
    log = {'loss': loss.cpu().detach()}
    
    if train:
        return log, model, linear, weight, lstm
    else:
        return log
    



def get_knowledge_states_for_generator(lstm, lstm_inputs, students, timesteps, configs):
    '''
    used during ***inference (generation) time*** to get a student's knowledge state
    '''
    if configs.use_lstm:
        # get lstm inputs
        lstm_ins = [lstm_inputs[s] for s in students]
        if configs.kt_model == 'lstm':
            max_len = max(len(i) for i in lstm_ins)
            padded_lstm_ins = [i + [torch.zeros(i[0].shape[0])]*(max_len - len(i)) for i in lstm_ins]
            padded_lstm_ins = torch.stack([torch.stack(x, dim=0) for x in padded_lstm_ins], dim=1).float() # dim=T*B*D
        
        # get knowledge states
        if configs.kt_model == 'lstm':
            if configs.lstm_init == 'rand':
                hidden_h, hidden_c = torch.rand(1, padded_lstm_ins.shape[1], configs.lstm_hid_dim).cuda(), torch.rand(1, padded_lstm_ins.shape[1], configs.lstm_hid_dim).cuda()
            elif configs.lstm_init == 'zero':
                hidden_h, hidden_c = torch.zeros(1, padded_lstm_ins.shape[1], configs.lstm_hid_dim).cuda(), torch.zeros(1, padded_lstm_ins.shape[1], configs.lstm_hid_dim).cuda()
                
        if configs.train_lstm:
            if configs.kt_model == 'lstm':
                out, hidden = lstm(padded_lstm_ins.cuda(), (hidden_h, hidden_c)) # shape = T*B*D
            
            else:
                input_q = np.vstack([l['q_data'] for l in lstm_ins])
                input_c = np.vstack([l['c_data'] for l in lstm_ins])
                input_q = torch.from_numpy(input_q).long().cuda()
                input_c = torch.from_numpy(input_c).long().cuda()
                if configs.kt_model == 'dkvmn':
                    input_q = torch.transpose(input_q, 0,1) # seqlen, BS
                    input_c = torch.transpose(input_c, 0,1)
                with torch.no_grad():
                    q_embed_data = lstm.q_embed(input_q)  # BS, seqlen, d_model#, c_ct
                qa_embed_data = lstm.qa_embed(input_c)
                
                if configs.kt_model == 'akt':
                    qa_embed_data = lstm.linear_qa(qa_embed_data)
                    out = lstm.model(q_embed_data, qa_embed_data)  # 211x512
                    out = torch.transpose(out, 0, 1)

                elif configs.kt_model == 'dkvmn':
                    memory_value = lstm.init_memory_value[None, :, :].expand(input_q.shape[1], -1, -1)
                    init_memory_key = lstm.init_memory_key
                    lstm.seqlen = input_q.size(0)
                    mem = lstm.memory
                    value_read_content_l = []
                    input_embed_l = []
                    for i in range(lstm.seqlen):
                        # Attention
                        q = q_embed_data[i]
                        correlation_weight = mem.attention(q, init_memory_key)

                        # Read Process
                        # Shape (batch_size, memory_state_dim)
                        read_content = mem.read(memory_value, correlation_weight)

                        # set_trace()
                        # save intermedium data
                        value_read_content_l.append(read_content[None, :, :])
                        input_embed_l.append(q[None, :, :])

                        # Write Process
                        qa = qa_embed_data[i]
                        memory_value = mem.write(qa, memory_value, correlation_weight)

                    all_read_value_content = torch.cat(value_read_content_l, dim=0)
                    if configs.kt_model == 'akt':
                        out = torch.transpose(all_read_value_content, 0, 1) # seqlen, BS, dim
                    elif configs.kt_model == 'dkvmn':
                        out = all_read_value_content
        else:
            with torch.no_grad():
                out, hidden = lstm(padded_lstm_ins.cuda(), (hidden_h, hidden_c)) # shape = T*B*D
        ks = out[timesteps, list(range(out.shape[1])), :] # extract the hidden states --> shape=B*D
    
    else:
        assert(configs.combine_method in ['exp_decay', 'kc_sim_decay', 'exp_kc_decay', 'no_decay'])
        if configs.combine_method == 'no_decay':
            ks = []
            for b in range(len(timesteps)):
                code_embs = torch.tensor(lstm_inputs[students[b]]['code_emb'][timesteps[b]]).cuda()
                ks.append( code_embs )
            return ks, None
        # compute the exponential decay factors
        if 'exp_' in configs.combine_method:
            exp_weights = []
            for b in range(len(timesteps)):
                w = torch.exp(-torch.arange(timesteps[b]+1).flip(dims=[0]).float())
                w = w / w.sum() # normalize
                exp_weights.append(w) # exponential decay for each student, up til t
        if 'kc_' in configs.combine_method:
            kc_weights = []
            prompt_kc = [lstm_inputs[s]['prompt_kc'] for s in students]
            for b in range(len(timesteps)):
                # set_trace()
                w = torch.tensor([jaccard_score(prompt_kc[b][i], prompt_kc[b][timesteps[b]+1]) for i in range(timesteps[b]+1)])
                w = w / w.sum() if w.sum() != 0 else torch.zeros_like(w) # normalize; do not normalize 0
                kc_weights.append(w)

        # compute the weights
        if configs.combine_method == 'exp_decay':
            weights = exp_weights
        elif configs.combine_method == 'kc_sim_decay':
            weights = kc_weights
        elif configs.combine_method == 'exp_kc_decay':
            weights = [exp_weights[i] * kc_weights[i] for i in range(len(exp_weights))]
            weights = [w / w.sum() if w.sum() != 0 else torch.zeros_like(w) for w in weights] # normalize
        elif configs.combine_method == 'no_decay':
            weights = []
            for b in range(len(timesteps)):
                w = torch.tensor([0 for i in range(timesteps[b]+1)])
                w[timesteps[b]] = 1
                weights.append(w)
        weights = [w.unsqueeze(0).to(torch.float32).cuda() for w in weights]
        # compute the weighted knowledge code embeddings, until the current time step
        ks = []
        for b in range(len(timesteps)):
            code_embs = torch.tensor(np.stack(lstm_inputs[students[b]]['code_emb'][:timesteps[b]+1])).cuda()
            ks.append( torch.mm(weights[b], code_embs) )
        ks = torch.cat(ks)
        out = None
        
    return ks, out


def assemble_generator_input(model, lstm, linear, weight, configs,
                             generator_input_ids, prompt_id_lens, lstm_inputs, students, timesteps, kc_vecs=None):
    '''
    linear: linear transform the knowledge state before adding in with the generator input
    weight: weight to apply to the knowledge state before adding in with the generator input
    '''
    
    # compute generator embeddings for the batch
    generator_input_wte = model.transformer.wte(generator_input_ids.cuda()) # shape=B*T*D
    
    # get knowledge states
    ks, out = get_knowledge_states_for_generator(lstm, lstm_inputs, students, timesteps, configs)
    
    # combine kc with generator input
    for i in range(len(students)):
        if configs.combine_method == 'add':
            generator_input_wte[i, :prompt_id_lens[i]] += configs.combine_weight * ks[i]
        elif configs.combine_method == 'average':
            generator_input_wte[i, :prompt_id_lens[i]] = (generator_input_wte[i, :prompt_id_lens[i]] + \
                                                        configs.combine_weight * ks[i]) / 2
        elif configs.combine_method == 'weight':
            generator_input_wte[i, :prompt_id_lens[i]] += weight * ks[i]
        elif configs.combine_method == 'linear':
            if configs.use_kc:
                assert(kc_vecs[i] is not None)
                generator_input_wte[i, :prompt_id_lens[i]] += linear( torch.cat((ks[i], kc_vecs[i].cuda()), dim=-1) )
            else:
                generator_input_wte[i, :prompt_id_lens[i]] += linear(ks[i])      
        elif configs.combine_method in ['exp_decay', 'kc_sim_decay', 'exp_kc_decay', 'no_decay']:
            generator_input_wte[i, :prompt_id_lens[i]] += linear(ks[i])  
    
    # testing 
    if configs.testing:
        # check whether the eos token that separates prompt and code does not get added with knowledge state
        for i in range(len(students)):
            assert(torch.equal(generator_input_wte[i,prompt_id_lens[i]], model.transformer.wte(generator_input_ids[i,prompt_id_lens[i]].cuda())))
    
    return generator_input_wte, ks