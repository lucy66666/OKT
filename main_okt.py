import os
import pickle
from datetime import datetime
import torch.nn.functional as F
import torch.optim as optim
import transformers

import yaml
import time
from munch import Munch
import neptune.new as neptune
from collections import defaultdict

from data_loader import *
from model import *
from trainer import *
from utils import *
from eval import *

## training options
mydict = yaml.safe_load(open("configs_okt.yaml", "r"))
configs = Munch(mydict)

def main(configs):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_random_seed(configs.seed)

    ## test on smaller portion of dataset
    if configs.testing:
        configs.use_neptune = False
        configs.save_model = False
        configs.epochs = 2
    
    ## use Neptune.ai to track experiments
    if configs.use_neptune:
        run = neptune.init(
            project=configs.neptune_project,
            api_token=configs.neptune_api,
            capture_hardware_metrics = False,
            name=configs.exp_name + '_{}'.format(now), # mark the experiment using the current date and time
            custom_run_id=configs.exp_name + '_{}'.format(now),
            tags=[now],
        )  # your credentials
        run["parameters"] = vars(configs)
        run['time'] = now

    if configs.save_model:
        os.mkdir(os.path.join(configs.model_save_dir, now))


    ## load the init dataset
    train_set, valid_set, test_set, dataset = read_data(configs)

    ## uncomment this part when running for AKT
    # configs.n_questions = len(dataset.ProblemID.unique()) + 1
    # configs.n_solutions = len(dataset.CodeStateID.unique()) + 1

    ## load model
    lstm, classifier, tokenizer, model, linear, weight = create_okt_model(configs)    

    ## load data
    collate_fn = CollateForOKT(tokenizer=tokenizer, configs=configs)
    start = time.time()
    _, train_loader, lstm_inputs = make_dataloader(train_set, dataset, None, 
                                                   collate_fn=collate_fn, 
                                                   configs=configs, do_lstm_dataset=True)
    _, valid_loader = make_dataloader(valid_set, None, None, 
                                      collate_fn=collate_fn, configs=configs, 
                                      do_lstm_dataset=False, train=False)
    _, test_loader  = make_dataloader(test_set , None, None, 
                                      collate_fn=collate_fn, configs=configs, 
                                      do_lstm_dataset=False, train=False)


    ## optimizers and loss function
    optimizers_generator = []
    optimizer = transformers.AdamW(model.parameters(), lr=configs.lr, correct_bias=True)
    optimizers_generator.append(optimizer)
    if configs.combine_method == 'weight':
        optimizer_weight = optim.Adam([weight], lr=configs.lr_weight)
        optimizers_generator.append(optimizer_weight)
    elif configs.combine_method == 'linear':
        optimizer_linear = optim.Adam(linear.parameters(), lr=configs.lr_linear)
        optimizers_generator.append(optimizer_linear)

    ## optimizer for lstm
    if configs.train_lstm and configs.use_lstm:
        optimizers_lstm = []
        if configs.kt_model == 'lstm':
            optimizer1 = optim.RMSprop(lstm.parameters(), lr=configs.lstm_lr, momentum=0.9)
        elif configs.kt_model in {'akt', 'dkvmn'}:
            optimizer1 = optim.Adam(lstm.parameters(), lr=configs.lstm_lr, betas=(0.9, 0.999), eps=1e-8)
        optimizers_lstm.append(optimizer1)
        if configs.use_classifier:
            optimizer2 = optim.Adam(classifier.parameters(), lr=configs.cls_lr)
            optimizers_lstm.append(optimizer2)
    else:
        optimizers_lstm = None

    ## scheduler
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 
                        num_warmup_steps=1000, num_training_steps=configs.epochs*len(train_loader)*5)


    ## start training
    best_valid_metrics =  {'loss': float('inf')} 
    best_test_metrics =  {'loss': float('inf')} 
    best_metrics_with_valid =  {'loss': float('inf')}

    for ep in range(configs.epochs):
        train_logs, test_logs, valid_logs = [], [], []
        
        ## training
        for idx, batch in enumerate(tqdm(train_loader)):
            train_log, model, linear, weight, lstm = generator_step(batch, lstm_inputs,
                                                        model, lstm, linear, weight,
                                                        optimizers_generator, optimizers_lstm,
                                                        configs, train=True, scheduler=scheduler, classifier=classifier)
            
            train_logs.append(train_log)
            ## save results to neptune.ai
            if configs.log_train_every_itr and configs.use_neptune:
                if (idx+1) % configs.log_train_every_itr == 0:
                    itr_train_logs = aggregate_metrics(train_logs)
                    for key in itr_train_logs:
                        run["metrics/train_every_{}_itr/{}".format(configs.log_train_every_itr,key)].log(itr_train_logs[key])
            
        ## validation
        for idx, batch in enumerate(valid_loader):
            valid_log = generator_step(batch, lstm_inputs,
                                            model, lstm, linear, weight,
                                            configs=configs, train=False, classifier=classifier)
            valid_logs.append(valid_log)
            
        ## testing
        for idx, batch in enumerate(test_loader):
            test_log = generator_step(batch, lstm_inputs,
                                            model, lstm, linear, weight,
                                            configs=configs, train=False, classifier=classifier)
            test_logs.append(test_log)
        
        ## logging
        train_logs = aggregate_metrics(train_logs)
        valid_logs = aggregate_metrics(valid_logs)
        test_logs  = aggregate_metrics(test_logs )
        print('train: ', train_logs)
        print('valid: ', valid_logs)
        print('test : ', test_logs )
        print()
        
        ## log the results and save models
        for key in valid_logs:
            ## only one key (loss) available for OKT
            if key == 'loss':
                if float(valid_logs[key])<best_valid_metrics[key]:
                    best_valid_metrics[key] = float(valid_logs[key])
                    best_metrics_with_valid[key] = float(test_logs[key])
                    ## Save the model with lowest validation loss
                    if configs.save_model:
                        if configs.use_neptune:
                            run["best_model_at_epoch"].log(ep)
                        torch.save(lstm, os.path.join(configs.model_save_dir, now, 'lstm'))
                        torch.save(model, os.path.join(configs.model_save_dir, now, 'model'))
                        if configs.use_classifier:
                            torch.save(classifier, os.path.join(configs.model_save_dir, now, 'classifier'))
                        if configs.combine_method == 'weight':
                            torch.save(weight, os.path.join(configs.model_save_dir, now, 'weight'))
                        elif configs.combine_method == 'linear':
                            torch.save(linear, os.path.join(configs.model_save_dir, now, 'linear'))

        for key in test_logs:
            if key == 'loss':
                if float(test_logs[key])<best_test_metrics[key]:
                    best_test_metrics[key] = float(test_logs[key])

        ## save results to neptune.ai
        if configs.use_neptune:
            for key in train_logs:
                run["metrics/train/"+key].log(train_logs[key])
            for key in valid_logs:
                run["metrics/valid/"+key].log(valid_logs[key])
            for key in test_logs:
                run["metrics/test/"+key].log(test_logs[key])
            for key in best_valid_metrics:
                run["metrics/valid/best_"+key].log(best_valid_metrics[key])
            for key in best_test_metrics:
                run["metrics/test/best_"+key].log(best_test_metrics[key])
            for key in best_metrics_with_valid:
                run["metrics/test/best_"+key+"_with_valid"].log(best_metrics_with_valid[key])
            run["epoch"].log(ep)


    ## evaluation after all training epochs
    results = {}
    if configs.save_model:
        ## load best models
        model = torch.load(os.path.join(configs.model_save_dir, now, 'model'))
        lstm = torch.load(os.path.join(configs.model_save_dir, now, 'lstm'))
        if configs.combine_method == 'linear':
            linear = torch.load(os.path.join(configs.model_save_dir, now, 'linear'))
        elif configs.combine_method == 'weight':
            weight = torch.load(os.path.join(configs.model_save_dir, now, 'weight'))            

    ## one by one for testing
    configs.batch_size = 1
    test_dataset, test_loader  = make_dataloader(test_set , None, None, 
                                                    collate_fn=collate_fn, 
                                                    configs=configs, do_lstm_dataset=False)
    assert(len(test_loader) == len(test_dataset)) # pass one data point at a time. 
    generated_codes = []
    ground_truth_codes = []
    prompts = []
    for idx in range(len(test_dataset)):
        generated_code, nll, ground_truth_code, prompt = generate_code(test_dataset, lstm_inputs, tokenizer, 
                                                                    idx, model, lstm, linear, weight, configs)
        generated_codes.append(generated_code)
        ground_truth_codes.append(ground_truth_code)
        prompts.append(prompt)
    
    ## compute codebleu
    codebleu_score, detailed_codebleu_score = compute_code_bleu(ground_truth_codes, generated_codes)
    results['codebleu'] = codebleu_score
    results['detailed_codebleu'] = detailed_codebleu_score
    
    ## compute diversity
    metrics = {'dist_1': Distinct_N(1), 
               'dist_2': Distinct_N(2), 
               'dist_3': Distinct_N(3),
    }
    for i, (name, metric) in enumerate(metrics.items()):
        metric_result = metric.compute_metric(generated_codes)
        results[name] = metric_result
                
    ## save results
    results['generated_codes'] = generated_codes
    results['ground_truth_codes'] = ground_truth_codes
    results['prompts'] = prompts
    if configs.save_model:
        with open(os.path.join(configs.model_save_dir, now, 'eval_logs.pkl'), 'wb') as f:
            pickle.dump(results, f)

    ## write results to neptune
    if configs.use_neptune:
        for idx, (k, v) in enumerate(results.items()):
            run['metrics/test/generation_{}'.format(k)] = v
        
    ## kill the neptune run
    if configs.use_neptune:
        time.sleep(10)
        run.stop()


if __name__ == "__main__":
    main(configs)