import os
import pickle
from datetime import datetime
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import yaml
from munch import Munch
import neptune.new as neptune
from collections import defaultdict

from data_loader import *
from model import *
from trainer import *
from utils import *

import time

## training options
mydict = yaml.safe_load(open("configs_student_model.yaml", "r"))
configs = Munch(mydict)


def main(configs):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_random_seed(configs.seed)

    if configs.use_neptune:
        run = neptune.init(
            project=configs.neptune_project,
            api_token=configs.neptune_api,
            capture_hardware_metrics = False,
            name=configs.exp_name + '_{}'.format(now), # mark the experiment using the current date and time
            custom_run_id=configs.exp_name + '_{}'.format(now),
            tags=[configs.exp_name, now],
        )  # your credentials
        run["parameters"] = vars(configs)
        run['time'] = now

    if configs.save_model:
        save_dir = os.path.join(configs.model_save_dir, now)
        os.mkdir(save_dir)
    
    ## load data
    collate_fn = CollateForLSTM(None)
    train_students, valid_students, test_students, dataset = read_data(configs)
    train_loader = make_dataloader(None, dataset, train_students, collate_fn=collate_fn, configs=configs)
    valid_loader = make_dataloader(None, dataset, valid_students, collate_fn=collate_fn, configs=configs, train=False)
    test_loader  = make_dataloader(None, dataset, test_students,  collate_fn=collate_fn, configs=configs, train=False)
    
    ## load model
    lstm, classifier = create_lstm_model(configs)  
    lstm.cuda();
    classifier.cuda();

    ## optimizers and loss function
    optimizer1 = optim.RMSprop(lstm.parameters(), lr=configs.lstm_lr, momentum=0.9)
    optimizer2 = optim.Adam(classifier.parameters(), lr=configs.cls_lr)
    schedulers = []
    if configs.use_scheduler_lstm:
        scheduler1 = ReduceLROnPlateau(optimizer1, 'min', factor=configs.scheduler_lstm_factor, patience=5)
        schedulers.append(scheduler1)
    if configs.use_scheduler_classifier:
        scheduler2 = ReduceLROnPlateau(optimizer2, 'min', factor=configs.scheduler_classifier_factor, patience=5)
        schedulers.append(scheduler2)
    if configs.lstm_train:
        optimizers = [optimizer1, optimizer2]
    else:
        optimizers = [optimizer2]
    if configs.label_type == 'binary':
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
        

    ## start training
    best_valid_metrics =  {'loss': float('inf'), 'acc': float('-inf'), 'auc': float('-inf')}
    best_test_metrics =  {'loss': float('inf'), 'acc': float('-inf'), 'auc': float('-inf')}
    best_metrics_with_valid =  {'loss': float('inf'), 'acc': float('-inf'), 'auc': float('-inf')}

    for ep in range(configs.epochs):
        train_logs, test_logs,valid_logs = [], [], []

        # training
        for idx, batch in enumerate(train_loader):
            train_log, lstm, classifier = lstm_step(batch, lstm, classifier, 
                                                    configs.lstm_hid_dim, 
                                                    optimizers, loss_fn, train=True, 
                                                    init=configs.lstm_init,
                                                    use_scheduler=configs.use_scheduler, 
                                                    schedulers=schedulers, configs=configs)
            train_logs.append(train_log)

        # validation
        print('validation ...')
        for idx, batch in enumerate(valid_loader):
            valid_log = lstm_step(batch, lstm, classifier, 
                                        configs.lstm_hid_dim, 
                                        None, loss_fn, train=False, 
                                        init=configs.lstm_init, 
                                        use_scheduler=None, schedulers=None, configs=configs)
            valid_logs.append(valid_log)

            
        # testing
        print('testing ...')
        for idx, batch in enumerate(test_loader):
            test_log =  lstm_step(batch, lstm, classifier, 
                                    configs.lstm_hid_dim, 
                                    None, loss_fn, train=False, 
                                    init=configs.lstm_init, 
                                    use_scheduler=None, schedulers=None, configs=configs)
            test_logs.append(test_log)
        
        # logging
        train_logs = aggregate_metrics(train_logs)
        valid_logs = aggregate_metrics(valid_logs)
        test_logs  = aggregate_metrics(test_logs )
        
        for key in valid_logs:
            if key == 'loss':
                if float(valid_logs[key])<best_valid_metrics[key]:
                    best_valid_metrics[key] = float(valid_logs[key])
                    best_metrics_with_valid[key] = float(test_logs[key])
                    # Save 
                    if configs.save_model:
                        if configs.use_neptune:
                            run["best_model_at_epoch"].log(ep)
                        torch.save(lstm, os.path.join(save_dir, 'student_model'))
                        torch.save(classifier, os.path.join(save_dir, 'classifier'))
            else:
                if key != 'pred':
                    if float(valid_logs[key])>best_valid_metrics[key]:
                        best_valid_metrics[key] = float(valid_logs[key])
                        best_metrics_with_valid[key] = float(test_logs[key])

        for key in test_logs:
            if key == 'loss':
                if float(test_logs[key])<best_test_metrics[key]:
                    best_test_metrics[key] = float(test_logs[key])
            else:
                if key != 'pred':
                    if float(test_logs[key])>best_test_metrics[key]:
                        best_test_metrics[key] = float(test_logs[key])

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

    # kill the neptune run
    if configs.use_neptune:
        time.sleep(5)
        run.stop()


if __name__ == "__main__":
    main(configs)