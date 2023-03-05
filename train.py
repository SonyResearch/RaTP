import numpy as np
from tqdm import tqdm
import torch

import modelopera
from utils.util import log_print
from pLabel import assign_pseudo_label
from datautil.mydataloader import InfiniteDataLoader

def train(args, model, old_model, task_id, dataloader, replay_dataset, eval_loaders, eval_name_dict):
    acc_record = {}
    all_val_acc_record = {}
    for tid in range(len(eval_name_dict['valid'])):
        all_val_acc_record['task{}'.format(tid)] = []
    best_valid_acc, target_acc = 0, 0

    max_epoch = args.max_epoch 
    model.get_optimizer(lr_decay=args.lr_decay1 if task_id > 0 else 1.0)
    model.optimizer = op_copy(model.optimizer)

    with tqdm(range(max_epoch)) as tepoch:
        tepoch.set_description(f"Task {task_id}")
        for epoch in tepoch:

            # progressly assign pseudo label
            if epoch % args.pseudo_fre == 0:
                pseudo_dataloader, plabel_sc = assign_pseudo_label(args, dataloader, replay_dataset, task_id, model, epoch)
                curr_dataloader = cat_pseudo_replay(args, pseudo_dataloader, replay_dataset)
                replay_dataloader = None
            
            model.naug = 0 if task_id > 0 else args.batch_size*args.steps_per_epoch
            for iter_ in range(args.steps_per_epoch):     # make sure each tasks has the same training iters. 
                minibatches = [(data) for data in next(iter(curr_dataloader))]     
                if minibatches[0].size(0) == 1: 
                    continue

                model.train()
                if task_id == 0:
                    step_vals = model.train_source(minibatches, task_id, epoch)
                else:
                    step_vals = model.adapt(minibatches, task_id, epoch, replay_dataloader, old_model)

            model.optimizer = lr_scheduler(model.optimizer, epoch, max_epoch)

            # only calculate accuracy of current domain
            for item in ['train', 'valid']:     
                acc_record[item] = np.mean(np.array([modelopera.accuracy(model, eval_loaders[eval_name_dict[item][task_id]])]))
            if plabel_sc is None:
                tepoch.set_postfix(**step_vals, **acc_record, naug=model.naug/(args.batch_size*args.steps_per_epoch))
            else: 
                tepoch.set_postfix(**step_vals, **acc_record, naug=model.naug/(args.batch_size*args.steps_per_epoch))

            # record accuracy of validation data of all tasks along epochs.
            for tid in range(len(eval_name_dict['valid'])):
                all_val_acc_record['task{}'.format(tid)].append(modelopera.accuracy(model, eval_loaders[eval_name_dict['valid'][tid]]))

            if acc_record['valid'] > best_valid_acc:
                best_valid_acc = acc_record['valid']
    
    log_print('task{} training result on max_epoch{}: {} {}'.format(task_id, max_epoch, step_vals, acc_record), args.log_file, p=False)
        
    return model, all_val_acc_record, pseudo_dataloader


def cat_pseudo_replay(args, dataloader, replay_dataset):
    if replay_dataset is not None:
        dataset = torch.utils.data.ConcatDataset([dataloader.dataset, replay_dataset])    
        dataloader = InfiniteDataLoader(dataset=dataset, weights=None, batch_size=args.batch_size, num_workers=args.N_WORKERS)
    return dataloader

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        # param_group['weight_decay'] = 1e-3
        # param_group['momentum'] = 0.9
        # param_group['nesterov'] = True
    return optimizer