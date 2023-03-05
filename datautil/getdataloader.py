# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import sklearn.model_selection as ms
from torch.utils.data import DataLoader, Dataset

import datautil.imgdata.util as imgutil
from datautil.imgdata.imgdataload import ImageDataset
from datautil.mydataloader import InfiniteDataLoader
from utils.util import log_print, train_valid_target_eval_names


def get_img_dataloader(args):
    '''
    Outputs:
    train_loaders: list. Each element is a dataloader for a source domain's training data.
    val_loaders: list. [source domain train dataloaders + target domain dataloaders + source domain test dataloaders]
    eval_name_dict: dictinonaty. keys: ['train', 'valid', 'target'], store the index of corresponding data in val_loaders

    e.g. PACS data. test_envs = []
    train_loaders: [training dataloader of 'Art', training dataloader of 'cartoon', training dataloader of 'photo', training dataloader of 'sketch']
    val_loaders: [training dataloader of 'Art', training dataloader of 'cartoon', training dataloader of 'photo', training dataloader of 'sketch',
                  test dataloader of 'Art',  test dataloader of 'cartoon', test dataloader of 'photo', test dataloader of 'sketch']
    eval_name_dict: ['train': [0,1,2,3], 'valid':[4,5,6,7], 'target':[]]
    task_sequence_name: ['Art', 'cartoon', 'photo', 'sketch']

    e.g. PACS data. test_envs = [0]
    train_loaders:  [training dataloader of 'cartoon', training dataloader of 'photo', training dataloader of 'sketch']
    val_loaders: [training dataloader of 'cartoon', training dataloader of 'photo', training dataloader of 'sketch', dataloader of 'art painting', test dataloader of 'cartoon', test dataloader of 'photo', test dataloader of 'sketch']

    dataloader return: images, class_label, domain_label     (datautil.imgdata.imgdataload.ImageDataset)
    images: torch tensor (batch, 3, 224, 224)
    class_label: torch tensor (batch,)
    domain_label: torch tensor (batch,)
    Note that when alg is consup and forAug is None(the case of using original supervised contrastive loss, images is return images is [batch_size*2, C, H, W], batch_size*2 is concatenate of two imgutil.image_train transform of the same original image.
    '''
    rate = 0.2            # test data rate
    trdatalist, tedatalist = [], []
    train_name_list, target_name_list = [], []

    names = args.img_dataset[args.dataset]
    args.domain_num = len(names)
    
    eval_name_dict = train_valid_target_eval_names(args)   # keys: train, valid, target
    args.eval_name_dict = eval_name_dict
    args.test_envs = args.order[1:]
    
    for i, domian_id in enumerate(args.order):
        if i == 0:
            tmpdatay = ImageDataset(args, args.task, args.data_dir,
                                    names[domian_id], domian_id, transform=imgutil.image_train(args), test_envs=args.test_envs).labels
            l = len(tmpdatay)
            if args.split_style == 'strat':
                indexall = np.arange(l)
                stsplit = ms.StratifiedShuffleSplit(
                    2, test_size=rate, train_size=1-rate, random_state=args.seed)
                stsplit.get_n_splits(indexall, tmpdatay)
                indextr, indexte = next(stsplit.split(indexall, tmpdatay))
            else:
                indexall = np.arange(l)
                np.random.seed(args.seed)
                np.random.shuffle(indexall)
                ted = int(l*rate)
                indextr, indexte = indexall[:-ted], indexall[-ted:]
                
            trdatalist.append(ImageDataset(args, args.task, args.data_dir,
                                           names[domian_id], domian_id, transform=imgutil.image_train_source(args), indices=indextr, test_envs=args.test_envs))
            tedatalist.append(ImageDataset(args, args.task, args.data_dir,
                                        names[domian_id], domian_id, transform=imgutil.image_test(args), indices=indexte, test_envs=args.test_envs))
        
        else:
            trdatalist.append(ImageDataset(args, args.task, args.data_dir,
                                           names[domian_id], domian_id, transform=imgutil.image_train(args), test_envs=args.test_envs))
            tedatalist.append(ImageDataset(args, args.task, args.data_dir,
                                           names[domian_id], domian_id, transform=imgutil.image_test(args), test_envs=args.test_envs))
        train_name_list.append(names[domian_id])
            
    # If use for InfiniteDataloader, it will fetch data recurrently.
    train_loaders = [InfiniteDataLoader(
         dataset=env,
         weights=None,
         batch_size=args.batch_size,
         num_workers=args.N_WORKERS)
         for env in trdatalist]

    eval_loaders = [DataLoader(
        dataset=env,
        batch_size=args.batch_size*2,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for env in trdatalist+tedatalist]
    
    log_print('domain training tasks sequence: {}, corresponding data size: {}'.format(train_name_list, [len(d.dataset) for d in train_loaders]), args.log_file)
    log_print('domain validation data size: {}\n'.format([len(eval_loaders[i].dataset) for i in eval_name_dict['valid']]), args.log_file)

    return train_loaders, eval_loaders, eval_name_dict, train_name_list

# def get_img_dataloader(args):
#     '''
#     Outputs:
#     train_loaders: list. Each element is a dataloader for a source domain's training data.
#     val_loaders: list. [source domain train dataloaders + target domain dataloaders + source domain test dataloaders]
#     eval_name_dict: dictinonaty. keys: ['train', 'valid', 'target'], store the index of corresponding data in val_loaders

#     e.g. PACS data. test_envs = []
#     train_loaders: [training dataloader of 'Art', training dataloader of 'cartoon', training dataloader of 'photo', training dataloader of 'sketch']
#     val_loaders: [training dataloader of 'Art', training dataloader of 'cartoon', training dataloader of 'photo', training dataloader of 'sketch',
#                   test dataloader of 'Art',  test dataloader of 'cartoon', test dataloader of 'photo', test dataloader of 'sketch']
#     eval_name_dict: ['train': [0,1,2,3], 'valid':[4,5,6,7], 'target':[]]
#     task_sequence_name: ['Art', 'cartoon', 'photo', 'sketch']

#     e.g. PACS data. test_envs = [0]
#     train_loaders:  [training dataloader of 'cartoon', training dataloader of 'photo', training dataloader of 'sketch']
#     val_loaders: [training dataloader of 'cartoon', training dataloader of 'photo', training dataloader of 'sketch', dataloader of 'art painting', test dataloader of 'cartoon', test dataloader of 'photo', test dataloader of 'sketch']

#     dataloader return: images, class_label, domain_label     (datautil.imgdata.imgdataload.ImageDataset)
#     images: torch tensor (batch, 3, 224, 224)
#     class_label: torch tensor (batch,)
#     domain_label: torch tensor (batch,)
#     Note that when alg is consup and forAug is None(the case of using original supervised contrastive loss, images is return images is [batch_size*2, C, H, W], batch_size*2 is concatenate of two imgutil.image_train transform of the same original image.
#     '''
#     rate = 0.2            # test data rate
#     trdatalist, tedatalist = [], []
#     train_name_list, target_name_list = [], []

#     names = args.img_dataset[args.dataset]
#     args.domain_num = len(names)
#     for i in range(len(names)):
#         if i in args.test_envs:
#             tedatalist.append(ImageDataset(args, args.task, args.data_dir,
#                                            names[i], i, transform=imgutil.image_test(args.dataset), test_envs=args.test_envs))
#             target_name_list.append(names[i])
#         else:
#             tmpdatay = ImageDataset(args, args.task, args.data_dir,
#                                     names[i], i, transform=imgutil.image_train(args), test_envs=args.test_envs).labels
#             l = len(tmpdatay)
#             if args.split_style == 'strat':
#                 indexall = np.arange(l)
#                 stsplit = ms.StratifiedShuffleSplit(
#                     2, test_size=rate, train_size=1-rate, random_state=args.seed)
#                 stsplit.get_n_splits(indexall, tmpdatay)
#                 indextr, indexte = next(stsplit.split(indexall, tmpdatay))
#             else:
#                 indexall = np.arange(l)
#                 np.random.seed(args.seed)
#                 np.random.shuffle(indexall)
#                 ted = int(l*rate)
#                 indextr, indexte = indexall[:-ted], indexall[-ted:]

#             if i != args.order[0]:    # use all target domain data for training and testing
#                 all_index = np.append(indextr, indexte)
#                 trdatalist.append(ImageDataset(args, args.task, args.data_dir,
#                                            names[i], i, transform=imgutil.image_train(args), indices=all_index, test_envs=args.test_envs))
#                 tedatalist.append(ImageDataset(args, args.task, args.data_dir,
#                                            names[i], i, transform=imgutil.image_test(args), indices=all_index, test_envs=args.test_envs))         
#             else:
#                 trdatalist.append(ImageDataset(args, args.task, args.data_dir,
#                                            names[i], i, transform=imgutil.image_train(args), indices=indextr, test_envs=args.test_envs))
#                 tedatalist.append(ImageDataset(args, args.task, args.data_dir,
#                                            names[i], i, transform=imgutil.image_test(args), indices=indexte, test_envs=args.test_envs))
#             train_name_list.append(names[i])
#             # test_name_list.append(names[i])

#     # If use for InfiniteDataloader, it will fetch data recurrently.
#     train_loaders = [InfiniteDataLoader(
#          dataset=env,
#          weights=None,
#          batch_size=args.batch_size,
#          num_workers=args.N_WORKERS)
#          for env in trdatalist]
    
#     # if use DataLoader instead of InfiniteDataLoader, accuracy will decrease and training time will largely increase.
#     # train_loaders = [DataLoader(
#     #    dataset=env,
#     #    batch_size=args.batch_size,
#     #    shuffle=True,
#     #    num_workers=args.N_WORKERS)
#     #    for env in trdatalist]

#     eval_loaders = [DataLoader(
#         dataset=env,
#         batch_size=args.batch_size*2,
#         num_workers=args.N_WORKERS,
#         drop_last=False,
#         shuffle=False)
#         for env in trdatalist+tedatalist]

#     eval_name_dict = train_valid_target_eval_names(args)   # keys: train, valid, target
#     train_loaders = change_order(args.order, train_loaders)
#     train_name_list = change_order(args.order, train_name_list)
#     eval_name_dict = change_eval_order(args.order, eval_name_dict)
#     args.eval_name_dict = eval_name_dict

#     log_print('domain training tasks sequence: {}, corresponding data size: {}'.format(train_name_list, [len(d.dataset) for d in train_loaders]), args.log_file)
#     log_print('domain validation data size: {}\n'.format([len(eval_loaders[i].dataset) for i in eval_name_dict['valid']]), args.log_file)
#     # log_print('target domain data: {}, corresponding data size: {}'.format(target_name_list, [len(eval_loaders[i].dataset) for i in eval_name_dict['target']]), args.log_file)

#     return train_loaders, eval_loaders, eval_name_dict, train_name_list

# def change_order(order, original_list):
#     '''
#     change training domain order based on args.order
#     e.g. 
#     original_list = [a,b,c], order = [2,1,0]
#     new_original_list = [c,b,a]
#     '''
#     new_list = []
#     for i in order:
#         new_list.append(original_list[i])
#     return new_list

# def change_eval_order(order, eval_name_dict):
#     eval_name_dict['train'] = change_order(order, eval_name_dict['train'])
#     eval_name_dict['valid'] = change_order(order, eval_name_dict['valid'])
#     return eval_name_dict


# class utilDataset(Dataset):
#     '''
#     construct pseudo dataset
#     input: images_dict.
#     '''
#     def __init__(self, images_dict, class_labels, domain_labels, loader, transform=None, target_transform=None):
#         self.x = images_dict                 # list of [PIL image]
#         self.labels = class_labels           # numpy array
#         self.dlabels = domain_labels         # numpy array
#         self.loader = loader
#         self.transform = transform
    
#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, index):
#         imgs = self.transform(self.loader(self.x[index])) if self.transform is not None else self.loader(self.x[index])
#         return imgs, self.labels[index], self.dlabels[index] 

#     def get_raw_data(self):
#         return self.x, self.labels, self.dlabels