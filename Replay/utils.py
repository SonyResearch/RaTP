import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

import datautil.imgdata.util as imgutil
from datautil.mydataloader import InfiniteDataLoader

class ExemplarDataset(Dataset):
    '''
    Used for compute_class_mean
    input: imgs should be PIL image.
    '''
    def __init__(self, imgs, transform):
        self.imgs = imgs
        self.transform = transform
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        return self.transform(self.imgs[index])
        # return self.transform(Image.fromarray(self.imgs[index]))

class ReplayDataset(Dataset):
    '''
    construct replay dataset
    input: imgs should be PIL image.
    '''
    def __init__(self, images, class_labels, domain_labels, transform=None, target_transform=None):
        self.images = images
        self.labels = class_labels
        self.dlabels = domain_labels
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        imgs = self.transform(self.images[index]) if self.transform is not None else self.images[index]
        return imgs, self.labels[index], self.dlabels[index]
    
    def get_raw_data(self):
        return self.images, self.labels, self.dlabels

def concat_list(data_list):
    '''
    flatten list
    input: list of list [[..], .., [..]]
    return list [..]
    '''
    datas = []
    for l in data_list:
        for i in l:
            datas.append(i)
    return datas
