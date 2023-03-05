# coding=utf-8
from torch.utils.data import Dataset
import numpy as np
from datautil.util import Nmax
from datautil.imgdata.util import rgb_loader, l_loader
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
import torch


class ImageDataset(object):
    def __init__(self, args, task, root_dir, domain_name, domain_label=-1, labels=None, transform=None, target_transform=None, indices=None, test_envs=[], mode='Default'):
        self.args = args
        self.imgs = ImageFolder(root_dir+domain_name).imgs
        self.domain_num = 0
        self.task = task
        self.dataset = args.dataset
        imgs = [item[0] for item in self.imgs]
        labels = [item[1] for item in self.imgs]
        self.labels = np.array(labels)     # np.array
        self.x = imgs                      # list of file dir
        self.transform = transform
        self.target_transform = target_transform
        if indices is None:
            self.indices = np.arange(len(imgs))
        else:
            self.indices = indices
        if mode == 'Default':
            self.loader = default_loader
        elif mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        self.dlabels = np.ones(self.labels.shape) * \
            (domain_label-Nmax(test_envs, domain_label))    # np.array

    def set_labels(self, tlabels=None, label_type='domain_label'):
        assert len(tlabels) == len(self.x)
        if label_type == 'domain_label':
            self.dlabels = tlabels
        elif label_type == 'class_label':
            self.labels = tlabels

    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)
        else:
            return x

    def __getitem__(self, index):
        index = self.indices[index]
        img = self.input_trans(self.loader(self.x[index]))
        ctarget = self.target_trans(self.labels[index])
        dtarget = self.target_trans(self.dlabels[index])
        return img, ctarget, dtarget

    def __len__(self):
        return len(self.indices)

    def get_raw_data(self):
        img_dict = []
        clabel = []
        dlabel = []
        for i in self.indices:
            img_dict.append(self.x[i])
            clabel.append(self.labels[i])
            dlabel.append(self.dlabels[i])
        return img_dict, np.array(clabel), np.array(dlabel)
