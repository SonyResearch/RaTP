# coding=utf-8
import torch.nn as nn
import numpy as np


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

def freeze_proxy(model):
    '''
    freeze the PCL proxy and classifier weight in adaptation step.
    '''
    model.fc_proj.requires_grad = False
    model.classifier.requires_grad = False

def freeze_classifier(model):
    '''
    freeze the classifier of model in adaptation step.
    '''
    for k, v in model.classifier.named_parameters():
        v.requires_grad = False



