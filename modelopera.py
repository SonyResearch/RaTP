# coding=utf-8
import torch
from network import img_network


def get_fea(args):
    if args.net.startswith('vgg'):
        net = img_network.VGGBase(args)
    elif args.net == 'LeNet':
        net = img_network.LeNetBase()
    elif args.net == 'DTN':
        net = img_network.DTNBase()
    elif args.net.startswith('res'):
        net = img_network.ResBase(args)
    else:
        net = img_network.VGGBase(args)
    return net


def accuracy(network, loader):
    correct = 0
    total = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            p = network(x)

            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()
            total += len(x)
    network.train()
    return correct / total
