# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import transforms
from modelopera import get_fea
from opt import *
from network.common_network import feat_encoder
import PCA
from RandMix import RandMix
    
def Entropy_(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

class RaTP(torch.nn.Module):

    def __init__(self, args):
        super(RaTP, self).__init__()
        self.args = args
        self.task_id = 0
        self.naug = 0    
        self.fea_rep = None
        self.featurizer = get_fea(args)

        # training algorithm model
        fea_dim = args.proj_dim[args.dataset]
        self.encoder = feat_encoder(args, self.featurizer.in_features, fea_dim)
        self._initialize_weights(self.encoder)
        
        self.classifier = nn.Parameter(torch.FloatTensor(args.num_classes, fea_dim))
        nn.init.kaiming_uniform_(self.classifier, mode='fan_out', a=math.sqrt(5))
                    
        # Data augment algorithm
        self.data_aug = RandMix(1).cuda()
        if args.dataset == 'dg5':
            self.aug_tran = transforms.Normalize([0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            self.aug_tran = transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def forward(self, x):
        x = self.featurizer(x)
        x = self.encoder(x)
        self.fea_rep = x
        pred = F.linear(x, self.classifier)
        return pred

    def get_optimizer(self, lr_decay=1.0):
        self.optimizer = torch.optim.SGD([
            {'params': self.featurizer.parameters(), 'lr': lr_decay * self.args.lr},
            {'params': self.encoder.parameters()},
        	{'params': self.classifier},
            ], lr=self.args.lr, weight_decay=self.args.weight_decay)
        

################################################## train source and adapt ######################################################################

    def train_source(self, minibatches, task_id, epoch):
        self.task_id = task_id
        all_x = minibatches[0].cuda().float()   
        all_y = minibatches[1].cuda().long()    

        # Data Augmentation using RandMix
        ratio = epoch / self.args.max_epoch
        data_fore = self.aug_tran(torch.sigmoid(self.data_aug(all_x, ratio=ratio)))
        all_x = torch.cat([all_x, data_fore])    # [original, aug]
        all_y = torch.cat([all_y, all_y])

        loss, loss_dict = self.PCAupdate(all_x, all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    
    def adapt(self, minibatches, task_id, epoch, replay_dataloader=None, old_model=None):
        self.task_id = task_id
        all_x = minibatches[0].cuda().float()   
        all_y = minibatches[1].cuda().long()    

        # Data Augmentation using RandMix
        all_x, all_y = self.select_aug(all_x, all_y, epoch)

        loss, loss_dict = self.PCAupdate(all_x, all_y, old_model)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.scheduler.step()
        return {'loss': loss.item()}


################################################################ Algorithms ####################################################

    def PCAupdate(self, all_x, all_y, old_model=None):
        pred = self(all_x)
        
        # cross entropy loss
        loss_cls = F.nll_loss(F.log_softmax(pred, dim=1), all_y)
        
        # pca loss
        proxy = self.classifier
        features = self.fea_rep
        if self.task_id > 0:
            old_proxy = old_model.classifier
            loss_pcl = PCA.PCALoss(self.args.num_classes, self.args.PCL_scale)(features, all_y, proxy, old_proxy, mweight=self.args.MPCL_alpha)
        else:
            loss_pcl = PCA.PCLoss(num_classes=self.args.num_classes, scale=self.args.PCL_scale)(features, all_y, proxy)
        
        loss_dict = {'ce': loss_cls.item(), 'pcl': (self.args.loss_alpha1 * loss_pcl).item()}
        loss = loss_cls + self.args.loss_alpha1 * loss_pcl
        
        # distill loss
        if old_model is not None:
            distill_loss = self.args.distill_alpha * self.distill_loss(pred, all_x, old_model)
            loss += distill_loss
            loss_dict['distill'] = distill_loss.item()

        return loss, loss_dict
    
    def distill_loss(self, pred, all_x, old_model):
        old_model.cuda().eval()
        with torch.no_grad():
            old_logist = nn.Softmax(dim=1)(old_model(all_x))
        
        if self.args.distill == 'CE':
            loss = F.cross_entropy(pred, old_logist)
        elif self.args.distill == 'KL':
            loss = nn.KLDivLoss(reduction="batchmean")(nn.LogSoftmax(dim=1)(pred), old_logist)
        elif self.args.distill == 'feaKL':
            loss = nn.KLDivLoss(reduction="batchmean")(nn.LogSoftmax(dim=1)(self.fea_rep), nn.Softmax(dim=1)(old_model.fea_rep))
        return loss
            
        
################################################################ Utils ####################################################
    def _initialize_weights(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_() 

    def select_aug(self, all_x, all_y, epoch):
        ratio = epoch / self.args.max_epoch
        if self.args.aug_tau > 0:
            self.eval()
            with torch.no_grad():
                pred = nn.Softmax(dim=1)(self(all_x))
                ov, idx = torch.max(pred, 1)
                bool_index = ov > self.args.aug_tau
                data_fore = all_x[bool_index]
                y_fore = all_y[bool_index]
                data_fore = self.aug_tran(torch.sigmoid(self.data_aug(data_fore, ratio=ratio)))
            self.train()
        else:
            data_fore = self.aug_tran(torch.sigmoid(self.data_aug(all_x, ratio=ratio)))
            y_fore = all_y
        all_x = torch.cat([all_x, data_fore])   
        all_y = torch.cat([all_y, y_fore])
        self.naug += len(y_fore)
        return all_x, all_y
