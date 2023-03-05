# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier 
import datautil.imgdata.util as imgutil
from datautil.mydataloader import InfiniteDataLoader
from utils.util import log_print
import Replay.utils as RPutils

def assign_pseudo_label(args, dataloader, replay_dataset, taskid, model, epoch, cur=False):
    pseudo_tau = 0 
    if taskid == 0 or args.pLabelAlg == 'ground':
        return dataloader, None
    
    else:
        image_dict, clabel, dlabel = dataloader.dataset.get_raw_data()
        images = [dataloader.dataset.loader(dict) for dict in image_dict]       # list of PIL image

        pseudo_image_dict = []
        pseudo_clabel = []
        pseudo_dlabel = []

        curr_dataset = RPutils.ReplayDataset(images, clabel, dlabel, transform=imgutil.image_test(args))
        curr_dataloader = DataLoader(dataset=curr_dataset,
                                        shuffle=False,
                                        batch_size=args.batch_size,
                                        num_workers=args.N_WORKERS)
        model.eval().cuda()
        pseudo_clabel, pacc_dict, bool_index = T2PL(args, curr_dataloader, model, pseudo_tau)
        for i, v in enumerate(bool_index):
            if v:
                pseudo_image_dict.append(image_dict[i])
                pseudo_dlabel.append(dlabel[i])
        model.train()
        pseudo_dataset = PseudoDataset(pseudo_image_dict, np.array(pseudo_clabel), np.array(pseudo_dlabel), loader=dataloader.dataset.loader, transform=imgutil.image_train(args))
        pseudo_dataloader = InfiniteDataLoader(dataset=pseudo_dataset, weights=None, batch_size=args.batch_size, num_workers=args.N_WORKERS)

        return pseudo_dataloader, pacc_dict #{'ps':len(pseudo_image_dict), 'pc':correct}

def T2PL(args, loader, model, pseudo_tau):
    start_test = True
    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs = data[0].cuda()
            labels = data[1]

            feas = model.encoder(model.featurizer(inputs))
            outputs = F.linear(feas, model.classifier)

            if start_test:
                all_fea = [feas.float().cpu()]
                all_output = [outputs.float().cpu()]
                all_label = [labels.float()]
                start_test = False
            else:
                all_fea.append(feas.float().cpu()) 
                all_output.append(outputs.float().cpu())
                all_label.append(labels.float()) 
    all_fea = torch.cat(all_fea, dim=0)
    all_output = torch.cat(all_output, dim=0)
    all_label = torch.cat(all_label, dim=0)
    
    all_output = nn.Softmax(dim=1)(all_output)
    ov, idx = torch.max(all_output, 1)
    bool_index = ov > pseudo_tau
    all_output = all_output[bool_index]
    all_fea = all_fea[bool_index]
    all_label = all_label[bool_index]
    
    acc_list = []
    
    # softmax predict
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    acc_list.append(accuracy)
    
    all_fea = all_fea / torch.norm(all_fea, p=2, dim=1, keepdim=True)
    
    all_fea = all_fea.float().cpu()  # (N, dim)
    K = all_output.size(1)
    aff = all_output.float().cpu()   # (N, C)
    
    # top k features for SHOT
    topk_num = max(all_fea.shape[0] // (args.num_classes * args.topk_beta), 1) 
    top_aff, top_fea = [], []
        
    for cls_idx in range(args.num_classes):
        feat_samp_idx = torch.topk(aff[:, cls_idx], topk_num)[1]                
        top_fea.append(all_fea[feat_samp_idx, :])        
        top_aff.append(aff[feat_samp_idx, :])
        
    top_aff = torch.cat(top_aff, dim=0).numpy()
    top_fea = torch.cat(top_fea, dim=0).numpy()
    _, top_predict = torch.max(torch.from_numpy(top_aff), 1)
    
    # SHOT      
    for _ in range(2):
        initc = top_aff.transpose().dot(top_fea)  
        initc = initc / (1e-8 + top_aff.sum(axis=0)[:,None])  

        cls_count = np.eye(K)[predict].sum(axis=0)   
        labelset = np.where(cls_count>0)    
        labelset = labelset[0]    

        dd = cdist(all_fea, initc[labelset], args.distance)   
        pred_label = dd.argmin(axis=1)   
        predict = labelset[pred_label]   
        
        top_cls_count = np.eye(K)[top_predict].sum(axis=0)   
        top_labelset = np.where(top_cls_count>0)    
        top_labelset = top_labelset[0]    

        top_dd = cdist(top_fea, initc[top_labelset], args.distance)   
        top_pred_label = top_dd.argmin(axis=1)   
        top_predict = top_labelset[top_pred_label]   

        top_aff = np.eye(K)[top_predict]         
        acc_list.append(np.sum(predict == all_label.float().numpy()) / len(all_fea))
        
    # knn on distance of each features and cluster center
    top_sample = []
    top_label = []
    topk_fit_num = max(all_fea.shape[0] // (args.num_classes * args.topk_beta), 1)
    topk_num = max(all_fea.shape[0] // (args.num_classes * args.topk_alpha), 1)
    
    for cls_idx in range(len(labelset)):     
        feat_samp_idx = torch.topk(torch.from_numpy(dd)[:, cls_idx], topk_fit_num, largest=False )[1]
            
        feat_cls_sample = all_fea[feat_samp_idx, :]
        feat_cls_label = torch.zeros([len(feat_samp_idx)]).fill_(cls_idx)

        top_sample.append(feat_cls_sample)
        top_label.append(feat_cls_label)
    top_sample = torch.cat(top_sample, dim=0).cpu().numpy()
    top_label = torch.cat(top_label, dim=0).cpu().numpy()

    knn = KNeighborsClassifier(n_neighbors=topk_num)
    knn.fit(top_sample, top_label)
    
    knn_predict = knn.predict(all_fea.cpu().numpy()).tolist()
    knn_predict = [int(i) for i in knn_predict]
    
    predict = labelset[knn_predict]
    acc_list.append(np.sum(predict == all_label.float().numpy()) / len(all_fea))
        
    # log_print("acc:" + " --> ".join("{:.3f}".format(acc) for acc in acc_list), args.log_file, p=False)
    acc_dict = {}
    for i in range(len(acc_list)):
        acc_dict['pa{}'.format(i)] = round(acc_list[i],3)

    return predict.astype('int'), acc_dict, bool_index


class PseudoDataset(Dataset):
    '''
    construct pseudo dataset
    input: images path.
    '''
    def __init__(self, images_dict, class_labels, domain_labels, loader, transform=None, target_transform=None):
        self.x = images_dict                 # list of [PIL image path]
        self.labels = class_labels           # numpy array
        self.dlabels = domain_labels         # numpy array
        self.loader = loader
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        imgs = self.transform(self.loader(self.x[index])) if self.transform is not None else self.loader(self.x[index])
        return imgs, self.labels[index], self.dlabels[index] 

    def get_raw_data(self):
        return self.x, self.labels, self.dlabels