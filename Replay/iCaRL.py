import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import functional as F

import Replay.utils as utils
import datautil.imgdata.util as imgutil
from utils.util import log_print

class iCaRL:
    def __init__(self, args):
        self.args = args
        self.exemplar_set = []   # list of list[PIL image] : [[exemplar1 PIL image], [exemplar2 PIL image]...]
        self.exemplar_label_set = []  # list of np.array : [array(exemplar1 labels), array(exemplar2 labels)...]
        self.exemplar_dlabel_set = []
        self.replay_dataset = None

    def update_dataloader(self, dataloader=None):
        exemplar_set = self.exemplar_set
        exemplar_label_set = self.exemplar_label_set
        exemplar_dlabel_set = self.exemplar_dlabel_set
        log_print('exemplar_set size: {}'.format(len(exemplar_set[0]) if len(exemplar_set)>0 else 0), self.args.log_file)
        replay_dataloader = None

        if len(exemplar_set) > 0:
            imgs = utils.concat_list(exemplar_set)
            labels = utils.concat_list(exemplar_label_set)
            dlabels = utils.concat_list(exemplar_dlabel_set)
            self.replay_dataset = utils.ReplayDataset(imgs, labels, dlabels, transform=imgutil.image_train(self.args))
            
        return self.replay_dataset

    def update(self, model, task_id, dataloader):
        if self.args.replay_mode == 'class':    # exemplar for each class and domain
            m=int(self.args.memory_size / (self.args.num_classes * (task_id+1)))
        elif self.args.replay_mode == 'domain':    # exemplar for each domain
            m=int(self.args.memory_size / (task_id+1))
        self._reduce_exemplar_sets(m)

        image_dict, class_label, domain_label = dataloader.dataset.get_raw_data()
        images = [dataloader.dataset.loader(dict) for dict in image_dict]       # list of PIL image

        if self.args.replay_mode == 'class':  # each exemplar contains data of one class in one specific doamin
            for c in range(self.args.num_classes):
                indices = np.where(class_label == c)[0]
                if len(indices) == 0:
                    log_print('No class {} pseudo labels!!!'.format(c), self.args.log_file)
                    continue
                imgs = [images[i] for i in indices]         # list of PIL image
                clabel = class_label[class_label == c]
                dlabel = domain_label[class_label == c]
                self._construct_exemplar_set(model, imgs, clabel, dlabel, m)
        elif self.args.replay_mode == 'domain':  # each exemplar contains data of all classes in one specific doamin
            self._construct_exemplar_set(model, images, class_label, domain_label, m)
    
    def _construct_exemplar_set(self, model, images, class_label, domain_label, m):
        '''
        construct exemplar for each class in each domain
        input images should be one class in one specific domain
        '''
        class_mean, feature_extractor_output = self.compute_class_mean(model, images, transform=imgutil.image_test(self.args))
        exemplar = []
        exemplar_index = []

        now_class_mean = np.zeros((1, model.featurizer.in_features))   # feature extracter output dimension
     
        for i in range(m):
            
            #icarl code
            # shape：batch_size*256
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            # shape：batch_size
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]

            # make sure selected example won't be selected again
            # if index in exemplar_index:
            #     raise ValueError("Exemplars should not be repeated!!!!")
            exemplar.append(images[index])
            exemplar_index.append(index)
            feature_extractor_output[index] += 10000 

        self.exemplar_set.append(exemplar)
        self.exemplar_label_set.append(class_label[exemplar_index])
        self.exemplar_dlabel_set.append(domain_label[exemplar_index])


    def _reduce_exemplar_sets(self, m):
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]
        for index in range(len(self.exemplar_label_set)):
            self.exemplar_label_set[index] = self.exemplar_label_set[index][:m]
        for index in range(len(self.exemplar_dlabel_set)):
            self.exemplar_dlabel_set[index] = self.exemplar_dlabel_set[index][:m]
        
    
    def compute_class_mean(self, model, images, transform):
        exemplar_dataset = utils.ExemplarDataset(images, transform)
        exemplar_dataloader = DataLoader(dataset=exemplar_dataset,
                                        shuffle=False,
                                        batch_size=self.args.batch_size,
                                        num_workers=self.args.N_WORKERS)
        model.eval()     # if not use this, it will affect evaluation steps after this evaluation, even they call model.eval().
        feature_extractor_outputs = []
        for i, x in enumerate(exemplar_dataloader):
            x = x.cuda()
            with torch.no_grad():
                feature_extractor_outputs.append(model.featurizer(x))
        feature_extractor_outputs = torch.cat(feature_extractor_outputs, dim=0)
        model.train()
        feature_extractor_outputs = F.normalize(feature_extractor_outputs.detach()).cpu().numpy()
        class_mean = np.mean(feature_extractor_outputs, axis=0)
        return class_mean, feature_extractor_outputs
