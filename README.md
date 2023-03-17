# Deja Vu: Continual Model Generalization for Unseen Domains  
Official Implementation for ICLR 2023 paper: [Deja Vu: Continual Model Generalization for Unseen Domains](https://arxiv.org/pdf/2301.10418.pdf)

![Overview](./fig/overview.jpg)  
RaTP first starts with a labeled source domain, applies RandMix on the full set of source data to generate augmentation data, and uses a simplified version of PCA for model optimization. Then, for continually arriving target domains, RaTP uses T2PL to generate pseudo labels for all unlabeled samples, applies RandMix on a top subset of these samples based on their softmax confidence, and optimizes the model by PCA.

# Dependencies:
pytorch==1.11.0  
torchvision==0.12.0  
numpy==1.20.3  
sklearn==0.24.2  

# Datasets
Download **Digit-Five** and **PACS** from https://github.com/jindongwang/transferlearning/tree/master/code/DeepDG. Rename them as `dg5` and `PACS` and place them in `./Dataset`.  
Download the subset of **DomainNet** used in our paper from https://drive.google.com/file/d/1LDnU3el-nHoqTgnvxEZP_PxdbdBapNKP/view?usp=sharing, and place it in `./Dataset`.  

# Usage
## Quick Start
After installing all dependency packages, you can use the following command to run the code on PACS
```
python main.py --gpu $gpu_id --order 2 0 1 3 --seed 2022 \
--aug_tau 0.5 --topk_alpha 20 --lr 0.005 --MPCL_alpha 0.5 \
--output result_mainpaper 
```
## Reproduce the experiment results in the paper
Please run all bash files in `scripts` with the following command.
```
cd ./scripts
bash PACS.sh dg5.sh subdomain_net.sh
```

# Performance
The visualization of results will be saved in `result_*` (you can customize the file name) after training. The following table reports the experiment results of running RaTP on PACS.
![Results](./fig/results_PACS.png)  

# Citation
```
@article{liu2023deja,
  title={DEJA VU: Continual Model Generalization For Unseen Domains},
  author={Liu, Chenxi and Wang, Lixu and Lyu, Lingjuan and Sun, Chen and Wang, Xiao and Zhu, Qi},
  journal={arXiv preprint arXiv:2301.10418},
  year={2023}
}
```

## Contact

If you have any questions regarding the code, please feel free to contact Lixu Wang (lixuwang2025@u.northwestern.edu) or Lingjuan Lyu (Lingjuan.Lv@sony.com).

###### Copyright 2023, Sony AI, Sony Corporation of America, All rights reserved.
