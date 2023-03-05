import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description='DG')
    # Data
    parser.add_argument('--data_dir', type=str, default='./Dataset', help='root data dir')
    parser.add_argument('--dataset', type=str, default='PACS', choices=['PACS', 'subdomain_net', 'dg5'])
    parser.add_argument('--order', type=int, nargs='+', help='training domain order')
    parser.add_argument('--test_envs', type=int, nargs='+',
                        default=[], help='no fixed target domains')
    parser.add_argument('--split_style', type=str, default='strat',help="the style to split the train and eval datasets")

    #training algorithm
    parser.add_argument('--loss_alpha1', type=float, default=1.0, help='loss weight')
    parser.add_argument('--PCL_scale', default=12, type=float, help='scale of cross entropy in PCL')
    parser.add_argument('--pLabelAlg', type=str, default="T2PL", choices=['T2PL', 'ground'], help='pesudo label assigning algorithm in target domain. ground is ground true label')
    parser.add_argument('--pseudo_fre', default=1, type=int, help='assign new pseudo label each pseduo_fre epoch')
    parser.add_argument('--replay', type=str, default='icarl', choices=['icarl', 'Finetune'], help='data replay algorithm')
    parser.add_argument('--replay_mode', type=str, default='class', choices=['class', 'domain'])
    parser.add_argument('--memory_size', type=int, help="replay exemplar size")
    parser.add_argument('--aug_tau', type=float, default=0.8, help='do augmentation whose pseudo label confidence larger than this value ')
    parser.add_argument('--distance', type=str, default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--distill', type=str, default='KL', choices=['CE', 'KL', 'feaKL'])   
    parser.add_argument('--distill_alpha', type=float, default=0.5)
    parser.add_argument('--topk_alpha', default=20, type=int, help='k nears in knn pseudo labeling.')
    parser.add_argument('--topk_beta', default=2, type=int, help='topk fitting samples in knn pseudo labeling.')
    parser.add_argument('--MPCL_alpha', type=float, default=0.5, help='MPCL weight')

    # Utils
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--output', type=str,
                        default="result_develop", help='result output path')
    parser.add_argument('--log_file', type=str, help="logging file name under output dir")
    parser.add_argument('--tsne', action='store_true', help='visualize embedding space using tSNE')

    # Model
    parser.add_argument('--net', type=str, default='resnet50',
                        help="featurizer: vgg16, resnet50, resnet101,DTNBase")
    parser.add_argument('--classifier', type=str,
                        default="linear", choices=["linear", "wn"])

    # Training
    parser.add_argument('--lr', type=float, default=5e-3, help="learning rate")
    parser.add_argument('--lr_decay1', type=float, default=1.0, help='feature extractor lr scheduler')
    parser.add_argument('--max_epoch', type=int,
                        default=30, help="max epoch")
    parser.add_argument('--steps_per_epoch', type=int, help='training steps in each epoch. totaly trained sampels in each epoch is steps_per_epoch*batch_size')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='batch_size')
    parser.add_argument('--gpu', type=int, default=0, help="device id to run")
    parser.add_argument('--N_WORKERS', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='for optimizer')

    # Don't need to change
    parser.add_argument('--data_file', type=str, default='',
                        help='root_dir')
    parser.add_argument('--task', type=str, default="img_dg",
                        choices=["img_dg"], help='now only support image tasks')
    
    args = parser.parse_args()

    # I/O
    args.data_dir = os.path.join(args.data_dir, args.dataset, '')
    args.result_dir = os.path.join(args.output, args.dataset)
    args.tSNE_dir = os.path.join(args.result_dir, 'tSNE')
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.tSNE_dir, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args = img_param_init(args)
    args = set_default_args(args)
    args.num_task = len(args.domains) - len(args.test_envs)

    args.saved_model_name = os.path.join(args.result_dir, 'source{}.pt'.format(args.order[0]))  

    return args

def set_default_args(args):
    args.order = [i for i in range(len(args.domains)-len(args.test_envs))] if args.order is None else args.order
    args.log_file = os.path.join(args.result_dir, 'order{}.log'.format(''.join(str(i) for i in args.order))) if args.log_file is None else os.path.join(args.result_dir, args.log_file)
    if args.replay == 'icarl':
        args.replay = 'iCaRL'
    
    memory_size = {'PACS':200, 'subdomain_net':200, 'dg5':200}
    steps_per_epoch = {'PACS':50, 'subdomain_net':70, 'dg5':800}
    args.memory_size = memory_size[args.dataset] if args.memory_size is None else args.memory_size
    args.steps_per_epoch = steps_per_epoch[args.dataset] if args.steps_per_epoch is None else args.steps_per_epoch
    
    return args

def img_param_init(args):
    dataset = args.dataset
    if dataset == 'PACS':
        domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    elif dataset == 'subdomain_net': 
        domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    elif dataset == 'dg5':
        domains = ['mnist', 'mnist_m', 'svhn', 'syn', 'usps']
    else:
        print('No such dataset exists!')
    args.domains = domains
    args.img_dataset = {
        'PACS': ['art_painting', 'cartoon', 'photo', 'sketch'],
        'subdomain_net': ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
        'dg5': ['mnist', 'mnist_m', 'svhn', 'syn', 'usps'],
    }
    if dataset == 'dg5':
        args.input_shape = (3, 32, 32)
        args.num_classes = 10
    else:
        args.input_shape = (3, 224, 224)
        if args.dataset == 'PACS':
            args.num_classes = 7
        elif args.dataset == 'subdomain_net':
            args.num_classes = 10

    args.proj_dim = {'dg5':128, 'PACS':256, 'subdomain_net':512}   # project dim for contrastive loss.

    return args