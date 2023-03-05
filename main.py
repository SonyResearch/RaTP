import sys
import copy

from arguments import get_args
from opt import *
from RaTP import RaTP
import modelopera
import Replay.alg as ReplayAlg
from datautil.getdataloader import get_img_dataloader
from utils.util import set_random_seed, save_checkpoint, log_print
from train import train
from utils.visual import save_plot_acc_epochs, fit_tSNE, visual_tSNE

if __name__ == '__main__':
    args = get_args()
    set_random_seed(args.seed)
    log_print('################################################', args.log_file)
    log_print('############### Attention: arguments steps_per_epoch should be changed with batch_size and dataset ! ####################', args.log_file)
    log_print('command args: {}'.format(sys.argv[1:]), args.log_file)
    log_print('arguments: {}\n'.format(args), args.log_file, p=False)

    # Get Data
    train_loaders, eval_loaders, eval_name_dict, task_sequence_name = get_img_dataloader(args)

    # Model
    model = RaTP(args).cuda()
    old_model = None   # used for knwoledge distillation algorithms
    Replay_algorithm_class = ReplayAlg.get_algorithm_class(args.replay)
    Replay_algorithm = Replay_algorithm_class(args)
    model.train()

    # initial statistics metrics
    target_domain_acc_list = []
    source_domain_acc_list = []
    all_val_acc_record = {}  # list of record list for each task. e.g.'task0': [initial acc, [acc along training of task0], [acc along training of task1]...]
    for tid in range(len(eval_name_dict['valid'])):
        all_val_acc_record['task{}'.format(tid)] = [[modelopera.accuracy(model, eval_loaders[eval_name_dict['valid'][tid]])]]
    if args.tsne:
        tSNE_dict = {'features':[], 'clabels':[], 'dlabels':[]}
        tSNE_dict = fit_tSNE(args, model, eval_loaders, tSNE_dict)


    # incremental train different domains
    for task_id, dataloader in enumerate(train_loaders):
        
        # construct replay exemplars
        replay_dataset = Replay_algorithm.update_dataloader()

        # main training
        model, val_acc_record, pseudo_dataloader = train(args, model, old_model, task_id, dataloader, replay_dataset, eval_loaders, eval_name_dict)
        for tid in range(len(eval_name_dict['valid'])):
            all_val_acc_record['task{}'.format(tid)].append(val_acc_record['task{}'.format(tid)])
        
        # show inter result.
        for tid in range(task_id+1):
            log_print('after task {}: {}'.format(tid, [all_val_acc_record['task{}'.format(i)][tid+1][-1] for i in range(len(eval_name_dict['valid']))]), args.log_file)

        # finish task
        Replay_algorithm.update(model, task_id, pseudo_dataloader)
        
        if args.tsne:
            tSNE_dict = fit_tSNE(args, model, eval_loaders, tSNE_dict)

        # save model after finishing a task. It will be used for knowledge distill algorithms
        save_checkpoint(args.saved_model_name, model, args)
        old_model = copy.deepcopy(model)
        model.cuda()
        old_model.cuda().eval()
    
    save_plot_acc_epochs(args, all_val_acc_record, task_sequence_name)
    if args.tsne:
        visual_tSNE(args, tSNE_dict)

    log_print('\nDGaccuracy matrix: ', args.log_file)
    log_print('at start: {}'.format([all_val_acc_record['task{}'.format(tid)][0][0] for tid in range(len(eval_name_dict['valid']))]), args.log_file)
    for tid in range(len(eval_name_dict['valid'])):
        log_print('after task {}: {}'.format(tid, [all_val_acc_record['task{}'.format(i)][tid+1][-1] for i in range(len(eval_name_dict['valid']))]), args.log_file)

    log_print('', args.log_file)




