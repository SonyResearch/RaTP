from cProfile import label
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.manifold import TSNE
import torch
import seaborn as sns
import pandas as pd
from statistics import mean

from Replay.utils import concat_list

def save_plot_acc_epochs(args, all_val_acc_record, task_sequence_name):
    '''
    all_val_acc_record: dict 
    e.g. 'task0': [initial acc, [acc along training of task0], [acc along training of task1]...]
    task_sequence_name: list 
    domain sequence name
    '''

    # save result dictionary
    with open(os.path.join(args.result_dir, 'order{}_lr{}_seed{}.pkl'.format(
        ''.join(str(i) for i in args.order), args.lr, args.seed)), 'wb') as f:
        pickle.dump(all_val_acc_record, f)

    # plot result
    num_task = len(all_val_acc_record)
    plt.clf() 
    fig, ax = plt.subplots() #figsize=(5,4)
    x = range(len(concat_list(all_val_acc_record['task0'])))

    for tid in range(num_task):
        acc_end = [a[-1] for a in all_val_acc_record['task{}'.format(tid)]]
        acc_end.pop(0)
        
        if tid == 0:
            label_name = '{}_{}_{}_fa{}'.format(task_sequence_name[tid], [round(100*elem,1) for elem in acc_end], round(100*mean([elem for elem in acc_end]),1),
                                                 round(100*mean([elem for elem in acc_end[1:]]),1))
        elif tid == (num_task-1):
            label_name = '{}_{}_{}_dg{}'.format(task_sequence_name[tid], [round(100*elem,1) for elem in acc_end], round(100*mean([elem for elem in acc_end]),1),
                                                 round(100*mean([elem for elem in acc_end[:-1]]),1))
        else:
            label_name = '{}_{}_{}_g{}_f{}'.format(task_sequence_name[tid], [round(100*elem,1) for elem in acc_end], round(100*mean([elem for elem in acc_end]),1),
                                                 round(100*mean([elem for elem in acc_end[:tid]]),1), round(100*mean([elem for elem in acc_end[tid+1:]]),1))
            
        ax.plot(x, concat_list(all_val_acc_record['task{}'.format(tid)]), label=label_name)
    ax.set_ylabel('accuracy')
    ax.legend()

    # add grid at the begining of tasks
    ax.set_xticks([(len(all_val_acc_record['task0'][i])*i) for i in range(1, num_task)], minor=False)
    ax.xaxis.grid(True, which='major')
    
    # calculate metrics
    da_acc, dg_acc, forget_acc = calculate_metrics(all_val_acc_record)

    ax.set_title('{}_{}_da{}_dg{}_fg{}'.format(
        args.dataset, args.seed, da_acc, dg_acc, forget_acc))
    
    plt.savefig(os.path.join(args.result_dir, 'order{}_lr{}_seed{}.jpg'.format(
        ''.join(str(i) for i in args.order), args.lr, args.seed)))


def calculate_metrics(all_val_acc_record):   # for ablation study. First average in each domain, then average all domain.
    num_task = len(all_val_acc_record)
    da, dg_av, fa_av = [], [], []
    for tid in range(num_task):
        dg, forget = [], []
        acc_end = [a[-1] for a in all_val_acc_record['task{}'.format(tid)]]
        acc_end.pop(0)    # (num_task, num_task)
        da.append(acc_end[tid])
        for i in range(0, tid):
            dg.append(acc_end[i])
        if len(dg) > 0:
            dg_av.append(mean(dg))
        for i in range(tid+1, num_task):
            forget.append(acc_end[i])
        if len(forget) > 0:
            fa_av.append(mean(forget))
    return round(100*mean(da),1), round(100*mean(dg_av),1), round(100*mean(fa_av),1)

def fit_tSNE(args, net, eval_loaders, tSNE_dict):
    '''
    fit a tSNE using eval data from all domain
    netF: feature extractor
    return:
    tsne_results: 2-D array
    clabels and dlabels: 1-D array
    '''
    # get embedding features using model feature extractor
    features = []
    clabels, dlabels = [], []
    net.eval()
    with torch.no_grad():
        for i in range(args.num_task):
            loader = eval_loaders[args.eval_name_dict['valid'][i]]
            for data in loader:                           # this line will change the performance!! ??
                x = data[0].cuda().float()
                clabel = data[1]
                dlabel = data[2]
                feature = net.featurizer(x)
                features.append(feature.tolist())
                clabels.append(clabel.tolist())
                dlabels.append(dlabel.tolist())
        features = concat_list(features)
        clabels = concat_list(clabels)
        dlabels = concat_list(dlabels)

        tsne = TSNE(n_components=2) #, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(features)
        tSNE_dict['features'].append(tsne_results)
        tSNE_dict['clabels'].append(clabels)
        tSNE_dict['dlabels'].append(dlabels)
    net.train()
    # visual_tSNE(args, tsne_results, clabels, dlabels, task_id)
    return tSNE_dict

def visual_tSNE(args, tSNE_dict):
    plt.clf() 
    fig, axes = plt.subplots(args.num_task+1, 2+len(args.domains), figsize=(5*(2+len(args.domains)), 5*(args.num_task+1)))
    for i in range(args.num_task+1):
        df = pd.DataFrame()
        df['x'] = tSNE_dict['features'][i][:,0]
        df['y'] = tSNE_dict['features'][i][:,1]
        df['class'] = tSNE_dict['clabels'][i]
        df['domain'] = tSNE_dict['dlabels'][i]

        sns.scatterplot(ax = axes[i][0],
            x="x", y="y",
            hue=df.domain,
            palette=sns.color_palette("hls", len(args.domains)),
            data=df,
            legend="full",
            alpha=0.3
        )
        sns.scatterplot(ax = axes[i][1],
            x="x", y="y",
            hue=df['class'],
            palette=sns.color_palette("hls", args.num_classes),
            data=df,
            legend="full",
            alpha=0.3
        )

        for j in range(len(args.domains)):
            df['xd'] = df['x'][df['domain']==j]
            df['yd'] = df['y'][df['domain']==j]
            df['classd'] = df['class'][df['domain']==j]

            sns.scatterplot(ax = axes[i][2+j],
            x="xd", y="yd",
            hue=df['classd'],
            palette=sns.color_palette("hls", args.num_classes),
            data=df,
            legend="full",
            alpha=0.3
            )
            axes[i][2+j].set_title('domain{}'.format(j))

    plt.savefig(os.path.join(args.tSNE_dir, 'order{}.jpg'.format(''.join(str(i) for i in args.order)))) 
    # save result dictionary
    with open(os.path.join(args.tSNE_dir, 'order{}_lr{}_seed{}.pkl'.format(
        ''.join(str(i) for i in args.order), args.lr, args.seed)), 'wb') as f:
        pickle.dump(tSNE_dict, f)
























    pass