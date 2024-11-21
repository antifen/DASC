import argparse
import csv
import json
import time
from torch.utils.data import DataLoader,WeightedRandomSampler
import torch
import os
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from torch import nn
import cupy as cp
from net.Res_18.RN18 import Resnet,dua_network
from pre.dataset import Dataset, Generator
from pre.constants import *
from sklearn.metrics import accuracy_score, f1_score


device = 'cuda' if torch.cuda.is_available() else 'cpu'
set_seeds(42, torch.cuda.is_available())
def get_file_name(dataset, indexes):
    return np.array(dataset.images)[indexes].tolist()

def hard_label_csv(csv_pth):
    csv_pth = os.path.join(csv_pth,'class_all.csv')
    df = pd.read_csv(csv_pth)
    df = df[(df['labels'] == 1)]
    file_list = df['File Name'].tolist()
    file_cluster_dict = dict(zip(df['File Name'], df['cluster']))
    cluster_counts = df['cluster'].value_counts().sort_index().tolist()
    return file_list,file_cluster_dict,cluster_counts

def main(args):

    # Prepare folders
    if not os.path.isdir(PATH_RESULTS + args.experiment_name + '/'):
        os.makedirs(PATH_RESULTS + args.experiment_name + '/')

    # Set train dataset

    hard_name_list,hard_label_dict,weigth_sample = hard_label_csv(args.dir_csv)
    print(len(hard_name_list))
    train_dataset = Dataset(args.dataset_id, partition='train', input_shape=args.input_shape, labels=1,
                            preallocate=args.preallocate, dir_images=args.dir_images,
                            dir_masks=args.dir_masks,select_list=hard_name_list,hard_label=hard_label_dict)
    class_weights = 1. / torch.tensor(weigth_sample, dtype=torch.float)

    print(f"Class Weights: {class_weights}")
    sample_weights = [class_weights[label] for label in train_dataset.Hard_Y]
    sample_weights = torch.tensor(sample_weights)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_generator = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              sampler=sampler,
                              num_workers=16,
                              pin_memory=True)
    if args.model_name == 'res':
        model = Resnet(in_channels=args.input_shape[0], n_classes=3, n_blocks=args.n_blocks, pretrained=args.pretrained,
                       mode=args.mode, aggregation=args.aggregation, backbone=args.backbone).to(device)
    # Set losses
    Lce = torch.nn.CrossEntropyLoss()


    opt = torch.optim.Adam(list(model.parameters()), lr=args.lr)
    history, val_acc_min = [], 0
    N = train_dataset.__len__()

    labels = train_dataset.get_all_labels()
    a = labels
    ###################################################
    sample_stats = {}
    for index in range(len(train_dataset)):
        sample_stats[index] = {
        'correct':0,
        'sample_count':0
        }

    for i_epoch in range(args.epochs):
        index_list = []
        Y_train , Yhat_train = [], []
        loss_ce_over_all, loss_constraint_over_all = 0.0, 0.0
        flags = torch.zeros(N).bool().to(device)
        for i_iteration, (X,_,Hrad_Y,indexes) in enumerate(train_generator):
            model.train()
            index_list.append(indexes.tolist())
            X = torch.tensor(X.clone().detach().numpy()).cuda().float().to(device)
            Hrad_Y = [item for sublist in Hrad_Y for item in sublist]
            Hrad_Y = torch.tensor(Hrad_Y).cuda().float().to(device)

            pred, feature,cam = model(X,reshape_cam=True)

            pred_probs = F.softmax(pred, dim=1)


            _, predicted_classes = torch.max(pred_probs, 1)
            ce = Lce(pred,Hrad_Y.to(torch.long))
            L = ce * args.alpha_ce


            #################################
            # Backward gradients
            L.backward()
            opt.step()
            opt.zero_grad()

            # Track predictions and losses
            Y_train.append(Hrad_Y.detach().cpu().numpy().astype(int))
            Yhat_train.append(predicted_classes.detach().cpu().numpy().astype(int))
            loss_ce_over_all += ce.item()

            correct_predictions = (Hrad_Y.detach().cpu().numpy().astype(int) == predicted_classes.detach().cpu().numpy().astype(int))
            flags[indexes] = torch.tensor(correct_predictions, device=device, dtype=torch.bool)

            # Display losses and acc per iteration
            info = "[INFO] Epoch {}/{}  -- Step {}/{}: Lce={:.4f}".format(
                    i_epoch + 1, args.epochs, i_iteration + 1, len(train_generator), ce.detach().cpu().numpy())
            print(info, end='\r')
        list_again = sum(index_list, [])
        if i_epoch > args.warmup:
            for index1 in list_again:
                if flags[index1]:
                    sample_stats[index1]['correct'] += 1
                sample_stats[index1]['sample_count'] += 1




        loss_training = loss_ce_over_all / len(train_generator)

        train_accuracy = accuracy_score(np.concatenate(Y_train), np.concatenate(Yhat_train, 0))
        train_f1 = f1_score(np.concatenate(Y_train), np.concatenate(Yhat_train, 0), average='weighted')

        # Display losses per epoch
        info = "[INFO] Epoch {}/{}  -- Step {}/{}: Lce={:.4f} accuracy={:.4f} ; f1={:.4f}".format(
            i_epoch + 1, args.epochs, len(train_generator), len(train_generator), loss_training, train_accuracy, train_f1
       )
        print(info, end='\n')


        # Track learning curves
        h = [loss_training, train_accuracy, train_f1]#, metrics_test["accuracy"], metrics_test["f1"] metrics_val["accuracy"],metrics_val["f1"]
        h_caption = ['loss_train', 'metric_train_acc', 'metric_train_f1']

        # Save learning curves
        history.append(h)
        history_final = pd.DataFrame(history, columns=h_caption)
        history_final.to_excel(PATH_RESULTS + args.experiment_name + '/lc.xlsx')

    ##############################################################################
    csv_file_path = 'hard.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['File Name','correct','sample_count','labels'])
        for index, stats in sample_stats.items():
            file_name = get_file_name(train_dataset, [index])[0]
            writer.writerow([file_name,stats['correct'],stats['sample_count'],a[index]])
    # Save last model
    torch.save(model.state_dict(), PATH_RESULTS + args.experiment_name + '/network_weights_last.pth')
    argparse_dict = vars(args)
    with open(PATH_RESULTS + args.experiment_name + '/setting.txt', 'w') as f:
        json.dump(argparse_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories and partition
    parser.add_argument("--dataset_id", default='MIDOG21', type=str)
    parser.add_argument("--dir_images", default='JPEGImages', type=str)
    parser.add_argument("--dir_masks", default='SegmentationClass', type=str)
    parser.add_argument("--dir_csv", default='/root/cls/hard_clus', type=str)
    parser.add_argument("--experiment_name", default="hard_clus", type=str)
    # Hyperparameter
    parser.add_argument("--model_name", default='res', type=str,help='res,AT,gcov_res')
    parser.add_argument("--input_shape", default=(3, 80, 80), type=list)
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--epochs", default=70, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--n_blocks", default=3, type=int)
    parser.add_argument("--pretrained", default=True, type=lambda x: (str(x).lower() == 'true'), help="xxx")
    parser.add_argument("--backbone", default="Res_18", type=str)

    # Weakly supervised setting
    parser.add_argument("--mode", default='instance', type=str)
    parser.add_argument("--aggregation", default='max', type=str)



    # Other settings
    parser.add_argument("--preallocate", default=True, type=lambda x: (str(x).lower() == 'true'), help="xxx")


    args = parser.parse_args()

    main(args)
