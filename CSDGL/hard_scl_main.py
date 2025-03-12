import argparse
import csv
import json
import sys
import time

import torch
import os
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from collections import Counter

sys.path.append(os.getcwd())
from SCLR.loss import SupConLoss_rank, SupConLoss,my_loss,LabelSmoothingCrossEntropy
from SCLR.utils import *
from net.Res_18.RN18 import Resnet, dua_network
from pre.dataset import Dataset, Generator
from pre.constants import *
from SCLR.loss import LDAMLoss
from net.Res_18.utils import distill_dataset, predict_dataset
from utils2.misc import set_seeds, sigmoid
from utils2.evaluation import evaluate_image_level, evaluate_motisis_localization
from utils2.loss2 import compute_l2_reg_loss, GCELoss
from net.res38_cls.res38 import Net
from train_utils import train_sclb,warm_train,train_hard_sclb



device = 'cuda' if torch.cuda.is_available() else 'cpu'

def hard_label_csv(csv_pth):
    csv_pth = os.path.join(csv_pth,'SCL.csv')
    df = pd.read_csv(csv_pth)
    file_list = df['File Name'].tolist()
    file_cluster_dict = dict(zip(df['File Name'], df['hard_labels']))
    cluster_counts = df['hard_labels'].value_counts().sort_index().tolist()
    return file_list,file_cluster_dict,cluster_counts
set_seeds(42, torch.cuda.is_available())


def main(args):
    # Prepare folders
    if not os.path.isdir(PATH_RESULTS + args.experiment_name + '/'):
        os.makedirs(PATH_RESULTS + args.experiment_name + '/')

    hard_name_list, hard_label_dict, _ = hard_label_csv(args.dir_csv)
    # hard_name_list = filter_and_sample_images(hard_name_list)
    train_dataset = Dataset(args.dataset_id, partition='train', input_shape=args.input_shape, labels=1,
                            preallocate=args.preallocate, dir_images=args.dir_images,
                            dir_masks=args.dir_masks,select_list=hard_name_list,mask_train=True,hard_label=hard_label_dict,scl=False,linear=False,scl_hard=True)


    val_dataset = Dataset(args.dataset_id, partition='val', input_shape=args.input_shape, labels=1,
                          mask_train=args.location_constraint,preallocate=args.preallocate, dir_images='JPEGImages', dir_masks=args.dir_masks)


    train_generator = Generator(train_dataset, args.batch_size, shuffle=True, balance=True, sigle=False,
                                mask_train=False,scl_hard=True)

    train_loader_cluster = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,num_workers=16, pin_memory=True )
    # Network architecture
    if args.model_name == 'res':
        model = Resnet(in_channels=args.input_shape[0], n_classes=2, n_blocks=args.n_blocks, pretrained=args.pretrained,
                       mode=args.mode, aggregation=args.aggregation, backbone=args.backbone, SCL=args.SCL).cuda()
    # Set losses
    Lce = torch.nn.CrossEntropyLoss()
    lgce = GCELoss(num_classes=2, gpu=0)
    lsmooth = LabelSmoothingCrossEntropy()
    #cluster par
    cluster_number = [4, 4]

    all_labels = train_dataset.get_scl_lable_labels()
    label_counts = Counter(all_labels)
    all_unique_labels = sorted(label_counts.keys())
    cls_num_list = [label_counts[label] for label in all_unique_labels]
    print('cls num list:')
    print(cls_num_list)
    args.cls_num_list = cls_num_list
    print(sum(cls_num_list))

    opt = torch.optim.Adam(list(model.parameters()), lr=args.lr)
    # Training loop
    history, val_acc_min = [], 0
    for i_epoch in range(args.epochs):
        loss_ce_over_all, loss_constraint_over_all = 0.0, 0.0
        i = 0

        if args.warmup > i_epoch:
            criterion_sup = SupConLoss(temperature=args.temperature).to(device)
            warm_train(model,opt,train_generator,criterion_sup,args,device,i_epoch)

        else:
            if args.hard == False :
                if i_epoch % args.step == 0:
                    targets,density = cluster(train_loader_cluster, model, cluster_number, args)
                    train_dataset.new_labels = targets
                criterion_sup = SupConLoss_rank(num_class=args.num_classes,temperature=args.temperature,ranking_temperature=density).to(device)
                Y_train, Yhat_train, loss_ce_over_all, loss_constraint_over_all, loss_sup_all  = train_sclb(model,opt,train_generator,Lce,criterion_sup,args,device,i_epoch)
            elif args.hard:
                if i_epoch % args.step == 0:
                    targets,density = cluster(train_loader_cluster, model, cluster_number, args)
                    train_dataset.new_labels = targets
                criterion_sup = my_loss(num_class=args.num_classes,temperature=args.temperature,ranking_temperature=density).to(device)
                Y_train,Yhat_train,loss_ce_over_all,loss_constraint_over_all,loss_sup_all = train_hard_sclb(model,opt,train_generator,Lce,lgce,lsmooth,None,criterion_sup,args,device,i_epoch)

            model.eval()
            # Validation predictions
            Y_val, Yhat_val, Mhat_val = predict_dataset(val_dataset, model, bs=64, stride=args.stride)
            # Train metrics
            metrics_train, _ = evaluate_image_level(np.concatenate(Y_train), np.concatenate(Yhat_train, 0))
            loss_training = loss_ce_over_all / len(train_generator)
            loss_sup_traing = loss_sup_all / len(train_generator)
            # Validation metrics
            metrics_val, th = evaluate_image_level(Y_val, Yhat_val)
            # Display losses per epoch
            info = "[INFO] Epoch {}/{}  -- Step {}/{}: Lce={:.4f} Lsup={:.4f} accuracy={:.4f} ; f1={:.4f} || accuracy_val={:.4f} ; f1={:.4f}".format(
                i_epoch + 1, args.epochs, len(train_generator), len(train_generator), loss_training,loss_sup_traing,
                metrics_train["accuracy"], metrics_train["f1"], metrics_val["accuracy"], metrics_val["f1"])
            if args.location_constraint:
                loss_constraint_training = loss_constraint_over_all / len(train_generator)
                info += " || Lcons={:.4f}".format(loss_constraint_training)
            print(info, end='\n')

            # Track learning curves
            h = [loss_training,loss_sup_traing,metrics_train["accuracy"], metrics_train["f1"], metrics_val["accuracy"],
                 metrics_val["f1"]]  # , metrics_test["accuracy"], metrics_test["f1"]
            h_caption = ['loss_train', 'los_sup','metric_train_acc', 'metric_train_f1', 'metric_val_acc',
                         'metric_val_f1', ]
            if args.location_constraint:
                h.append(loss_constraint_training)
                h_caption.append('loss_centroid')

            # Save learning curves
            history.append(h)
            history_final = pd.DataFrame(history, columns=h_caption)
            history_final.to_excel(PATH_RESULTS + args.experiment_name + '/lc.xlsx')

            # Save model
            if metrics_val["f1"] > val_acc_min:
                print('Validation F1 improved from ' + str(round(val_acc_min, 5)) + ' to ' + str(
                    round(metrics_val["f1"], 5)) + '  ... saving model')
                torch.save(model.state_dict(), PATH_RESULTS + args.experiment_name + '/network_weights_best.pth')
                val_acc_min = metrics_val["f1"]
            print('Validation cm: ', end='\n')
            print(metrics_val["cm"], end='\n')
    ##############################################################################

    # Save last model
    torch.save(model.state_dict(), PATH_RESULTS + args.experiment_name + '/network_weights_last.pth')

    # Final localization results using the best model
    model.load_state_dict(torch.load(PATH_RESULTS + args.experiment_name + '/network_weights_best.pth'))

    print('Predicting to validate', end='\n')
    # Validation predictions
    Y_val, Yhat_val, Mhat_val = predict_dataset(val_dataset, model, bs=64)
    # Validation metrics
    metrics_val, th_val = evaluate_image_level(Y_val, sigmoid(Yhat_val))

    # Save input args
    args.threshold_val = float(round(th_val, 2))
    argparse_dict = vars(args)
    with open(PATH_RESULTS + args.experiment_name + '/setting.txt', 'w') as f:
        json.dump(argparse_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories and partition
    parser.add_argument("--dataset_id", default='MIDOG21', type=str)
    parser.add_argument("--dir_images", default='JPEGImages_filter', type=str)
    parser.add_argument("--dir_masks", default='SegmentationClass', type=str)

    parser.add_argument("--data_type", default="2021_hard_abil", type=str)

    parser.add_argument("--experiment_name", default="last", type=str)

    parser.add_argument("--stride", default=1, type=int, help='')
    # Hyperparameter

    parser.add_argument("--model_name", default='res', type=str, help='res')
    parser.add_argument("--input_shape", default=(3, 224, 224), type=list)
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--epochs", default=99, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--n_blocks", default=3, type=int)
    parser.add_argument("--pretrained", default=True, type=lambda x: (str(x).lower() == 'true'), help="xxx")
    parser.add_argument("--augmentation", default=True, type=lambda x: (str(x).lower() == 'true'), help="xxx")
    parser.add_argument("--backbone", default="Res_18", type=str, help='RN50,Res_18,res38')
    ###################SCL
    parser.add_argument("--dir_csv", default='/root/cls/local_data/results/2021_hard_abil/2nd_fp2_nopenaty', type=str, help='')
    parser.add_argument("--SCL", default=True, type=bool, help='')
    parser.add_argument("--train_rule", default='Rank', type=str, help='')
    parser.add_argument("--hard", default=True, type=bool, help='')
    parser.add_argument('--temperature', default=0.1, type=float, help='softmax temperature')
    parser.add_argument('--alpha_sup', default=0.1, type=float, help='')
    parser.add_argument('--feat_dim', default=512, type=int, help='feature dimenmion for model')
    parser.add_argument('--num_classes', default=2, type=int, help='')  #
    parser.add_argument('--step', default=10, type=int, help='K epoch updata cluster')
    parser.add_argument('--cluster_method', default=False, type=str, help='chose to balance cluster')
    # Weakly supervised setting
    parser.add_argument("--mode", default='instance', type=str)
    parser.add_argument("--aggregation", default='max', type=str)

    # Constraints
    parser.add_argument("--alpha_ce", default=1, type=float)
    parser.add_argument("--location_constraint", default=True, type=lambda x: (str(x).lower() == 'true'), help="xxx")
    parser.add_argument("--alpha_location", default=0.1, type=float)
    parser.add_argument("--temperature_loc", default=10, type=float)
    parser.add_argument("--margin", default=10, type=float)
    parser.add_argument("--tlb", default=50, type=float)
    parser.add_argument("--constraint_type", default='l2', type=str, help="Options: l2 / lp ")

    # Other settings
    parser.add_argument("--preallocate", default=True, type=lambda x: (str(x).lower() == 'true'), help="xxx")
    parser.add_argument("--save_visualization", default=False, type=lambda x: (str(x).lower() == 'true'), help="xxx")

    args = parser.parse_args()
    args.experiment_name = args.data_type + '/' + args.experiment_name
    main(args)
