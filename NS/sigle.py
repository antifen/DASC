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

sys.path.append(os.getcwd())

from pre.dataset import Dataset, Generator
from pre.constants import *
from utils2.misc import set_seeds, sigmoid
from utils2.evaluation import evaluate_image_level

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'


set_seeds(42, torch.cuda.is_available())


def main(args):
    # Prepare folders
    if not os.path.isdir(PATH_RESULTS + args.experiment_name + '/'):
        os.makedirs(PATH_RESULTS + args.experiment_name + '/')

    # Set train dataset
    train_dataset = Dataset(args.dataset_id, partition='train', input_shape=args.input_shape, labels=1, sigle=True,
                            mask_train=args.location_constraint,
                            preallocate=args.preallocate, dir_images=args.dir_images,
                            dir_masks=args.dir_masks)
    # Set validation and testing datasets
    val_dataset = Dataset(args.dataset_id, partition='test', input_shape=args.input_shape, labels=1,
                          mask_train=args.location_constraint,
                          preallocate=args.preallocate, dir_images=args.dir_images, dir_masks=args.dir_masks)

    # Training dataset distillation - only for student training
    # Prepare data generator
    train_generator = Generator(train_dataset, args.batch_size, shuffle=True, balance=True, sigle=True,
                                mask_train=args.location_constraint)

    # Network architecture
    if args.model_name == 'res':
        model = Resnet(in_channels=args.input_shape[0], n_classes=2, n_blocks=args.n_blocks, pretrained=args.pretrained,
                       mode=args.mode, aggregation=args.aggregation, backbone=args.backbone, SCL=args.SCL).cuda()
    train(model, train_generator, val_dataset, args)

    ###################################################


def train(model, train_generator, val_dataset, args):
    # Set losses
    Lce = torch.nn.CrossEntropyLoss()

    all_loss = []
    opt = torch.optim.Adam(list(model.parameters()), lr=args.lr)
    # Training loop
    history, val_acc_min = [], 0
    for i_epoch in range(args.epochs):
        index_list = []
        Y_train, Yhat_train = [], []
        loss_ce_over_all, loss_constraint_over_all = 0.0, 0.0

        for i_iteration, (X, Y, indexes) in enumerate(train_generator):
            model.train()

            index_list.append(indexes)
            X = torch.tensor(X).cuda().float().to(device)
            Y = [item for sublist in Y for item in sublist]
            Y = torch.tensor(Y).cuda().float().to(device)
            # pred1, pred2, pred, cam1, cam2, cam,features = model(X,S)
            pred, feature, cam = model(X)
            ce = Lce(pred, Y.to(torch.long))
            # gce = lgce(pred.to(device), Y)

            # l2 = compute_l2_reg_loss(model, include_frozen=True, reg_final=True, reg_biases=False)
            L = (ce) * args.alpha_ce
            L.backward()
            opt.step()
            opt.zero_grad()
            # Track predictions and losses
            Y_train.append(Y.detach().cpu().numpy())

            Yhat_train.append(torch.softmax(pred, dim=1).detach().cpu().numpy())
            loss_ce_over_all += (ce).item()
            # Display losses and acc per iteration
            info = "[INFO] Epoch {}/{}  -- Step {}/{}: Lce={:.4f}".format(
                i_epoch + 1, args.epochs, i_iteration + 1, len(train_generator), ce.detach().cpu().numpy())
            print(info, end='\r')

        model.eval()
        # Validation predictions
        Y_val, Yhat_val, Mhat_val = predict_dataset(val_dataset, model, bs=64, stride=args.stride)
        # Train metrics
        metrics_train, _ = evaluate_image_level(np.concatenate(Y_train), np.concatenate(Yhat_train, 0))
        loss_training = loss_ce_over_all / len(train_generator)
        # Validation metrics
        metrics_val, th = evaluate_image_level(Y_val, Yhat_val)
        # Display losses per epoch
        info = "[INFO] Epoch {}/{}  -- Step {}/{}: Lce={:.4f} accuracy={:.4f} ; f1={:.4f} || accuracy_val={:.4f} ; f1={:.4f}".format(
            i_epoch + 1, args.epochs, len(train_generator), len(train_generator), loss_training,
            metrics_train["accuracy"], metrics_train["f1"], metrics_val["accuracy"], metrics_val["f1"])

        print(info, end='\n')

        # Track learning curves
        h = [loss_training, metrics_train["accuracy"], metrics_train["f1"], metrics_val["accuracy"],
             metrics_val["f1"]]  # , metrics_test["accuracy"], metrics_test["f1"]
        h_caption = ['loss_train', 'metric_train_acc', 'metric_train_f1', 'metric_val_acc',
                     'metric_val_f1', ]

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
        if i_epoch % 10 == 0:
            torch.save(model.state_dict(), PATH_RESULTS + args.experiment_name +'/'+ str(i_epoch) + '.pth')
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
    parser.add_argument("--dataset_id", default='', type=str)
    parser.add_argument("--dir_images", default='', type=str)
    parser.add_argument("--dir_masks", default='', type=str)
    parser.add_argument("--experiment_name", default="1_NS", type=str)
    # Hyperparameter

    parser.add_argument("--model_name", default='res', type=str, help='')
    parser.add_argument("--input_shape", default=(3, 80, 80), type=list)
    parser.add_argument("--warmup", default=0, type=int)
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--n_blocks", default=4, type=int)
    parser.add_argument("--pretrained", default=True, type=lambda x: (str(x).lower() == 'true'), help="xxx")
    parser.add_argument("--augmentation", default=True, type=lambda x: (str(x).lower() == 'true'), help="xxx")
    parser.add_argument("--backbone", default="Res_18", type=str, help='RN50,Res_18')


    # Weakly supervised setting
    parser.add_argument("--mode", default='instance', type=str)
    parser.add_argument("--aggregation", default='max', type=str)
    # Other settings
    parser.add_argument("--preallocate", default=True, type=lambda x: (str(x).lower() == 'true'), help="xxx")
    args = parser.parse_args()

    main(args)
