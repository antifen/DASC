import argparse
import csv
import json
import time

import torch
import os
import torch.nn.functional as F
import pandas as pd
import numpy as np

from torch import nn

from net.Res_18.RN18 import Resnet,dua_network
from pre.dataset import Dataset, Generator
from pre.constants import *

from net.Res_18.utils import distill_dataset, predict_dataset, constraint_localization
from utils2.misc import set_seeds, sigmoid
from utils2.evaluation import evaluate_image_level, evaluate_motisis_localization
from tool.newlosses import GeneralizedCrossEntropy,NFLandNCE
from tqdm import tqdm
# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# set_seeds(42, torch.cuda.is_available())



def main(args):

    # Prepare folders
    if not os.path.isdir(PATH_RESULTS + args.experiment_name + '/'):
        os.makedirs(PATH_RESULTS + args.experiment_name + '/')

    # Set train dataset
    train_dataset = Dataset(args.dataset_id, partition='train', input_shape=args.input_shape, labels=1,
                            preallocate=args.preallocate, dir_images=args.dir_images,
                            dir_masks=args.dir_masks)
    # Set validation and testing datasets
    val_dataset = Dataset(args.dataset_id, partition='test', input_shape=args.input_shape, labels=1,
                          preallocate=args.preallocate, dir_images=args.dir_images, dir_masks=args.dir_masks)


    # Training dataset distillation - only for student training
    # Prepare data generator
    train_generator = Generator(train_dataset, args.batch_size,shuffle=True, balance=True)

    # Network architecture
    if args.model_name == 'res':
        weak_model = Resnet(in_channels=args.input_shape[0], n_classes=2, n_blocks=args.n_blocks, pretrained=args.pretrained,
                       mode=args.mode, aggregation=args.aggregation, backbone=args.backbone)
        strong_mdoel = Resnet(in_channels=args.input_shape[0], n_classes=2, n_blocks=args.n_blocks,
                            pretrained=args.pretrained,
                            mode=args.mode, aggregation=args.aggregation, backbone=args.backbone)
        model = dua_network(weak_model,strong_mdoel,n_classes=2,n_blocks=args.n_blocks, pretrained=args.pretrained,
                       mode=args.mode, aggregation=args.aggregation, backbone=args.backbone).to(device)

    # Set losses
    Lce = torch.nn.CrossEntropyLoss()


    # Lgce = GeneralizedCrossEntropy(num_classes=2).to(device)
    # Set optimizer
    opt = torch.optim.Adam(list(model.parameters()), lr=args.lr)
    # Training loop
    history, val_acc_min = [], 0
    N = train_dataset.__len__()

    s_prev_confidence = torch.ones(N).to(device) * 1 / N
    w_prev_confidence = torch.ones(N).to(device) * 1 / N
    ws_prev_confidence = torch.ones(N).to(device) * 1 / N
    scd_prev_confidence = torch.full((N,), 1-1 / N).to(device)
    scd = torch.ones(N).to(device)
    w_probs = torch.zeros(N, 2).to(device)
    s_probs = torch.zeros(N, 2).to(device)
    ws_probs = torch.zeros(N, 2).to(device)
    labels = train_dataset.get_all_labels()
    selected_flags = torch.zeros(N).bool().to(device)
    a = labels
    labels = torch.tensor(labels).to(device)

    ###################################################
    sample_stats = {}
    for index in range(len(train_dataset)):
        sample_stats[index] = {
            'clean_flags': 0,
            'hard_flags': 0,
            'correction_flags': 0,
            'learning':0,
            'forget':0,
            'scd':0,
            'sample_count':0
        }
    # we are saving for each sample the correct class, loss, probability of prediction, class prediction
    dict_infos = {}

    for i_epoch in range(args.epochs):
        index_list = []
        Y_train , Yhat_train = [], []
        loss_ce_over_all = 0.0, 0.0
        i = 0

        for i_iteration, (X,S,Y,indexes) in enumerate(train_generator):
            model.train()
            index_list.append(indexes)
            X = torch.tensor(X).cuda().float().to(device)
            S = torch.tensor(S).cuda().float().to(device)
            Y = [item for sublist in Y for item in sublist]
            Y = torch.tensor(Y).cuda().float().to(device)
            # Forward net
            pred1, pred2, pred, cam1, cam2, cam,features = model(X,S, reshape_cam=True)
            if i_epoch > args.warmup:
                features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
                features = torch.flatten(features, 1)
                for index,feature,tar in zip(indexes, features,Y):
                    feature_vector = (feature > 0).cpu().numpy()
                    dict_infos[i] = {
                        'features': feature_vector,
                        'label': tar.item(),
                        'index': index
                                       }
                    i = i + 1

            # pred_logits, cam_logits = model(X,S, reshape_cam=True)

            ce1 = Lce(pred1, Y.to(torch.long))
            ce2 = Lce(pred2, Y.to(torch.long))
            ce = Lce(pred,Y.to(torch.long))
            L = (0.25 * ce1 + 0.25 * ce2 + 0.5 * ce) * args.alpha_ce


            with torch.no_grad():
                w_prob = F.softmax(pred1, dim=1)
                w_probs[indexes] = w_prob
                s_prob = F.softmax(pred2, dim=1)
                s_probs[indexes] = s_prob
                ws_prob = F.softmax(pred, dim=1)
                ws_probs[indexes] = ws_prob


            #################################
            # Backward gradients
            L.backward()
            opt.step()
            opt.zero_grad()

            # Track predictions and losses
            Y_train.append(Y.detach().cpu().numpy())

            Yhat_train.append(torch.softmax(pred,dim=1).detach().cpu().numpy())
            loss_ce_over_all += (0.25 * ce1 + 0.25 * ce2 + 0.5 * ce).item()

            # Display losses and acc per iteration
            info = "[INFO] Epoch {}/{}  -- Step {}/{}: Lce={:.4f}".format(
                    i_epoch + 1, args.epochs, i_iteration + 1, len(train_generator), ce.detach().cpu().numpy())
            print(info, end='\r')


        if i_epoch > args.warmup:
            df_infos = pd.DataFrame.from_dict(dict_infos, "index")
            df_infos['features'] = df_infos['features'].apply(lambda x: x.astype(float))

            features_np = np.array(df_infos['features'].tolist(), dtype=np.float32)

            features = torch.tensor(features_np).cuda()

            df_infos['label'] = df_infos['label'].astype(int)
            labels1 = torch.tensor(df_infos['label'].tolist(), dtype=torch.long).cuda()

            unique_labels = labels1.unique()
            mean_features = torch.zeros_like(features)

            for label in unique_labels:
                label_mask = labels1 == label
                label_features = features[label_mask]
                mean_label_features = label_features.mean(dim=0)
                mean_features[label_mask] = mean_label_features

            df_infos['mean-features'] = mean_features.cpu().tolist()


            df_infos['SCD'] = df_infos.apply(lambda row: distance_to_center_features(row), axis=1)



            scd_normalized = (df_infos['SCD'] - df_infos['SCD'].min()) / (df_infos['SCD'].max() - df_infos['SCD'].min())
            df_infos['SCD'] = scd_normalized
            for index, row in df_infos.iterrows():
                scd_index = row['index']
                scd_value = row['SCD']
                scd[scd_index] = scd_value

        with torch.no_grad():
            w_prob_max, w_label = torch.max(w_probs, dim=1)
            s_prob_max, s_label = torch.max(s_probs, dim=1)
            ws_prob_max, ws_label = torch.max(ws_probs, dim=1)

        ###############Selection###############
        list1 = list(set(sum(index_list, [])))

        w_mask = w_probs[labels >= 0, labels] > w_prev_confidence[labels >= 0]
        s_mask = s_probs[labels >= 0, labels] > s_prev_confidence[labels >= 0]

        clean_flags = w_mask & s_mask
        selected_flags = w_mask + s_mask
        w_selected_flags = w_mask & (~clean_flags)  # H_w
        s_selected_flags = s_mask & (~clean_flags)  # H_s
        hard_flags = w_selected_flags + s_selected_flags  # H
        noise_flags =(~selected_flags) #N
        ##############################################
        momentum = 0.99
        w_prev_confidence[list1] = momentum * w_prev_confidence[list1] + (1 -momentum) * w_prob_max[list1]

        s_prev_confidence[list1] = momentum * s_prev_confidence[list1] + (1 - momentum) * s_prob_max[list1]

        ws_prev_confidence[list1] = momentum * ws_prev_confidence[list1] + (1 - momentum) * ws_prob_max[list1]

        scd_prev_confidence[list1] = momentum * scd_prev_confidence[list1] - (1 - momentum) * scd[list1]

        count(sample_stats,w_prev_confidence,s_prev_confidence,ws_prev_confidence,scd_prev_confidence,hard_flags,noise_flags)


        # --- Validation at epoch end
        model.eval()

        # Validation predictions
        Y_val, Yhat_val, Mhat_val = predict_dataset(val_dataset, model, bs=64,stride=2)

        # Train metrics
        metrics_train, _ = evaluate_image_level(np.concatenate(Y_train), np.concatenate(Yhat_train, 0))
        loss_training = loss_ce_over_all / len(train_generator)
        # Validation metrics
        metrics_val, th = evaluate_image_level(Y_val, Yhat_val)


        # Display losses per epoch
        info = "[INFO] Epoch {}/{}  -- Step {}/{}: Lce={:.4f} accuracy={:.4f} ; f1={:.4f} || accuracy_val={:.4f} ; f1={:.4f}".format(
            i_epoch + 1, args.epochs, len(train_generator), len(train_generator), loss_training, metrics_train["accuracy"], metrics_train["f1"]
            , metrics_val["accuracy"], metrics_val["f1"])
        if args.location_constraint:
            loss_constraint_training = loss_constraint_over_all / len(train_generator)
            info += " || Lcons={:.4f}".format(loss_constraint_training)
        print(info, end='\n')

        # Track learning curves
        h = [loss_training, metrics_train["accuracy"], metrics_train["f1"], metrics_val["accuracy"],
             metrics_val["f1"]]#, metrics_test["accuracy"], metrics_test["f1"]
        h_caption = ['loss_train', 'metric_train_acc', 'metric_train_f1', 'metric_val_acc',
                     'metric_val_f1',]

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
    # Save last model
    torch.save(model.state_dict(), PATH_RESULTS + args.experiment_name + '/network_weights_last.pth')

    with open(PATH_RESULTS + args.experiment_name + '/setting.txt', 'w') as f:
        json.dump(argparse_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories and partition
    parser.add_argument("--dataset_id", default='MIDOG21', type=str)
    parser.add_argument("--dir_images", default='JPEGImages', type=str)
    parser.add_argument("--dir_masks", default='SegmentationClass', type=str)
    parser.add_argument("--experiment_name", default="2nd_fp2", type=str)

    # Hyperparameter
    parser.add_argument("--model_name", default='res', type=str,help='res,AT,gcov_res')
    parser.add_argument("--input_shape", default=(3, 80, 80), type=list)
    parser.add_argument("--warmup", default=5, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--n_blocks", default=4, type=int)
    parser.add_argument("--pretrained", default=True, type=lambda x: (str(x).lower() == 'true'), help="xxx")
    parser.add_argument("--backbone", default="Res_18", type=str)

    # Weakly supervised setting
    parser.add_argument("--mode", default='instance', type=str)
    parser.add_argument("--aggregation", default='max', type=str)

    # Other settings
    parser.add_argument("--preallocate", default=True, type=lambda x: (str(x).lower() == 'true'), help="xxx")

    args = parser.parse_args()

    main(args)
