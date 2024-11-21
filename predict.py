import json
import os
import argparse
import sys

import imutils
from skimage import morphology, measure, io

sys.path.append(os.getcwd())
from torch.backends import cudnn
from tqdm import tqdm
cudnn.enabled = True
import numpy as np
import torch
from torch.backends import cudnn

cudnn.enabled = True
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

import os.path
import cv2
from net.Res_18.RN18 import Resnet,dua_network


def distance_mitosis(ref, pred):

    d = np.sqrt(np.sum(np.square(ref-pred)))

    return d

def evaluate_motisis_localization(X, M, Mhat, dir_out="", filename = '',th=0.5, save_visualization=False, maximum_d=30.,
                                  visualize_negatives=False):
    # Prepare folders
    if save_visualization:
        if not os.path.isdir(dir_out):
            os.makedirs(dir_out)
        if not os.path.isdir(dir_out + 'positives/'):
            os.mkdir(dir_out + 'positives/')
        # if not os.path.isdir(dir_out + 'negatives/'):
        #     os.mkdir(dir_out + 'negatives/')

    TP, FP, FN = 0., 0., 0.
    for i_case in np.arange(0, X.shape[0]):
        print(str(i_case + 1) + '/' + str(X.shape[0]), end='\r')

        TP_case = []
        FP_case = []
        FN_case = []

        x = np.transpose(X[i_case, :, :, :], (1, 2, 0))

        img = Image.fromarray((x * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)

        m = M[i_case, :, :]
        mhat = Mhat[i_case, :, :] > th

        # labelled mitosis
        labels_ref = measure.label(m)
        props_ref = measure.regionprops(labels_ref)

        # Predicted mitosis
        labels_pred = measure.label(mhat)
        props_pred = measure.regionprops(labels_pred)
        y_center, y_bbox = record_connected_domain(props_pred, lenth=np.max(labels_pred))
        temp_y_center = y_center
        props_pred = merge_connected_domains(temp_y_center)

        # Check for FN
        if len(props_ref) > 0:
            for iprop in props_ref:
                if len(props_pred) == 0:
                    FN += 1
                    FN_case.append(iprop.centroid)
                else:
                    d_pred = [distance_mitosis(np.array((i_prop_pred[0],i_prop_pred[1])), np.array(iprop.centroid)) for i_prop_pred
                              in props_pred]
                    if np.min(d_pred) > maximum_d:
                        FN += 1
                        FN_case.append(iprop.centroid)

        # Check for TP and FP
        if len(props_pred) > 0:
            for iprop in props_pred:
                if len(props_ref) == 0:
                    FP += 1
                    FP_case.append((iprop[0],iprop[1]))
                else:
                    d_ref = [distance_mitosis(np.array(i_prop_ref.centroid), np.array((iprop[0],iprop[1]))) for i_prop_ref in
                             props_ref]
                    if np.min(d_ref) < maximum_d:
                        TP += 1
                        TP_case.append((iprop[0],iprop[1]))
                    else:
                        FP += 1
                        FP_case.append((iprop[0],iprop[1]))

        if save_visualization:

            r = 40
            w = 10
            for icentroid in FN_case:
                color = (0, 0, 255)
                draw.ellipse((icentroid[1] - r, icentroid[0] - r, icentroid[1] + r, icentroid[0] + r),
                             outline=color, fill=None, width=w)
            for icentroid in TP_case:
                color = (0, 255, 0)
                draw.ellipse((icentroid[1] - r, icentroid[0] - r, icentroid[1] + r, icentroid[0] + r),
                             outline=color, fill=None, width=w)
            for icentroid in FP_case:
                color = (255, 255, 0)

                draw.ellipse((icentroid[1] - r, icentroid[0] - r, icentroid[1] + r, icentroid[0] + r),
                             outline=color, fill=None, width=w)

            if len(FN_case) == 0 and len(TP_case) == 0 and len(FP_case) == 0:
                if visualize_negatives:
                    img.save(dir_out + 'negatives/' + filename)
            else:
                img.save(dir_out + 'positives/' + filename )

    precision = TP / (TP + FP + 1e-3)
    recall = TP / (TP + FN + 1e-3)

    F1 = (2 * recall * precision) / (recall + precision + 1e-3)

    return {'TP': TP, 'FP': FP, 'FN': FN, 'F1': F1}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def record_connected_domain(properties, lenth):
    point_properties = properties
    region_center = []
    region_bbox = []
    for i in range(lenth):
        prop = point_properties[i]
        if prop.area >= 7:
            center = prop.centroid
            bbox = prop.bbox
            region_center.append(center)
            region_bbox.append(bbox)
    return region_center, region_bbox


def merge_connected_domains(temp_center_list):
    merge_center_list = []
    for i in range(len(temp_center_list)):
        if temp_center_list[i][0] == -100 and temp_center_list[i][1] == -100:
            continue
        start_x = temp_center_list[i][0]
        start_y = temp_center_list[i][1]
        mean_x = 0
        mean_y = 0
        temp = []
        for j in range(len(temp_center_list)):
            if temp_center_list[j][0] == -100 and temp_center_list[j][1] == -100:
                continue
            else:
                image_x = temp_center_list[j][0]
                image_y = temp_center_list[j][1]
                distance = np.sqrt(pow((image_x - start_x), 2) + pow((image_y - start_y), 2))
                if distance <= 10:
                    temp.append((image_x, image_y))
                    if distance != 0:
                        temp_center_list[j] = ([-100, -100])

        for t in range(len(temp)):
            mean_x += temp[t][0]
            mean_y += temp[t][1]
        mean_x = mean_x / len(temp)
        mean_y = mean_y / len(temp)
        merge_center_list.append((mean_x, mean_y))

    return merge_center_list


class InferDataset():
    def __init__(self, image_file, data_points, width, highth):

        self.image = np.array(io.imread(image_file),dtype=np.float32)
        self.data_points = data_points

        self.width = width
        self.highth = highth

    def __getitem__(self, index):
        [x_center, y_center] = self.data_points[index]
        img = self.image[x_center - self.width // 2:x_center + self.width // 2,
              y_center - self.highth // 2:y_center + self.highth // 2, :]
        img = imutils.resize(img, height=args.input_shape[1])
        im = np.transpose(img, (2, 0, 1)) / 255.0

        return x_center, y_center, im

    def __len__(self):
        return len(self.data_points)


def test_phase(args):
    dir_model = args.dir_model
    thor = args.thor
    input_path = args.input_path  # "path"
    output_path = args.output_path  # "output_path"
    h_path = args.h_pth
    num_workers = args.num_workers
    batch_size = args.batch_size

    image_path = input_path + "/nor_JPEGImages"
    mask_path = args.mask_path
    label_gt_path = input_path + "/csv"
    savepath = output_path + '/pre'
    txt_output = output_path + "/pre_point"
    pre_h_mask = output_path + "/pre_h_mask"

    for dir_name in [savepath, txt_output, pre_h_mask]:
        if os.path.exists(dir_name):
            continue
        else:
            os.mkdir(dir_name)

    width = 80
    highth = 80
    ############################################################
    if args.model == 'res18_all':
        with open(dir_model + 'setting.txt', 'r') as f:
            setting = json.load(f)
        # Load model
        weights = torch.load(dir_model + "network_weights_best.pth")
        if not "backbone" in list(setting.keys()):
            setting['backbone'] = "RN18"
            for key in list(weights.keys()):
                weights[key.replace('resnet18_model', 'model')] = weights.pop(key)
        model = Resnet(in_channels=3, n_classes=2, n_blocks=setting['n_blocks'], pretrained=True,
                       mode=setting['mode'], aggregation=setting['aggregation'], backbone=setting['backbone']).to(device)
        model.load_state_dict(weights, strict=True)


    patch_names_list = os.listdir(image_path)


    TP, FP, FN = [], [], []
    for patch_name in tqdm(patch_names_list):

        infer_points_list = []
        basename = patch_name[:-4]
        image_file = image_path + '/' + patch_name
        h_file = h_path + '/' + patch_name
        masl_file = mask_path + '/' + patch_name
        h_channel = Image.open(h_file)
        h_channel = np.array(h_channel, dtype='float32')



        im = np.array(io.imread(image_file))
        [w, h] = im.shape[:2]

        im = np.transpose(im, (2, 0, 1))/255

        output_map = np.zeros((w, h))


        mask = np.array(io.imread(masl_file))
        mask = mask / 255
        mask = np.double(mask > 0)


        y_region = measure.label(h_channel, connectivity=2, background=0)
        y_properties = measure.regionprops(y_region)
        y_center, y_bbox = record_connected_domain(y_properties, lenth=np.max(y_region))
        temp_y_center = y_center
        m_center = merge_connected_domains(temp_y_center)
        for region in m_center:
            x_center = int(region[0])
            y_center = int(region[1])

            if x_center - width // 2 < 0:
                x_center = width // 2
            if y_center - highth // 2 < 0:
                y_center = highth // 2
            if x_center + width // 2 > w:
                x_center = w - width // 2
            if y_center + highth // 2 > h:
                y_center = h - highth // 2

            point = [x_center, y_center]
            infer_points_list.append(point)

        if len(infer_points_list)%batch_size==1:
            infer_points_list.append((256,256))
        infer_dataset = InferDataset(image_file, infer_points_list, width=width, highth=highth)
        infer_data_loader = DataLoader(infer_dataset,
                                       shuffle=False,
                                       num_workers=num_workers,
                                       pin_memory=False,
                                       batch_size=batch_size
                                       )
        torch.cuda.empty_cache()


        for iter, (x_center, y_center, img_list) in enumerate(infer_data_loader):

            model.eval()
            with torch.no_grad():
                pred_logits, _, cam_logits = model(img_list.float().clone().detach().to(device))
                prob2 = torch.softmax(pred_logits, dim=-1).cpu().data.numpy()
            for (x, y, img_,pre) in zip(x_center, y_center, img_list,prob2):
                    # i = i + 1
                if pre[1] >= 0.5:
                    img_ = img_.unsqueeze(0)

                    model.eval()
                    with torch.no_grad():
                        pred_logits, _, cam_logits = model(img_.float().clone().detach().to(device))
                        mask_pred = torch.nn.sigmoid(cam_logits[:, 1, :, :].squeeze().cpu().detach().numpy())

                    if x - width // 2 < 0:
                        x = width // 2
                    if y - highth // 2 < 0:
                        y = highth // 2
                    if x + width // 2 > w:
                        x = w - width // 2
                    if y + highth // 2 > h:
                        y = h - highth // 2

                    norm_cam = np.array(mask_pred)
                    norm_cam = cv2.resize(norm_cam,(80,80))
                    output_map[int(x) - width // 2:int(x) + width // 2,
                    int(y) - highth // 2:int(y) + highth // 2] = np.maximum(norm_cam, \
                                                                            output_map[
                                                                            int(x) - width // 2:int(x) + width // 2,
                                                                            int(y) - highth // 2:int(y) + highth // 2])
        loc_test_metrics = evaluate_motisis_localization(X=np.expand_dims(im, 0),
                                                         M=np.expand_dims(mask, 0),
                                                         Mhat=np.expand_dims(output_map, 0),
                                                         filename = patch_name,
                                                         dir_out=savepath,
                                                         th=thor,
                                                         save_visualization=True)

        TP.append(loc_test_metrics['TP'])
        FP.append(loc_test_metrics['FP'])
        FN.append(loc_test_metrics['FN'])

    TP = np.sum(TP)
    FP = np.sum(FP)
    FN = np.sum(FN)

    precision = TP / (TP + FP + 1e-3)
    recall = TP / (TP + FN + 1e-3)
    F1 = (2 * recall * precision) / (recall + precision + 1e-3)

    print("Results: F1=%2.3f | Recall=%2.3f | Precision=%2.3f | " % (F1, recall, precision), end="\n")
    print("Disentangled: TP=%d | FP=%d | FN=%d | " % (TP, FP, FN), end="\n")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimenmion for model')
    parser.add_argument("--dataset", default='MIDOG21', type=str)
    parser.add_argument("--model", default='res18_all', type=str,help='res18,res18_all')
    parser.add_argument("--dir_model", default='/root/cls/local_data/results/2021/last_sup_2021_4/',type=str)
    parser.add_argument("--input_shape", default=(3, 80, 80), type=list)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--thor", default=0.9, type=float)
    parser.add_argument("--num_workers", default=8, type=int)

    parser.add_argument("--root_dir", default=r'/root/data/cls', type=str)
    parser.add_argument("--input_path", default=r'/root/data/cls/2021_hard/test_all', type=str)
    parser.add_argument("--output_path", default=r'/root/data/cls/2021_hard/test_all/output', type=str)
    parser.add_argument("--h_pth",default=r'/root/data/cls/2021_hard/test_all/nor_h_channel', type=str)
    parser.add_argument("--mask_path", default=r'/root/data/MIDOG2021/point_mask_simple', type=str)#point_mask_hard point_mask_simple
    args = parser.parse_args()
    test_phase(args)