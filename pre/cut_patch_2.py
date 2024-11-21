import random
#分类预处理

import argparse
import glob
import json
import sys

import imutils
import math
import os
import shutil
import re

import cv2
import torch
sys.path.append(os.getcwd())
from constants import *
from skimage import measure, io
from tqdm import tqdm
import numpy as np
from PIL import Image
import pandas as pd
from net.Res_18.RN18 import Resnet,dua_network
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

Image.MAX_IMAGE_PIXELS = None
def record_connected_domain(properties, lenth):
    point_properties = properties
    region_center = []
    region_bbox = []
    for i in range(lenth):
        prop = point_properties[i]
        if prop.area >= 20:
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
                if distance <= 55:
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

def get_bbox_df(annotation_file="/drive/MyDrive/MIDOG_Challenge/MIDOG.json",only_mitosis = True):
    #MIDOG
    hamamatsu_rx_ids = list(range(0, 51))
    hamamatsu_360_ids = list(range(51, 101))
    aperio_ids = list(range(101, 151))
    leica_ids = list(range(151, 201))

    rows = []
    with open(annotation_file) as f:
        data = json.load(f)
        categories = {1: 'mitotic figure', 2: 'hard negative'}

        for row in data["images"]:
            file_name = row["file_name"]
            image_id = row["id"]
            width = row["width"]
            height = row["height"]

            scanner = "Hamamatsu XR"
            if image_id in hamamatsu_360_ids:
                scanner = "Hamamatsu S360"
            if image_id in aperio_ids:
                scanner = "Aperio CS"
            if image_id in leica_ids:
                scanner = "Leica GT450"

            for annotation in [anno for anno in data['annotations'] if anno["image_id"] == image_id]:
                box = annotation["bbox"]
                cat = categories[annotation["category_id"]]
                point = [0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])]
                if only_mitosis:
                    if cat == 'mitotic figure':
                        rows.append([file_name, image_id, width, height, box, point, cat, scanner])
                else:
                    rows.append([file_name, image_id, width, height, box, point, cat, scanner])
    df = pd.DataFrame(rows, columns=["file_name", "image_id", "width", "height", "box", "point", "cat", "scanner"])
    return (df)

def gen_batches(iterator, batch_size, include_partial=True):
  """ generate the tile batches from the tile iterator
  Args:
    iterator: the tile iterator
    batch_size: batch size
    include_partial: boolean value to keep the partial batch or not

  Return:
    the iterator for the tile batches
  """
  batch = []
  for item in iterator:
    batch.append(item)
    if len(batch) == batch_size:
      yield batch
      batch = []
  if len(batch) > 0 and include_partial:
    yield batch


def create_mask(h, w, im, coords, radius):
    mask = np.zeros((h, w), dtype=bool)
    for row, col in coords:
        assert 0 <= row <= h, "row is outside of the image height"
        assert 0 <= col <= w, "col is outside of the image width"

        # mitosis mask as a circle with radius `radius` pixels centered on the given location
        y, x = np.ogrid[:h, :w]
        mitosis_mask = np.sqrt((y - row)**2 + (x - col)**2) <= radius

        # indicate mitosis patch area on mask
        mask = np.logical_or(mask, mitosis_mask)

    # Set pixels in the mask to 1 where the input image is black
    black_pixels = np.all(im == 0, axis=-1)  # Check if all color channels are 0
    mask[black_pixels] = 1

    return mask



def extract_patch(im, row, col, size):

  dims = np.ndim(im)
  assert dims >= 2, "image must be of shape (h, w, ...)"
  h, w = im.shape[0:2]
  assert 0 <= row <= h, "row {} is outside of the image height {}".format(row, h)
  assert 0 <= col <= w, "col {} is outside of the image width {}".format(col, w)

  half_size = round(size / 2)
  row_lower = int(row) - half_size
  row_upper = int(row) + half_size
  col_lower = int(col) - half_size
  col_upper = int(col) + half_size

  # clip the bounds to the size of the image and compute padding to add to patch
  row_pad_lower = abs(row_lower) if row_lower < 0 else 0
  row_pad_upper = row_upper - h if row_upper > h else 0
  col_pad_lower = abs(col_lower) if col_lower < 0 else 0
  col_pad_upper = col_upper - w if col_upper > w else 0
  row_lower = max(0, row_lower)
  row_upper =min(row_upper, h)
  col_lower = max(0, col_lower)
  col_upper = min(col_upper, w)

  # extract patch
  patch = im[row_lower:row_upper, col_lower:col_upper]

  # pad with reflection on the height and width as needed to yield a patch of the desired size
  # NOTE: all remaining dimensions (such as channels) receive 0 padding
  padding = ((row_pad_lower, row_pad_upper), (col_pad_lower, col_pad_upper)) + ((0, 0),) * (dims-2)

  # Note: the padding content starts from the second row/col of the
  # input patch instead of the first row/col
  patch_padded = np.pad(patch, padding, 'reflect')

  return patch_padded


def gen_normal_coords(mask,center_list, patch_size):

    assert np.ndim(mask) == 2, "mask must be of shape (h, w)"
    h, w = mask.shape


    for row, col in center_list:
        half_size = round(patch_size / 2)
        row_lower = int(row - half_size)
        row_upper = int(row + half_size)
        col_lower = int(col - half_size)
        col_upper = int(col + half_size)

        # clip the bounds to the size of the image and compute padding to add to patch
        row_pad_lower = abs(row_lower) if row_lower < 0 else 0
        row_pad_upper = row_upper - h if row_upper > h else 0
        col_pad_lower = abs(col_lower) if col_lower < 0 else 0
        col_pad_upper = col_upper - w if col_upper > w else 0
        row_lower = max(0, row_lower)
        row_upper = min(row_upper, h)
        col_lower = max(0, col_lower)
        col_upper = min(col_upper, w)

        # extract patch
        patch = mask[row_lower:row_upper, col_lower:col_upper]
        # check that the patch region around (row, col) is not in a mitotic region
        # half_patch_size = patch_size // 2
        # patch_region = mask[row - half_patch_size:row + half_patch_size, col - half_patch_size:col + half_patch_size]

        if np.sum(patch) == 0:
            yield row, col

class InferDataset():
    def __init__(self, image_file, data_points, width, highth):
        # img = cv2.imread(image_file)
        # self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.image = np.array(io.imread(image_file),dtype=np.float32)
        self.data_points = data_points
        # self.transform = transform
        self.width = width
        self.highth = highth

    def __getitem__(self, index):
        [x_center, y_center] = self.data_points[index]
        x_center = int(x_center)
        y_center = int(y_center)
        img1 = self.image[x_center - self.width // 2:x_center + self.width // 2,
              y_center - self.highth // 2:y_center + self.highth // 2, :]

        img1 = imutils.resize(img1, height=args.input_shape[1])#做resiz操作
        im2 = np.transpose(img1, (2, 0, 1)) / 255.0
        # if self.transform:
        #     img = self.transform(img)
        #     img = Image.fromarray(img)
        return x_center, y_center, im2

    def __len__(self):
        return len(self.data_points)

def gen_fp_coords(im_path,normal_coords, model,pred_threshold, batch_size):

  ori = np.array(io.imread(im_path))
  [w, h] = ori.shape[:2]

  highth = args.patch_size
  width = args.patch_size
  result = []
  infer_points_list = []
  if len(normal_coords) % batch_size == 1:
      normal_coords.append((50, 50))
  for region in normal_coords:
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


  infer_dataset = InferDataset(im_path, infer_points_list, width=width, highth=highth)
  infer_data_loader = DataLoader(infer_dataset,
                                     shuffle=False,
                                     num_workers=8,
                                     pin_memory=False,
                                     batch_size=batch_size
                                     )
  torch.cuda.empty_cache()
  for iter, (x_center, y_center, img_list) in enumerate(infer_data_loader):
      model.eval()
      with torch.no_grad():
          pred_logits, _, cam_logits = model(img_list.float().clone().detach().cuda())
          prob2 = torch.softmax(pred_logits, dim=-1).cpu().data.numpy()
      for (x, y ,pre) in zip(x_center, y_center, prob2):
          if pre[1] > pred_threshold:
              result.append((x,y))
  return result

def gen_random_translation(h, w, row, col, max_shift):

  assert 0 <= row <= h, "row is outside of the image height"
  assert 0 <= col <= w, "col is outside of the image width"
  assert max_shift >= 0, "max_shift must be >= 0"

  # NOTE: np.random.randint has exclusive upper bounds
  row_shifted = min(max(0, row + np.random.randint(-max_shift, max_shift + 1)), h)
  col_shifted = min(max(0, col + np.random.randint(-max_shift, max_shift + 1)), w)
  row_shift = row_shifted - row
  col_shift = col_shifted - col
  return row_shift, col_shift


def gen_patches(im,gt,coords, size, translations, max_shift, p,mitosis):

  assert np.ndim(im) == 3, "image must be of shape (h, w, c)"
  # h, w, c = im.shape
  h, w, c = im.shape
  assert 1 < size <= min(h, w), "size must be > 1 and within the bounds of the image"
  # assert rotations >= 0, "rotations must be >0"
  assert translations >= 0, "translations must be >0"
  assert max_shift >= 0, "max_shift must be >= 0"
  assert 0 <= p <= 1, "p must be a valid decimal probability"

  # convert to uint8 type in order to use PIL to rotate
  orig_dtype = im.dtype
  im = im.astype(np.uint8)
  gt = gt.astype(np.uint8)
  rads = math.pi / 4  # 45 degrees, which is worst case
  bounding_size = math.ceil((size+2*max_shift) * (math.cos(rads) + math.sin(rads)))
  row_center = col_center = round(bounding_size / 2)
  # TODO: either emit a warning, or add a parameter to allow empty corners
  #assert bounding_size < min(h, w), "patch size is too large to avoid empty corners after rotation"
  #
  if mitosis == 1:
      for row, col in coords:
        rotate_type = np.random.choice([0, 1], p=[0.8, 0.2])
        gt_bounding_patch = Image.fromarray(extract_patch(gt, row, col, bounding_size))
        bounding_patch = Image.fromarray(extract_patch(im, row, col, bounding_size))  # PIL for rotation
        # start_angle = np.random.randint(0, 61)  # Randomly select start_angle between 0 and 60
        # rotations
        # for theta in np.linspace(start_angle, 180, rotations+1, dtype=int):  # always include 0 degrees
        if rotate_type == 0:
            theta = np.random.randint(60, 181)
            rotated_patch = np.asarray(bounding_patch.rotate(theta, Image.BILINEAR))  # then back to numpy
            gt_rotated_patch = np.asarray(gt_bounding_patch.rotate(theta, Image.BILINEAR))  # then back to numpy
            # random translations
            shifts = [gen_random_translation(h, w, row, col, max_shift) for _ in range(translations)]
            for row_shift, col_shift in [(0, 0)] + shifts:  # always include 0 shift
                # translations_type = np.random.choice([0, 1], p=[0.5, 0.5])
                # if translations_type == 0:
                patch = extract_patch(rotated_patch, row_center + row_shift, col_center + col_shift, size)
                gt_patch = extract_patch(gt_rotated_patch, row_center + row_shift, col_center + col_shift, size)
                patch = patch.astype(orig_dtype)  # convert back to original data type
                gt_patch = gt_patch.astype(orig_dtype)  # convert back to original data type
                # sample from a Bernoulli distribution with probability `p` 取消
                yield patch,gt_patch, row, col, theta, row_shift, col_shift
                # else:
                #     patch = extract_patch(rotated_patch, row_center + 0, col_center + 0, size)
                #     gt_patch = extract_patch(gt_rotated_patch, row_center + 0, col_center + 0, size)
                #     patch = patch.astype(orig_dtype)  # convert back to original data type
                #     gt_patch = gt_patch.astype(orig_dtype)  # convert back to original data type
                #     # sample from a Bernoulli distribution with probability `p` 取消
                #     yield patch, gt_patch, row, col, theta, 0, 0
        else:
            theta = 0
            rotated_patch = np.asarray(bounding_patch.rotate(theta, Image.BILINEAR))  # then back to numpy
            gt_rotated_patch = np.asarray(gt_bounding_patch.rotate(theta, Image.BILINEAR))  # then back to numpy
            shifts = [gen_random_translation(h, w, row, col, max_shift) for _ in range(translations)]
            for row_shift, col_shift in [(0, 0)] + shifts:  # always include 0 shift
                # translations_type = np.random.choice([0, 1], p=[0.5, 0.5])
                # if translations_type == 0:
                patch = extract_patch(rotated_patch, row_center + row_shift, col_center + col_shift, size)
                gt_patch = extract_patch(gt_rotated_patch, row_center + row_shift, col_center + col_shift, size)
                gt_patch = gt_patch.astype(orig_dtype)
                patch = patch.astype(orig_dtype)  # convert back to original data type
                # sample from a Bernoulli distribution with probability `p，取消
                yield patch, gt_patch,row, col, theta, row_shift, col_shift
                # else:
                #     patch = extract_patch(rotated_patch, row_center + 0, col_center + 0, size)
                #     gt_patch = extract_patch(gt_rotated_patch, row_center + 0, col_center + 0, size)
                #     patch = patch.astype(orig_dtype)  # convert back to original data type
                #     gt_patch = gt_patch.astype(orig_dtype)  # convert back to original data type
                #     # sample from a Bernoulli distribution with probability `p` 取消
                #     yield patch, gt_patch, row, col, theta, 0, 0
  elif mitosis == 0:
      for row, col in coords:
        gt_bounding_patch = Image.fromarray(extract_patch(gt, row, col, bounding_size))
        bounding_patch = Image.fromarray(extract_patch(im, row, col, bounding_size))  # PIL for rotation
        theta = 0
        rotated_patch = np.asarray(bounding_patch.rotate(theta, Image.BILINEAR))  # then back to numpy
        gt_rotated_patch = np.asarray(gt_bounding_patch.rotate(theta, Image.BILINEAR))  # then back to numpy
        shifts = [gen_random_translation(h, w, row, col, max_shift) for _ in range(translations)]
        for row_shift, col_shift in [(0, 0)] + shifts:  # always include 0 shift
            patch = extract_patch(rotated_patch, row_center + row_shift, col_center + col_shift, size)
            gt_patch = extract_patch(gt_rotated_patch, row_center + row_shift, col_center + col_shift, size)
            gt_patch = gt_patch.astype(orig_dtype)
            patch = patch.astype(orig_dtype)  # convert back to original data type
            # sample from a Bernoulli distribution with probability `p`
            yield patch, gt_patch, row, col, theta, row_shift, col_shift

def save_patch(patch, gt_patch,path,gt_path,case, row, col, rotation, row_shift, col_shift,suffix,mitosis,
    ext="png"):

  # TODO: extract filename generation and arg extraction into separate functions
  if int(mitosis) == 1:#有丝分裂
      filename = f"{case}_{row}_{col}_{rotation}_{row_shift}_{col_shift}_{suffix}_{1}.{ext}"
      file_path = os.path.join(path, filename)
      Image.fromarray(patch).save(file_path, subsampling=0, quality=100)

      gt_filename = f"{case}_{row}_{col}_{rotation}_{row_shift}_{col_shift}_{suffix}_{1}.{ext}"
      gt_file_path = os.path.join(gt_path, gt_filename)
      Image.fromarray(gt_patch).save(gt_file_path, subsampling=0, quality=100)
  elif int(mitosis) == 0:#非有丝分离
      # NOTE: the subsampling and quality parameters will only affect jpeg images
      gt_filename = f"{case}_{row}_{col}_{rotation}_{row_shift}_{col_shift}_{suffix}_{0}.{ext}"
      gt_file_path = os.path.join(gt_path, gt_filename)
      Image.fromarray(gt_patch).save(gt_file_path, subsampling=0, quality=100)

      filename = f"{case}_{row}_{col}_{rotation}_{row_shift}_{col_shift}_{suffix}_{0}.{ext}"
      file_path = os.path.join(path, filename)
      Image.fromarray(patch).save(file_path, subsampling=0, quality=100)

def process(args):
    train_size = args.train_size
    dataset = args.dataset
    seed = np.random.randint(1e9)
    patch_size = args.patch_size
    translations_train = args.translations_train

    p_train = args.p_train
    stride_train = round(patch_size * (3 / 4))


    mode = args.mode
    images_path = args.images_path
    labels_path = args.labels_path
    gt_path = args.gt_path
    h_label_pth = args.norhchannel_path
    base_save_path = args.base_save_path
    dist = args.dist
    max_shift = round(patch_size / 5)

    np.random.seed(seed)

    if dataset == "tupac":
        ext = "tif"
    elif dataset == 'midog2021':
        ext = 'png'
    elif dataset == "icpr2014":
        # 实现icpr2014数据集的处理逻辑
        ext = 'png'
    else:
        raise Exception("erro")
    train_args = ('train', translations_train, p_train, stride_train)
    if args.fp_mdoel:
        model = Resnet(in_channels=3, n_classes=2, n_blocks=3,
                       pretrained=False,
                       mode='instance', aggregation='max', backbone='Res_18').cuda()
        model.load_state_dict(torch.load(args.model_path))

    for split_args in [train_args]:
        split_name, translations, p, stride = split_args

        if dataset == 'tupac':
            region_im_all_paths = glob.glob(os.path.join(images_path, f"*.{ext}"))
            val_id, test_id = TUPAC16_ID_VAL, TUPAC16_ID_TEST
            val_test_ids = val_id + test_id
            file_names = [os.path.basename(path).split('.')[0] for path in region_im_all_paths]
            if mode == 'train':
                idx = np.in1d(file_names, val_test_ids)
                region_im_paths = [region_im_all_paths[i] for i in range(len(region_im_all_paths)) if not idx[i]]
            elif mode == 'test':
                idx = np.in1d(file_names, test_id)
                region_im_paths = [region_im_all_paths[i] for i in range(len(region_im_all_paths)) if idx[i]]

        elif dataset == 'icpr2014':
            region_im_paths = glob.glob(os.path.join(images_path, f"*.{ext}"))
        elif dataset == 'midog2021':
            region_im_all_paths = glob.glob(os.path.join(images_path, f"*.{ext}"))
            val_id, test_id = MIDOG21_ID_VAL,MIDOG21_ID_TEST
            val_test_ids = val_id + test_id
            file_names = [os.path.basename(path).split('.')[0] for path in region_im_all_paths]
            if mode == 'train':
                idx = np.in1d(file_names, val_test_ids)
                region_im_paths = [region_im_all_paths[i] for i in range(len(region_im_all_paths)) if not idx[i]]
            elif mode == 'test':
                idx = np.in1d(file_names, test_id)
                region_im_paths = [region_im_all_paths[i] for i in range(len(region_im_all_paths)) if idx[i]]
        for region_im_path in tqdm(region_im_paths):
            if dataset == 'tupac':
                region_gt_path = os.path.join(gt_path, f"{os.path.splitext(os.path.basename(region_im_path))[0]}_restored.tif")
                region_h_label_path = os.path.join(h_label_pth,f"{os.path.splitext(os.path.basename(region_im_path))[0]}.png")
            elif dataset == 'midog2021':
                # region_gt_path = os.path.join(gt_path, f"{os.path.splitext(os.path.basename(region_im_path))[0]}_restored.tiff")
                region_gt_path = os.path.join(gt_path,
                                              f"{os.path.splitext(os.path.basename(region_im_path))[0]}.png")
                region_h_label_path = os.path.join(h_label_pth,
                                                   f"{os.path.splitext(os.path.basename(region_im_path))[0]}.png")
            elif dataset == 'icpr2014':
                region_gt_path = os.path.join(gt_path,
                                              f"{os.path.splitext(os.path.basename(region_im_path))[0]}.png")
                region_h_label_path = os.path.join(h_label_pth,
                                                   f"{os.path.splitext(os.path.basename(region_im_path))[0]}.png")
            im = np.array(Image.open(region_im_path))
            gt = np.array(Image.open(region_gt_path))
            h_label = Image.open(region_h_label_path)

            mask = np.array(h_label, dtype='float32')
            y_region = measure.label(mask, connectivity=2, background=0)
            y_properties = measure.regionprops(y_region)
            y_center, y_bbox = record_connected_domain(y_properties, lenth=np.max(y_region))
            temp_y_center = y_center
            m_center = merge_connected_domains(temp_y_center)


            h, w, c = im.shape

            if dataset == 'tupac':
                case = os.path.basename(region_im_path)[:2]

                coords_path = os.path.join(labels_path, f"Mitosis_{case}_coordinates.csv")

            elif dataset == 'midog2021':
                case = os.path.basename(region_im_path)[:3]
                coords_path = os.path.join(labels_path, f"Mitosis_{case}_coordinates.csv")
            elif dataset == 'icpr2014':
                case = os.path.basename(region_im_path).split(".")[0]
                coords_path = os.path.join(labels_path, f"Mitosis_{case}_coordinates.csv")
            else:
                coords_path = ''
                print(case, 'no coords path')

            if os.path.isfile(coords_path):
                if dataset == "tupac":
                    coords = np.loadtxt(coords_path, dtype=np.int64, delimiter=',', ndmin=2,
                                        usecols=(0, 1))
                elif dataset == 'midog2021':
                    coords = np.loadtxt(coords_path, dtype=np.int64, delimiter=',', ndmin=2,
                                        usecols=(0, 1))
                elif dataset == 'icpr2014':
                    coords = np.loadtxt(coords_path, dtype=np.int64, delimiter=',', ndmin=2,
                                        usecols=(0, 1))
            else:
                coords = []
                print(case,'no coords')

            mitosis = 1
            save_path = os.path.join(base_save_path, split_name, "JPEGImages")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            gt_save_path = os.path.join(base_save_path, split_name, "gt")
            if not os.path.exists(gt_save_path):
                os.makedirs(gt_save_path)

            patch_gen = gen_patches(im, gt, coords, patch_size, translations, max_shift, 1, mitosis)
            o = 0
            for i, (patch, gt_patch, row, col, rot, row_shift, col_shift) in enumerate(patch_gen):
                save_patch(patch, gt_patch, save_path, gt_save_path, case, row, col, rot, row_shift, col_shift, i, mitosis)
                o = o + 1
            if o == 0:
                o = 1

            mitosis = 0
            if args.fp_mdoel:
                mask = create_mask(h, w, im, coords, dist)
                normal_coords_gen = gen_normal_coords(mask, m_center, patch_size)
                normal_coords_list = list(normal_coords_gen)
                fp_coords = gen_fp_coords(region_im_path,normal_coords_list,model,pred_threshold=0.5,batch_size=128)
                patch_gen = gen_patches(im, gt, fp_coords, patch_size, 0, max_shift, p, mitosis)
                for i, (patch, gt_patch, row, col, rot, row_shift, col_shift) in enumerate(patch_gen):
                    save_patch(patch, gt_patch, save_path, gt_save_path, case, row, col, rot, row_shift, col_shift, i, mitosis)
            else:
                mask = create_mask(h, w, im, coords, dist)
                normal_coords_gen = gen_normal_coords(mask, m_center, patch_size)
                normal_coords_list = list(normal_coords_gen)
                random_selected_coords = random.sample(normal_coords_list,min(o,len(normal_coords_list)))
                patch_gen = gen_patches(im, gt, random_selected_coords, patch_size, 0, max_shift, p, mitosis)
                for i, (patch, gt_patch, row, col, rot, row_shift, col_shift) in enumerate(patch_gen):
                    save_patch(patch, gt_patch, save_path, gt_save_path, case, row, col, rot, row_shift, col_shift, i, mitosis)

    point_mask_save_path = os.path.join(base_save_path, split_name, "SegmentationClass")
    if not os.path.exists(point_mask_save_path):
        os.makedirs(point_mask_save_path)

    for root, dirs, files in os.walk(gt_save_path):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                mask = np.array(io.imread(image_path))
                [w, h] = mask.shape[:2]
                mask = mask / 255
                mask = np.double(mask > 0)

                y_region = measure.label(mask, connectivity=2, background=0)
                y_properties = measure.regionprops(y_region)

                mask = np.zeros((w, h))
                for prop in y_properties:

                    center_y, center_x = prop.centroid
                    center = (int(center_y), int(center_x))
                    mask[center] = 1
                cv2.imwrite(os.path.join(point_mask_save_path,file),mask * 255)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='process')
    parser.add_argument('--mode', type=str, default='train', help='traintest?')
    #################################################################################################
    parser.add_argument('--dataset', type=str, default='icpr2014', help='tupac,midog2021,icpr2014')
    #########NS
    parser.add_argument('--fp_mdoel', type=bool, default=True, help='是否需要模型负样本采样')
    parser.add_argument('--model_path', type=str,
                        default='/root/cls/local_data/results/2014/1st_2014_fp_search/network_weights_best.pth',
                        help='模型路径')
    parser.add_argument("--input_shape", default=(3, 80, 80), type=list)
    #######path
    parser.add_argument('--images_path', type=str, default=r'/root/data/icpr2014/JPEGImages',
                        help='')
    parser.add_argument('--labels_path', type=str, default=r'/root/data/icpr2014/tog_csv_reverse',
                        help='')
    parser.add_argument('--gt_path', type=str,
                        default=r'/root/data/icpr2014/mask_circle_simple', help='GT')
    parser.add_argument('--norhchannel_path', type=str,
                        default=r'/root/data/icpr2014/nor_hchannel', help='Hcahnnel')
    parser.add_argument('--base_save_path', type=str, default=r'/root/data/cls/2014/fp/',
                        help='')
    ##################################################################################################################
    parser.add_argument('--patch_size', type=int, default=80, help='')
    parser.add_argument('--translations_train', type=int, default=3,help='aug number')
    parser.add_argument('--dist', type=int, default=80, help='dis thr')
    args = parser.parse_args()
    process(args)
