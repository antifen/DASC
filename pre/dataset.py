import re

import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2
import torchvision
import imutils
import numpy as np
import random
import os
import torch
import kornia
from PIL import Image

from skimage import io
import torchvision.transforms as transforms


from pre.constants import *

import matplotlib.pyplot as plt
import pre.newTransforms as myTransforms

plt.interactive(False)
scl_augmentation = myTransforms.Compose([
    myTransforms.RandomChoice([myTransforms.RandomHorizontalFlip(p=1),
                               myTransforms.RandomVerticalFlip(p=1),
                               myTransforms.AutoRandomRotation()]),  # above is for: randomly selecting one for process
    # myTransforms.RandomAffineCV2(alpha=0.1),  # alpha \in [0,0.15],
    # myTransforms.RandomAffine(degrees=0, translate=[0, 0.2], scale=[0.8, 1.2], shear=[-10, 10, -10, 10], fillcolor=(228, 218, 218)),
    myTransforms.RandomElastic(alpha=2, sigma=0.06),
    myTransforms.ColorJitter(brightness=(0.65, 1.35), contrast=(0.5, 1.5)),
    myTransforms.RandomChoice([myTransforms.ColorJitter(saturation=(0, 2), hue=0.3),
                               myTransforms.HEDJitter(theta=0.05)]),
    myTransforms.RandomGaussBlur(radius=[0.5, 1.5]),
    # myTransforms.ToTensor(),  # operated on original image, rewrite on previous transform.
    # myTransforms.Normalize([0.6270, 0.5013, 0.7519], [0.1627, 0.1682, 0.0977])
])
strong_augmentation = myTransforms.Compose([
    myTransforms.RandomChoice([myTransforms.RandomHorizontalFlip(p=1),
                               myTransforms.RandomVerticalFlip(p=1),
                               myTransforms.AutoRandomRotation()]),  # above is for: randomly selecting one for process
    # myTransforms.RandomAffineCV2(alpha=0.1),  # alpha \in [0,0.15],
    # myTransforms.RandomAffine(degrees=0, translate=[0, 0.2], scale=[0.8, 1.2], shear=[-10, 10, -10, 10], fillcolor=(228, 218, 218)),
    myTransforms.RandomElastic(alpha=2, sigma=0.06),
    myTransforms.ColorJitter(brightness=(0.65, 1.35), contrast=(0.5, 1.5)),
    myTransforms.RandomChoice([myTransforms.ColorJitter(saturation=(0, 2), hue=0.3),
                               myTransforms.HEDJitter(theta=0.05)]),
    myTransforms.RandomGaussBlur(radius=[0.5, 1.5])
    # myTransforms.ToTensor(),  # operated on original image, rewrite on previous transform.
    # myTransforms.Normalize([0.6270, 0.5013, 0.7519], [0.1627, 0.1682, 0.0977])
])

weak_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1, hue=0.1),  # 轻微调整颜色
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),  # 轻微旋转和平移
    transforms.RandomVerticalFlip(p=0.5),

    # transforms.ToTensor()  # 转换为张量
])
sigle_augmentation = myTransforms.Compose([
    myTransforms.RandomHorizontalFlip(p=1),
    myTransforms.RandomVerticalFlip(p=1),
    myTransforms.AutoRandomRotation(),
    myTransforms.ColorJitter(brightness=(0.65, 1.35), contrast=(0.5, 1.5)),
    myTransforms.RandomChoice([myTransforms.ColorJitter(saturation=(0, 2), hue=0.3),
                               myTransforms.HEDJitter(theta=0.05)]),
    # myTransforms.ToTensor(),  # operated on original image, rewrite on previous transform.
    # myTransforms.Normalize([0.6270, 0.5013, 0.7519], [0.1627, 0.1682, 0.0977])
])
def resize_mask_with_points(mask, new_size):
    original_size = mask.shape
    points = np.argwhere(mask > 0)
    values = mask[mask > 0]
    scale_x = new_size[1] / original_size[1]
    scale_y = new_size[0] / original_size[0]
    new_points = np.round(points * [scale_y, scale_x]).astype(int)
    new_mask = np.zeros(new_size, dtype=np.uint8)
    for idx, point in enumerate(new_points):
        if 0 <= point[0] < new_size[0] and 0 <= point[1] < new_size[1]:
            new_mask[point[0], point[1]] = values[idx]
    return new_mask

class Dataset(object):
    def __init__(self, dataset_id, partition='train', input_shape=(3, 512, 512), labels=1, preallocate=True, dir_images='images_norm', dir_masks='dir_masks',sigle = False,mask_train = False,
                 select_list = None , hard_label = None,scl = False,linear = False,transform = None):
        if "TUPAC16" in dataset_id:

            dir_dataset = PATH_TUPAC_Train_IMAGES
        elif "MIDOG21" in dataset_id:

            dir_dataset = PATH_MIODG21_Train_IMAGES
        else:
            print("Processing dataset not valid... ")
            return

        self.dir_dataset = dir_dataset
        self.partition = partition#是train,test,还是val
        self.resize = torchvision.transforms.Resize((input_shape[-2], input_shape[-1]))
        self.labels = labels
        self.preallocate = preallocate#提前加载图片
        self.input_shape = input_shape
        self.dir_images = dir_images
        self.dir_masks = dir_masks
        self.hard_label = hard_label
        self.sigle = sigle
        self.mask = mask_train
        self.scl = scl
        self.linear = linear
        self.aug = transform
        self.new_labels = []
        self.transform = transform
        ###############GANZHOU dataset  test val 的路径改变
        if self.partition == 'test' or self.partition == 'val':
            if "MIDOG21" in dataset_id:
                dir_dataset = PATH_MIDOG21_Test_IMAGES
                print(dir_dataset)
            if "TUPAC16" in dataset_id:
                dir_dataset = PATH_TUPAC_Test_IMAGES
                print(dir_dataset)
        self.images = os.listdir(dir_dataset + self.dir_images + '/')#读取图片

        #选择图片
        if select_list:
            self.images = [img for img in self.images if img in select_list]

        if self.partition == 'train':
                self.images = random.sample(self.images, len(self.images)//1)
                print('train',len(self.images))

        elif self.partition == 'val':

            self.images = random.sample(self.images, len(self.images)//1)
            print('val',len(self.images))

        elif self.partition == 'test':
            self.images = random.sample(self.images, len(self.images)//1)
            print('test',len(self.images))

        else:
            print('Wrong partition', end='\n')

        if self.preallocate:
            # Pre-load images
            if self.mask:
                self.M = np.zeros((len(self.images), input_shape[1], input_shape[2]), dtype=np.float32)
                self.N = np.zeros((len(self.images), 1))
            self.X = np.zeros((len(self.images), input_shape[0], input_shape[1], input_shape[2]), dtype=np.float32)
            self.Y = np.zeros((len(self.images), labels))
            self.Hard_Y = np.zeros((len(self.images), labels))
            for iImage in np.arange(0, len(self.images)):
                print(str(iImage + 1) + '/' + str(len(self.images)), end='\r')
                im = np.array(io.imread(dir_dataset + self.dir_images + '/' + self.images[iImage]), dtype=np.float32)
                im = imutils.resize(im, height=self.input_shape[1])
                im = np.transpose(im, (2, 0, 1))
                if self.mask:
                    mask = np.array(io.imread(dir_dataset + self.dir_masks + '/' + self.images[iImage]))
                    # mask = imutils.resize(mask, height=self.input_shape[1]) / 255
                    if self.input_shape[1] != 80:
                        mask = resize_mask_with_points(mask,(self.input_shape[1],self.input_shape[2]))
                    mask = mask / 255
                    mask = np.double(mask > 0)

                # # Intensity normalization
                # im = im / 255
                if hard_label:
                    self.Hard_Y[iImage,:] = np.array(self.hard_label[str(self.images[iImage])])

                if os.path.isfile(dir_dataset + self.dir_masks + '/' + self.images[iImage]):
                        if  int(re.search(r'_(\d+)\.png$', self.images[iImage]).group(1))== 1:
                            self.Y[iImage, :] = np.array([1.0])
                        elif int(re.search(r'_(\d+)\.png$', self.images[iImage]).group(1))== 0:
                            self.Y[iImage, :] = np.array([0.0])
                        else:
                            assert ('no label match')
                else:
                    assert ('no masks!!')



                self.X[iImage, :, :, :] = im
                if self.mask:
                    self.M[iImage, :, :] = mask
                    self.N[iImage, :] = np.sum(mask[:, :])

        if self.partition == 'train' and self.mask:
            idx = np.squeeze(np.argwhere(np.squeeze(self.N) <= 1))
            self.filter_cases(idx)


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.images)

    def __getitem__(self, index):
        'Generates one sample of data'
        if self.partition =='train':
            if self.hard_label:
                image = self.X[index, :, :, :]
                image = np.transpose(image, (1, 2, 0))
                image = Image.fromarray(np.uint8(image))

                weak = weak_augmentation(image)
                weak = np.transpose(weak, (2, 0, 1))
                weak = np.array(weak) / 255.0
                weak = weak.astype(np.float32)

                y = self.Y[index, :].astype(int)
                hard_y = self.Hard_Y[index, :].astype(int)
                return weak, y,hard_y,index

            if self.scl:
                if self.new_labels != []:
                    new_labels = self.new_labels[index]
                else:
                    new_labels = -1
                image = self.X[index, :, :, :]
                image = np.transpose(image, (1, 2, 0))
                image = Image.fromarray(np.uint8(image))

                if self.transform is not None:
                    img = self.transform(image)

                y = self.Y[index, :].astype(int)
                hard_y = self.Hard_Y[index, :].astype(int)
                return img, y, hard_y, new_labels,index

            if self.linear:
                if self.new_labels != []:
                    new_labels = self.new_labels[index]
                else:
                    new_labels = -1
                image = self.X[index, :, :, :]

                image = np.transpose(image, (1, 2, 0))
                image = Image.fromarray(np.uint8(image))
                img = scl_augmentation(image)

                img = np.transpose(img, (2, 0, 1))
                img = np.array(img) / 255.0
                img = img.astype(np.float32)
                # print(img.shape)
                y = self.Y[index, :].astype(int)
                hard_y = self.Hard_Y[index, :].astype(int)
                return img,y,hard_y,new_labels
            if self.sigle:
                if self.mask:
                    x = self.X[index, :, :, :].astype(np.float32)/255.0
                    m = self.M[index, :, :].astype(int)
                    y = self.Y[index, :].astype(int)
                    return x, y, m
                else:
                    image = self.X[index, :, :, :]
                    image = np.transpose(image, (1, 2, 0))
                    image = Image.fromarray(np.uint8(image))

                    image = sigle_augmentation(image)
                    image = np.transpose(image, (2, 0, 1))
                    image = np.array(image) / 255.0
                    image = image.astype(np.float32)

                    y = self.Y[index, :].astype(int)
                    return image, y, index
            else:
                image= self.X[index, :, :, :]
                image = np.transpose(image, (1, 2, 0))
                image = Image.fromarray(np.uint8(image))

                weak = weak_augmentation(image)
                weak = np.transpose(weak, (2, 0, 1))
                weak =  np.array(weak) /255.0
                weak = weak.astype(np.float32)


                strong = strong_augmentation(image)
                strong = np.transpose(strong, (2, 0, 1))
                strong = np.array(strong) /255.0
                strong = strong.astype(np.float32)
                y = self.Y[index, :].astype(int)
                return weak,strong,y,index
        elif self.partition=='test' or self.partition == 'val':
            if self.mask:
                x = self.X[index, :, :, :].astype(np.float32)/255.0
                m = self.M[index, :, :].astype(int)
                y = self.Y[index, :].astype(int)

                return x, y, m
            else:
                x = (self.X[index, :, :, :]/255.0).astype(np.float32)
                y = self.Y[index, :].astype(int)
                return x, y


    def get_all_labels(self):
        all_labels = []
        for i in range(len(self)):
            _, _, y, _= self[i]
            all_labels.append(y[0])
        return all_labels

    def get_scl_lable_labels(self):
        all_labels = []
        for i in range(len(self)):
            _, y, _, _ = self[i]
            all_labels.append(y[0])
        return all_labels

class Generator(object):
    def __init__(self, dataset, batch_size, shuffle=True, balance=False,sigle = False,mask_train = False):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(0, len(self.dataset.images))
        self._idx = 0
        self.balance = balance
        self.sigle = sigle
        self.mask_train = mask_train
        if self.balance:
            self.indexes = balance_dataset(self.indexes, self.dataset.Y.flatten())
        self._reset()
        self.augmentations = AugmentationsSegmentation(strong_augmentation=False)
    def __len__(self):
        N = self.dataset.X.shape[0]
        b = self.batch_size
        return N // b

    def __iter__(self):
        return self

    def __next__(self):

        if self._idx + self.batch_size > self.dataset.X.shape[0]:
            self._reset()
            raise StopIteration()

        # Load images and include into the batch
        if self.sigle==False:
            X,S, Y = [], [], []
            index_list = []
            for i in range(self._idx, self._idx + self.batch_size):
                x,s,y,index = self.dataset.__getitem__(self.indexes[i])
                X.append(np.expand_dims(x, axis=0))
                S.append(np.expand_dims(s, axis=0))
                Y.append(np.expand_dims(y, axis=0))
                index_list.append(index)
            # Update index iterator
            self._idx += self.batch_size

            X = np.concatenate(X, axis=0)
            S = np.concatenate(S, axis=0)
            Y = np.concatenate(Y, axis=0)

            return X,S,Y,index_list
        elif self.sigle:
            if self.mask_train:
                X, Y, M = [], [], []
                for i in range(self._idx, self._idx + self.batch_size):
                    x, y, m = self.dataset.__getitem__(self.indexes[i])

                    X.append(np.expand_dims(x, axis=0))
                    Y.append(np.expand_dims(y, axis=0))
                    M.append(np.expand_dims(m, axis=0))

                # Update index iterator
                self._idx += self.batch_size

                X = np.concatenate(X, axis=0)
                Y = np.concatenate(Y, axis=0)
                M = np.concatenate(M, axis=0)

                return X, Y, M
            else:
                X, Y = [], []
                index_list = []
                for i in range(self._idx, self._idx + self.batch_size):
                    x, y, index = self.dataset.__getitem__(self.indexes[i])
                    X.append(np.expand_dims(x, axis=0))
                    Y.append(np.expand_dims(y, axis=0))
                    index_list.append(index)
                # Update index iterator
                self._idx += self.batch_size

                X = np.concatenate(X, axis=0)
                Y = np.concatenate(Y, axis=0)

                return X, Y, index_list


    def _reset(self):
        if self.shuffle:
            random.shuffle(self.indexes)
        self._idx = 0


def balance_dataset(indexes, Y):
    classes = [0, 1]
    counts = np.bincount(Y.astype(int))
    upsampling = [round(np.max(counts)/counts[iClass]) for iClass in classes]

    indexes_new = []
    for iClass in classes:
        if upsampling[iClass] == 1:
            indexes_iclass = indexes[Y == classes[iClass]]
        else:
            indexes_iclass = np.random.choice(indexes[Y == classes[iClass]], counts[iClass]*upsampling[iClass])
        indexes_new.extend(indexes_iclass)

    indexes_new = np.array(indexes_new)

    return indexes_new

class AugmentationsSegmentation(torch.nn.Module):
    def __init__(self, strong_augmentation=False):
        super(AugmentationsSegmentation, self).__init__()

        self.strong_augmentation = strong_augmentation

        # we define and cache our operators as class members
        self.kHor = kornia.augmentation.RandomHorizontalFlip(p=0.5)
        self.kVert = kornia.augmentation.RandomVerticalFlip(p=0.5)
        self.kAffine = kornia.augmentation.RandomRotation(p=0.5, degrees=[-90, 90])
        self.kTransp = RandomTranspose(p=0.5)

        if self.strong_augmentation:
            self.kElastic = kornia.augmentation.RandomElasticTransform(p=0.5)

    def forward(self, img, mask):
        img_out = img

        # Apply geometric tranform
        img_out = self.kTransp(self.kAffine(self.kVert(self.kHor(img_out))))

        # Infer geometry params to mask
        mask_out = self.kTransp(self.kAffine(self.kVert(self.kHor(mask, self.kHor._params), self.kVert._params), self.kAffine._params), self.kTransp._params)

        if self.strong_augmentation:
            img_out = self.kElastic(img_out)
            mask_out = self.kElastic(mask_out, self.kElastic._params)

        return img_out, mask_out

class RandomTranspose(torch.nn.Module):
    def __init__(self, p):
        super(RandomTranspose, self).__init__()
        self.p = p
        self._params = 0

    def forward(self, x, params=None):
        # Get random state for the operation
        if params is None:
            p = random.random()
            self._params = p
        else:
            p = self._params
        # Apply transform
        if p > 0.5:
            return torch.transpose(x, -2, -1)
        else:
            return x
