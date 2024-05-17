# -*- coding: utf-8 -*-
"""
@Time    : 2024/01/16/022 13:39
@Author  : NDWX
@File    : dataset.py
@Software: PyCharm
"""
import os

import albumentations as A
import cv2
import numpy as np
import torch.utils.data as D
from albumentations.pytorch import ToTensorV2
from skimage import io
import random

from .transform import CopyPaste, ISPRSLabelResize


# 构建dataset
class change_dataset(D.Dataset):
    def __init__(self, image_A_paths, image_B_paths, label_paths, mode):
        self.image_A_paths = image_A_paths
        self.image_B_paths = image_B_paths
        self.label_paths = label_paths
        self.mode = mode
        self.len = len(image_A_paths)
        assert len(image_A_paths) == len(image_B_paths), '前后时相影像数量不匹配'

        self.infer_transform = A.Compose([
            A.Normalize(),
            ToTensorV2()
        ], additional_targets={'image_2': 'image'})

        self.val_transform = A.Compose([
            ISPRSLabelResize(),
            A.Normalize(),
            ToTensorV2()
        ], additional_targets={'image_2': 'image'}, is_check_shapes=False)

        self.train_transform = A.Compose([
            ISPRSLabelResize(),
            CopyPaste(p=0.5),
            A.Flip(p=0.5),
            A.Rotate(30, p=0.5),
            A.RandomResizedCrop(height=512, width=512, scale=(0.5, 1.0), p=0.5),
            A.HueSaturationValue(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ], additional_targets={'image_2': 'image'}, is_check_shapes=False)

    @staticmethod
    def __remap__(label):
        color_dict = {
            0: 0,
            1: 200,
            2: 150,
            3: 100,
            4: 250,
            5: 220,
            6: 50
        }
        new_label = np.zeros(shape=label.shape)
        for key in color_dict:
            new_label[label == color_dict[key]] = key
        return new_label

    def __getitem__(self, index):
        imageA = io.imread(self.image_A_paths[index])
        imageB = io.imread(self.image_B_paths[index])

        if self.mode == "train":
            label = cv2.imread(self.label_paths[index], -1)
            label = self.__remap__(label)

            if label.max() == 0:
                new_index = index
                while label.max() == 0:
                    new_index = random.randint(0, len(self.image_A_paths) - 1)
                    label = cv2.imread(self.label_paths[new_index], -1)
                    label = self.__remap__(label)

                imageA = io.imread(self.image_A_paths[new_index])
                imageB = io.imread(self.image_B_paths[new_index])

            transformed_data = self.train_transform(image=imageA, image_2=imageB, mask=label)
            imageA, imageB, label = transformed_data['image'], transformed_data['image_2'], transformed_data['mask']
            return imageA, imageB, label

        elif self.mode == "val":
            label = cv2.imread(self.label_paths[index], -1)
            label = self.__remap__(label)

            transformed_data = self.val_transform(image=imageA, image_2=imageB, mask=label)
            imageA, imageB, label = transformed_data['image'], transformed_data['image_2'], transformed_data['mask']
            return imageA, imageB, label

        elif self.mode == "infer":
            transformed_data = self.infer_transform(image=imageA, image_2=imageB)
            imageA, imageB = transformed_data['image'], transformed_data['image_2']
            return imageA, imageB, os.path.basename(self.image_A_paths[index])

    def __len__(self):
        return self.len


# 构建数据加载器
def get_dataloader(image_A_paths, image_B_paths, label_paths, mode, batch_size,
                   shuffle, num_workers, drop_last):
    dataset = change_dataset(image_A_paths, image_B_paths, label_paths, mode)

    dataloader = D.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=True, drop_last=drop_last)
    return dataloader


# 生成dataloader
def build_dataloader(train_path, val_path, batch_size):
    train_loader = get_dataloader(train_path[0], train_path[1], train_path[2], "train",
                                  batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)
    valid_loader = get_dataloader(val_path[0], val_path[1], val_path[2], "val",
                                  batch_size,
                                  shuffle=False,
                                  num_workers=0,
                                  drop_last=False)
    return train_loader, valid_loader


# 生成infer dataloader
def build_infer_dataloader(infer_path, batch_size):
    infer_loader = get_dataloader(infer_path[0], infer_path[1], None, "infer",
                                  batch_size,
                                  shuffle=False,
                                  num_workers=0,
                                  drop_last=False)
    return infer_loader
