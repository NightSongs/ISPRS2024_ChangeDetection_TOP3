# -*- coding: utf-8 -*-
"""
@Time    : 2022/11/28/028 13:46
@Author  : NDWX
@File    : tools.py
@Software: PyCharm
"""
import os
import glob


def check_zero_percentage(matrix, threshold=0.8):
    """
    检查 2D 矩阵中像素值为 0 的比例是否大于指定阈值.

    参数:
    matrix (numpy.ndarray): 输入的 2D 矩阵
    threshold (float, optional): 阈值,默认为 0.8 (80%)

    返回:
    bool: 如果像素值为 0 的比例大于阈值,返回 True,否则返回 False
    """
    total_pixels = matrix.size
    zero_pixels = (matrix == 0).sum()
    zero_percentage = zero_pixels / total_pixels

    return zero_percentage > threshold


def get_dataset(txt_root, data_root):
    train_imageA_set, train_imageB_set, train_label_set = [], [], []
    val_imageA_set, val_imageB_set, val_label_set = [], [], []
    with open(os.path.join(txt_root, "train.txt"), "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            train_imageA_set.append(os.path.join(data_root, "T1/{}第二期影像.tif".format(line)))
            train_imageB_set.append(os.path.join(data_root, "T2/{}第三期影像.tif".format(line)))
            train_label_set.append(os.path.join(data_root, "gt/{}.tif".format(line)))
    train_dataset = [train_imageA_set, train_imageB_set, train_label_set]

    with open(os.path.join(txt_root, "val.txt"), "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            val_imageA_set.append(os.path.join(data_root, "T1/{}第二期影像.tif".format(line)))
            val_imageB_set.append(os.path.join(data_root, "T2/{}第三期影像.tif".format(line)))
            val_label_set.append(os.path.join(data_root, "gt/{}.tif".format(line)))
    val_dataset = [val_imageA_set, val_imageB_set, val_label_set]
    return train_dataset, val_dataset
# def get_dataset(txt_root, data_root):
#     import cv2
#     train_imageA_set, train_imageB_set, train_label_set = [], [], []
#     val_imageA_set, val_imageB_set, val_label_set = [], [], []
#     with open(os.path.join(txt_root, "train.txt"), "r") as f:
#         for line in f.readlines():
#             line = line.strip('\n')
#             label = cv2.imread(os.path.join(data_root, "gt/{}.tif".format(line)), -1)
#             if (50 in label) or (220 in label) or (100 in label):
#                 for _ in range(3):
#                     train_imageA_set.append(os.path.join(data_root, "T1/{}第二期影像.tif".format(line)))
#                     train_imageB_set.append(os.path.join(data_root, "T2/{}第三期影像.tif".format(line)))
#                     train_label_set.append(os.path.join(data_root, "gt/{}.tif".format(line)))
#             else:
#                 train_imageA_set.append(os.path.join(data_root, "T1/{}第二期影像.tif".format(line)))
#                 train_imageB_set.append(os.path.join(data_root, "T2/{}第三期影像.tif".format(line)))
#                 train_label_set.append(os.path.join(data_root, "gt/{}.tif".format(line)))
#     train_dataset = [train_imageA_set, train_imageB_set, train_label_set]
#
#     with open(os.path.join(txt_root, "val.txt"), "r") as f:
#         for line in f.readlines():
#             line = line.strip('\n')
#             val_imageA_set.append(os.path.join(data_root, "T1/{}第二期影像.tif".format(line)))
#             val_imageB_set.append(os.path.join(data_root, "T2/{}第三期影像.tif".format(line)))
#             val_label_set.append(os.path.join(data_root, "gt/{}.tif".format(line)))
#     val_dataset = [val_imageA_set, val_imageB_set, val_label_set]
#     return train_dataset, val_dataset


def get_test_dataset(data_root):
    test_imageA_set = sorted(glob.glob(os.path.join(data_root, "T1/*.tif")))
    test_imageB_set = sorted(glob.glob(os.path.join(data_root, "T2/*.tif")))
    test_dataset = [test_imageA_set, test_imageB_set]
    return test_dataset


def get_only_val_dataset(txt_root, data_root):
    val_imageA_set, val_imageB_set = [], []
    with open(os.path.join(txt_root, "val.txt"), "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            val_imageA_set.append(os.path.join(data_root, "T1/{}第二期影像.tif".format(line)))
            val_imageB_set.append(os.path.join(data_root, "T2/{}第三期影像.tif".format(line)))
    val_dataset = [val_imageA_set, val_imageB_set]
    return val_dataset
