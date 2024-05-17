# -*- coding: utf-8 -*-
"""
@Time    : 2024/01/16/022 13:53
@Author  : NDWX
@File    : infer.py
@Software: PyCharm
"""
import cv2
import numpy as np
import os
import time
import torch
import torch.nn.functional as func
from torch.optim.swa_utils import AveragedModel
from tqdm import tqdm

from nets.upernet import UPerNet
from utils.dataset import build_infer_dataloader
from utils.tools import get_only_val_dataset, get_test_dataset

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def remap4submit(label, color_dict):
    new_label = np.zeros(shape=label.shape, dtype=np.uint8)
    for key in color_dict:
        new_label[label == key] = color_dict[key]
    return new_label


def remap4eval(label, color_dict):
    new_label = np.zeros(shape=label.shape, dtype=np.uint8)
    for key in color_dict:
        new_label[label == color_dict[key]] = key
    return new_label


def calculate_miou(gt_dir, pred_dir, color_dict):
    # 计算mIoU
    class_num = 7
    miou_list = []
    for i in range(class_num):
        true_positive = 0
        false_positive = 0
        false_negative = 0
        # 只计算val集
        for filename in os.listdir(pred_dir):
            gt_path = os.path.join(gt_dir, filename)
            pred_path = os.path.join(pred_dir, filename)

            gt_img = cv2.imread(gt_path, -1)
            pred_img = cv2.imread(pred_path, -1)

            gt_img = remap4eval(gt_img, color_dict)
            pred_img = remap4eval(pred_img, color_dict)

            true_positive += np.sum((gt_img == i) & (pred_img == i))
            false_positive += np.sum((gt_img != i) & (pred_img == i))
            false_negative += np.sum((gt_img == i) & (pred_img != i))
        miou = true_positive / (true_positive + false_positive + false_negative + 1e-8)
        miou_list.append(miou)
    mean_miou = np.mean(miou_list)
    print(f"Mean IoU: {mean_miou:.4f}")


def find_small_areas(image, k):
    image = image.astype(np.uint8)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= k:
            cv2.drawContours(mask, [contour], 0, 1, -1)
    return mask


if __name__ == '__main__':
    time1 = time.time()
    SWA = False
    EVAL = True
    batch_size = 16
    test_data_root = "/mnt/c/dataset_tmp"
    txt_root = "/mnt/d/competition/ISPRS2024/data/split/fold1"
    save_dir = "/mnt/d/competition/ISPRS2024/results"
    os.makedirs(save_dir, exist_ok=True)
    test_dataset = get_test_dataset(test_data_root)

    model = UPerNet(backbone_name='convnext_base_clip',
                    dropout=0.5,
                    drop_path_rate=0.5,
                    pretrained=False,
                    num_classes=7,
                    fusion_form='conv',
                    scse=False)
    if SWA:
        model = AveragedModel(model)
    model = model.cuda()

    state_dict = torch.load(f"checkpoints/upernet_convnext_base_clip_drop0.5_droppath0.5_Convf_62e_aug/seg_model_all.pth")

    model.load_state_dict(state_dict)
    model.eval()

    test_loader = build_infer_dataloader(test_dataset, batch_size)

    color_dict = {
        0: 0,
        1: 200,
        2: 150,
        3: 100,
        4: 250,
        5: 220,
        6: 50
    }
    output_images = []

    for x1, x2, path in tqdm(test_loader):
        output = 0
        with torch.no_grad():
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            out1 = model(x1, x2)
            if out1.shape[-2:] != [610, 610]:
                out1 = func.interpolate(out1, size=(610, 610), mode='bilinear', align_corners=True)
            out1 = out1.cpu().data.numpy()
            output = output + out1
        for i in range(output.shape[0]):
            file_name = path[i]
            new_name = file_name.split("第")[0]
            pred = np.argmax(output[i], axis=0).astype(np.uint8)
            pred = remap4submit(pred, color_dict)

            # tmp = np.zeros_like(pred)
            # tmp[pred > 0] = 1
            #
            # mask = find_small_areas(tmp, 250)
            # pred[mask == 1] = 0  # 后处理需要考虑时间分数

            output_images.append((os.path.join(save_dir, f"{new_name}.tif"), pred))

    for image_path, image_data in output_images:
        cv2.imwrite(image_path, image_data)
    time2 = time.time()
    print(f"infer time: {round(time2 - time1, 2)}s")

    if EVAL:
        gt_dir = "/mnt/d/competition/ISPRS2024/data/gt"
        calculate_miou(gt_dir, save_dir, color_dict)
