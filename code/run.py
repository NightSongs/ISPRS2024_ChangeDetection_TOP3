# -*- coding: utf-8 -*-
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as func
from tqdm import tqdm
import time

from nets.upernet import UPerNet
from utils.dataset import build_infer_dataloader
from utils.tools import get_test_dataset

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def remap(label, color_dict):
    new_label = np.zeros(shape=label.shape, dtype=np.uint8)
    for key in color_dict:
        new_label[label == key] = color_dict[key]
    return new_label


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
    batch_size = 16
    time.sleep(100)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    model_path_list = ["checkpoints/upernet_resnet18_drop0.5_droppath0.5_Convf_62e_w-o-aug/seg_model_all.pth"]

    test_dataset = get_test_dataset(input_dir)

    for i, model_path in enumerate(model_path_list):
        i += 1
        model = UPerNet(backbone_name='resnet18',
                        dropout=0.5,
                        drop_path_rate=0.5,
                        pretrained=False,
                        num_classes=7,
                        fusion_form='conv',
                        scse=False)
        # if 'base' in model_path:
            # model = UPerNet(backbone_name='convnext_base_clip',
            #                 dropout=0.5,
            #                 drop_path_rate=0.5,
            #                 pretrained=False,
            #                 num_classes=7,
            #                 fusion_form='conv',
            #                 scse=False)
        # else:
        #     model = UPerNet(backbone_name='convnext_large_clip',
        #                     dropout=0.5,
        #                     drop_path_rate=0.5,
        #                     pretrained=False,
        #                     num_classes=7,
        #                     fusion_form='conv',
        #                     scse=False)
        model = model.cuda()
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        model.eval()
        globals()[f'model{i}'] = model  # 以全局变量的形式提前加载模型, 能够有效地加速推理过程(需要忽略编辑器报错)

    model_list = [model1]

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

            for model in model_list:
                out1 = model(x1, x2)

                if out1.shape[-2:] != [610, 610]:
                    out1 = func.interpolate(out1, size=(610, 610), mode='bilinear', align_corners=True)

                out1 = out1.cpu().data.numpy()

                output = output + out1

        output = output / len(model_list)

        for i in range(output.shape[0]):
            file_name = path[i]
            new_name = file_name.split("第")[0]
            pred = np.argmax(output[i], axis=0).astype(np.uint8)
            pred = remap(pred, color_dict)

            # tmp = np.zeros_like(pred)
            # tmp[pred > 0] = 1

            # mask = find_small_areas(tmp, 250)
            # pred[mask == 1] = 0

            output_images.append((os.path.join(output_dir, f"{new_name}.tif"), pred))

    for image_path, image_data in output_images:
        cv2.imwrite(image_path, image_data)
