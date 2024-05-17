import os
import cv2
import numpy as np


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


if __name__ == '__main__':
    gt_dir = "/mnt/d/competition/ISPRS2024/data/gt"
    pred_dir = "/mnt/d/competition/ISPRS2024/results"
    color_dict = {
        0: 0,
        1: 200,
        2: 150,
        3: 100,
        4: 250,
        5: 220,
        6: 50
    }
    calculate_miou(gt_dir, pred_dir, color_dict)
