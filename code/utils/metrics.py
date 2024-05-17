# -*- coding: utf-8 -*-
"""
@Time    : 2022/11/28/028 13:45
@Author  : NDWX
@File    : metrics.py
@Software: PyCharm
"""
import numpy as np
import torch
import torch.nn.functional as func

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# @torch.no_grad()
# def cal_val_iou(model, loader, num_classes):
#     val_seg_iou = []
#     model.eval()
#     for x1, x2, seg_y in loader:
#         x1, x2, seg_y = x1.to(DEVICE), x2.to(DEVICE), seg_y.to(DEVICE)
#         seg_out = model(x1, x2)
#
#         if seg_out.shape[-2:] != seg_y.shape[-2:]:
#             seg_out = func.interpolate(seg_out, size=seg_y.shape[-2:], mode='bilinear', align_corners=True)
#
#         seg_out = seg_out.argmax(1)
#         iou_seg = cal_iou(seg_out, seg_y, num_classes)
#         val_seg_iou.append(iou_seg)
#     return np.mean(val_seg_iou)

# def cal_iou(pred, mask, c):
#     iou_result = []
#     for idx in range(c):
#         p = (mask == idx).int().reshape(-1)
#         t = (pred == idx).int().reshape(-1)
#         uion = p.sum() + t.sum()
#         overlap = (p * t).sum()
#         iou = 2 * overlap / (uion + 1e-8)
#         iou_result.append(iou.abs().data.cpu().numpy())
#     return np.stack(iou_result)
@torch.no_grad()
def cal_val_iou(model, loader, num_classes):
    val_seg_iou, output_list = [], []
    model.eval()
    for x1, x2, seg_y in loader:
        x1, x2, seg_y = x1.to(DEVICE), x2.to(DEVICE), seg_y.to(DEVICE)
        seg_out = model(x1, x2)
        if seg_out.shape[-2:] != seg_y.shape[-2:]:
            seg_out = func.interpolate(seg_out, size=seg_y.shape[-2:], mode='bilinear', align_corners=True)
        seg_out = seg_out.argmax(1).data.cpu().numpy()
        seg_y = seg_y.data.cpu().numpy()
        for i in range(seg_out.shape[0]):
            output_list.append([seg_out[i], seg_y[i]])
    # 计算IoU
    for c in range(num_classes):
        true_positive = 0
        false_positive = 0
        false_negative = 0
        for [pred_img, gt_img] in output_list:
            true_positive += np.sum((gt_img == c) & (pred_img == c))
            false_positive += np.sum((gt_img != c) & (pred_img == c))
            false_negative += np.sum((gt_img == c) & (pred_img != c))
        miou = true_positive / (true_positive + false_positive + false_negative + 1e-8)
        val_seg_iou.append(miou)
    return np.mean(val_seg_iou)
