# -*- coding: utf-8 -*-
"""
@Time    : 2024/01/16/022 14:41
@Author  : NDWX
@File    : train.py
@Software: PyCharm
"""
import glob
import logging
import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from nets.upernet import UPerNet
from utils.dataset import build_dataloader
from utils.log import init_log
from utils.losses import DiceLoss

warnings.filterwarnings('ignore')
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# 固定随机种子
def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class HybirdLoss(nn.Module):
    def __init__(self, num_classes):
        super(HybirdLoss, self).__init__()
        self.num_classes = num_classes
        self.CELoss_fn = nn.CrossEntropyLoss()
        self.DiceLoss_fn = DiceLoss(mode='multiclass')

    def forward(self, pred, mask):
        one_hot_mask = self.to_one_hot(mask, self.num_classes, pred.device)
        loss_dice = self.DiceLoss_fn(pred, mask)
        m_list = torch.tensor([890054152, 8996211, 17260286, 2498273, 8101873, 715084, 2624119],
                              device=pred.device).float()
        # 计算频率
        m_list /= m_list.sum()
        # 计算LDAM(Label-Distribution-Aware Margin Loss)损失
        max_m = 6  # 这个超参数越大, 少数类别更突出
        m_list = 1.0 / torch.sqrt(torch.sqrt(m_list))
        m_list = m_list * (max_m / torch.max(m_list))
        batch_m = torch.einsum(
            "bchw,c->bchw", one_hot_mask, m_list
        )
        logits_m = pred - batch_m
        output = torch.where(one_hot_mask.bool(), logits_m, pred)
        loss_ce = self.CELoss_fn(self.get_probas(output), mask)
        loss = loss_ce + loss_dice
        return loss

    @staticmethod
    def get_probas(logits):
        return torch.softmax(logits, dim=2)

    @staticmethod
    def to_one_hot(mask, num_classes, device):
        b, h, w = mask.size()
        one_hot_mask = torch.zeros(b, num_classes, h, w, device=device)
        new_mask = mask.unsqueeze(1).clone()
        new_mask[torch.where(new_mask == 255)] = 0
        one_hot_mask.scatter_(1, new_mask, 1)
        return one_hot_mask.long()


# 加载模型
def load_model(DEVICE, num_classes):
    model = UPerNet(backbone_name='convnext_large_clip',
                    dropout=0.5,
                    drop_path_rate=0.5,
                    pretrained=True,
                    num_classes=num_classes,
                    fusion_form='conv',
                    scse=False)
    # model = UPerNet(backbone_name='hrnet_w48',
    #                 dropout=0.5,
    #                 drop_path_rate=0.5,
    #                 pretrained=True,
    #                 num_classes=num_classes,
    #                 fusion_form='conv',
    #                 scse=False)
    model.to(DEVICE)
    return model


# 训练函数
def train(num_epochs, optimizer, scheduler, loss_fn, train_loader, model, save_path):
    best_train_loss = 100.0
    for epoch in range(num_epochs):
        model.train()
        losses = []
        lr = optimizer.param_groups[0]['lr']
        for batch_index, (x1, x2, seg_y) in enumerate(
                tqdm(train_loader, desc=f'Epoch {epoch + 1} Train', leave=False)):
            optimizer.zero_grad()
            x1, x2, seg_y = x1.to(DEVICE), x2.to(DEVICE), seg_y.to(DEVICE)
            seg_y = seg_y.to(torch.long)
            seg_out = model(x1, x2)
            loss = loss_fn(seg_out, seg_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        train_loss = np.array(losses).mean()
        if best_train_loss >= train_loss and epoch >= 30:
            best_train_loss = train_loss
            torch.save(model.state_dict(), save_path)
            logger.info(
                f'\n Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, lr = {lr:.6f}, model saved')
        else:
            logger.info(
                f'\n Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, lr = {lr:.6f}')
    return best_train_loss


if __name__ == '__main__':
    random_seed = 42
    num_epochs = 62  # 510 254 126 62
    num_classes = 7
    batch_size = 8
    lr = 1e-4
    setup_seed(random_seed)
    data_root = '/mnt/c/dataset_tmp'
    model_save_dir = "upernet_convnext_large_clip_drop0.5_droppath0.5_Convf_62e_aug"
    # model_save_dir = "upernet_hrnet_w48_drop0.5_droppath0.5_Convf_62e_aug"
    os.makedirs(f"code/checkpoints/{model_save_dir}", exist_ok=True)
    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    train_dataset = [sorted(glob.glob('/mnt/c/dataset_tmp/T1/*.tif')),
                     sorted(glob.glob('/mnt/c/dataset_tmp/T2/*.tif')),
                     sorted(glob.glob('/mnt/c/dataset_tmp/gt/*.tif'))]
    train_loader, valid_loader = build_dataloader(train_dataset, train_dataset, int(batch_size))

    model_save_path = f"code/checkpoints/{model_save_dir}/seg_model_all.pth"
    model = load_model(DEVICE, num_classes)

    if 'clip' in model_save_dir:
        parameters = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'backbone' in name:
                parameters.append({'params': param, 'lr': 1e-5})
            else:
                parameters.append({'params': param, 'lr': lr})
        optimizer = torch.optim.AdamW(parameters,
                                      lr=lr, weight_decay=1e-3)
    else:
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=lr, weight_decay=1e-3)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=2,
        T_mult=2,
        eta_min=1e-5,
    )

    loss = HybirdLoss(num_classes).cuda()

    best_iou = train(num_epochs, num_classes, optimizer, scheduler, loss, train_loader, valid_loader, model,
                     model_save_path)
