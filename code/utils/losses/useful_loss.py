import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# from .dice import DiceLoss
# from .joint_loss import JointLoss
# from .soft_ce import SoftCrossEntropyLoss


class EdgeLoss(nn.Module):
    def __init__(self, ignore_index=255, edge_factor=1.0):
        super(EdgeLoss, self).__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.edge_factor = edge_factor

    def get_boundary(self, x):
        laplacian_kernel_target = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).cuda(device=x.device)
        x = x.unsqueeze(1).float()
        x = F.conv2d(x, laplacian_kernel_target, padding=1)
        x = x.clamp(min=0)
        x[x >= 0.1] = 1
        x[x < 0.1] = 0

        return x

    def compute_edge_loss(self, logits, targets):
        bs = logits.size()[0]
        boundary_targets = self.get_boundary(targets)
        boundary_targets = boundary_targets.view(bs, 1, -1)
        # print(boundary_targets.shape)
        logits = F.softmax(logits, dim=1).argmax(dim=1).squeeze(dim=1)
        boundary_pre = self.get_boundary(logits)
        boundary_pre = boundary_pre / (boundary_pre + 0.01)
        # print(boundary_pre)
        boundary_pre = boundary_pre.view(bs, 1, -1)
        # print(boundary_pre)
        # dice_loss = 1 - ((2. * (boundary_pre * boundary_targets).sum(1) + 1.0) /
        #                  (boundary_pre.sum(1) + boundary_targets.sum(1) + 1.0))
        # dice_loss = dice_loss.mean()
        edge_loss = F.binary_cross_entropy_with_logits(boundary_pre, boundary_targets)

        return edge_loss

    def forward(self, logits, targets):
        loss = (self.main_loss(logits, targets) + self.compute_edge_loss(logits, targets) * self.edge_factor) / (
                self.edge_factor + 1)
        return loss


class OHEM_CELoss(nn.Module):

    def __init__(self, thresh=0.7, ignore_index=255):
        super(OHEM_CELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_index = ignore_index
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_index].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


class UnetFormerLoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
        self.aux_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)

    def forward(self, logits, labels):
        if self.training and len(logits) == 2:
            logit_main, logit_aux = logits
            loss = self.main_loss(logit_main, labels) + 0.4 * self.aux_loss(logit_aux, labels)
        else:
            loss = self.main_loss(logits, labels)

        return loss


class AnnealingSoftmaxFocalLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, gamma: float = 2.0, normalize: bool = False):
        super().__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.normalize = normalize
        self.eps = 1e-7

    def forward(self, y_pred, y_true, t: float, t_max: float):
        """
        Args:
            y_pred: [N, #class, H, W]
            y_true: [N, H, W] from 0 to #class
        Returns:
            loss: scalar
        """
        losses = F.cross_entropy(y_pred, y_true, ignore_index=self.ignore_index, reduction='none')

        with torch.no_grad():
            p = y_pred.softmax(dim=1)
            modulating_factor = (1 - p).pow(self.gamma)
            valid_mask = ~y_true.eq(self.ignore_index)
            masked_y_true = torch.where(valid_mask, y_true, torch.zeros_like(y_true))
            modulating_factor = torch.gather(modulating_factor, dim=1, index=masked_y_true.unsqueeze(dim=1)).squeeze_(
                dim=1)
            scale = 1.
            if self.normalize:
                scale = losses.sum() / (losses * modulating_factor).sum()
                if t > t_max:
                    scale = scale
                else:
                    scale = self.cosine_annealing(1., scale, t, t_max)

        losses = scale * (losses * modulating_factor).sum() / (valid_mask.sum() + self.eps)
        return losses

    def cosine_annealing(self, lower_bound, upper_bound, t, t_max):
        return upper_bound + 0.5 * (lower_bound - upper_bound) * (math.cos(math.pi * t / t_max) + 1)


class PixelContrastCrossEntropyLoss(nn.Module):
    """
    The PixelContrastCrossEntropyLoss implementation based on torchtorch.

    The original article refers to
    Wenguan Wang, Tianfei Zhou, et al. "Exploring Cross-Image Pixel Contrast for Semantic Segmentation"
    (https://arxiv.org/abs/2101.11939).

    Args:
        temperature (float, optional): Controling the numerical similarity of features. Default: 0.1.
        base_temperature (float, optional): Controling the numerical range of contrast loss. Default: 0.07.
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
        max_samples (str, optional): Max sampling anchors. Default: 1024.
        max_views (int): Sampled samplers of a class. Default: 100.
    """

    def __init__(self, temperature=0.1, base_temperature=0.07, ignore_index=255, max_samples=1024, max_views=100):
        super(PixelContrastCrossEntropyLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.ignore_index = ignore_index
        self.max_samples = max_samples
        self.max_views = max_views

    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_index]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    raise UserWarning("error happened, please check your code!")

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None):
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        loss = self._contrastive(feats_, labels_)
        return loss


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class BCL(nn.Module):
    """
    batch-balanced contrastive loss
    no-change，1
    change，-1
    """

    def __init__(self, margin=2.0):
        super(BCL, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        y = label.clone()
        y[label > 0] = 1
        mask = (label == 0).float()
        distance = distance * mask
        pos_num = torch.sum((y == 1).float()) + 0.0001
        neg_num = torch.sum((y == -1).float()) + 0.0001

        loss_1 = torch.sum((1 + y) / 2 * torch.pow(distance, 2)) / pos_num
        loss_2 = torch.sum((1 - y) / 2 * mask *
                           torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
                           ) / neg_num
        loss = loss_1 + loss_2
        return loss


if __name__ == '__main__':
    targets = torch.randint(low=0, high=7, size=(4, 16, 16))
    # logits = torch.randn((4, 7, 128, 128))
    # logits = torch.argmax(logits, dim=1)
    feat1 = torch.randn((4, 7, 16, 16))
    feat2 = torch.randn((4, 7, 16, 16))
    # print(targets)
    # model = EdgeLoss()
    # loss = model.compute_edge_loss(logits, targets)
    model = ContrastiveLoss()
    loss = model(feat1, feat1, targets)
    print(loss)
