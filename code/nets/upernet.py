# -*- coding: utf-8 -*-
# @Time    : 2024/4/1 13:46
# @Author  : NightSongs
# @File    : upernet.py
# @Software: PyCharm
import torch
import timm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

nonlinearity = partial(F.relu, inplace=True)


def remap_checkpoint_convnext(state_dict):
    """ Remap FB checkpoints -> timm """
    if 'head.norm.weight' in state_dict or 'norm_pre.weight' in state_dict:
        return state_dict  # non-FB checkpoint
    if 'model' in state_dict:
        state_dict = state_dict['model']

    if 'visual.trunk.stem.0.weight' in state_dict:
        out_dict = {k.replace('visual.trunk.', 'model.'): v for k, v in state_dict.items() if
                    k.startswith('visual.trunk.')}

        out_dict = {k.replace('stages.', 'stages_'): v for k, v in out_dict.items()}
        out_dict = {k.replace('stem.', 'stem_'): v for k, v in out_dict.items()}
        out_dict.pop('model.head.norm.weight')
        out_dict.pop('model.head.norm.bias')
        return out_dict


def load_checkpoint(model_name, model, path, strict=False):
    # for timm 0.6.12
    state_dict = torch.load(path)
    if model_name == 'convnext-clip':
        new_state_dict = remap_checkpoint_convnext(state_dict)
        final_dict = {}
        for k, v in new_state_dict.items():
            if k.startswith('model.'):
                final_dict[k[6:]] = v
            else:
                final_dict[k] = v
        model.load_state_dict(final_dict, strict=strict)
    else:
        raise NotImplementedError
    print(f"load from {path}")
    return model


def get_backbone(backbone_name, pretrained=True, in_channels=3, out_indices=[1, 2, 3, 4]):
    if backbone_name == 'convnext_large_clip':
        backbone = timm.create_model('convnext_large', features_only=True, pretrained=False,
                                     in_chans=in_channels, out_indices=None)
    elif backbone_name == 'convnext_base_clip':
        backbone = timm.create_model('convnext_base', features_only=True, pretrained=False,
                                     in_chans=in_channels, out_indices=None)
    else:
        backbone = timm.create_model(backbone_name, features_only=True, pretrained=pretrained,
                                     in_chans=in_channels, out_indices=out_indices)
    if 'large_clip' in backbone_name and pretrained:
        backbone = load_checkpoint('convnext-clip',
                                   backbone,
                                   '/mnt/d/competition/ISPRS2024/pretrain/open_clip_pytorch_model_convnext-l.bin',
                                   strict=True)
    elif 'base_clip' in backbone_name and pretrained:
        backbone = load_checkpoint('convnext-clip',
                                   backbone,
                                   '/mnt/d/competition/ISPRS2024/pretrain/open_clip_pytorch_model_convnext-b.bin',
                                   strict=True)
    return backbone, backbone.feature_info.channels()


class AdaptiveAvgPool2dCustom(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2dCustom, self).__init__()
        self.output_size = np.array(output_size)

    def forward(self, x: torch.Tensor):
        '''
        Args:
            x: shape (batch size, channel, height, width)
        Returns:
            x: shape (batch size, channel, 1, output_size)
        '''
        shape_x = x.shape
        if (shape_x[-1] < self.output_size[-1]):
            paddzero = torch.zeros((shape_x[0], shape_x[1], shape_x[2], self.output_size[-1] - shape_x[-1]))
            paddzero = paddzero.to('cuda:0')
            x = torch.cat((x, paddzero), axis=-1)

        stride_size = np.floor(np.array(x.shape[-2:]) / self.output_size).astype(np.int32)
        kernel_size = np.array(x.shape[-2:]) - (self.output_size - 1) * stride_size
        avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
        x = avg(x)
        return x


class sSE(nn.Module):
    def __init__(self, out_channels):
        super(sSE, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=1, kernel_size=1, padding=0),
            nn.BatchNorm2d(1))

    def forward(self, x):
        x = self.conv(x)
        x = F.sigmoid(x)
        return x


class cSE(nn.Module):
    def __init__(self, out_channels, act_fun=nonlinearity):
        super(cSE, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=int(out_channels / 2), kernel_size=1, padding=0),
            nn.BatchNorm2d(int(out_channels / 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=int(out_channels / 2), out_channels=out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.act_fun = act_fun

    def forward(self, x):
        x = nn.AvgPool2d(x.size()[2:])(x)
        x = self.conv1(x)
        x = self.act_fun(x)
        x = self.conv2(x)
        x = F.sigmoid(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1, align_corners=True):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.Upsample(scale_factor=upsampling, mode='bilinear',
                                 align_corners=align_corners) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class PSPBlock(nn.Module):

    def __init__(self, in_channels, out_channels, pool_size, use_bathcnorm=True):
        super().__init__()
        if pool_size == 1:
            use_bathcnorm = False
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
            # AdaptiveAvgPool2dCustom(output_size=(pool_size, pool_size)),  # onnx不支持AdaptiveAvgPool2d, 需要重写
            Conv2dReLU(in_channels, out_channels, (1, 1), use_batchnorm=use_bathcnorm)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.pool(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x


class PSPModule(nn.Module):
    def __init__(self, in_channels, out_channels, sizes=(1, 2, 3, 6), use_bathcnorm=True):
        super().__init__()

        self.blocks = nn.ModuleList([
            PSPBlock(in_channels, out_channels, size, use_bathcnorm=use_bathcnorm) for size in sizes
        ])

    def forward(self, x):
        xs = [block(x) for block in self.blocks] + [x]
        x = torch.cat(xs, dim=1)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels, drop_path_rate=0.0, scse=False):
        super().__init__()
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(pyramid_channels),
            nn.ReLU(inplace=True)
        )
        self.drop_path = nn.Identity()
        if drop_path_rate > 0:
            self.drop_path = DropPath(drop_path_rate)

        self.scse = scse
        if self.scse:
            self.spatial_gate = sSE(pyramid_channels)
            self.channel_gate = cSE(pyramid_channels, act_fun=nn.ReLU(inplace=True))

    def forward(self, x, skip=None):
        # x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        skip = self.skip_conv(skip)
        if self.scse:
            g1 = self.spatial_gate(skip)
            g2 = self.channel_gate(skip)
            skip = g1 * skip + g2 * skip
        x = self.drop_path(x) + skip
        return x


class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(
                    policy
                )
            )
        self.policy = policy

    def forward(self, x):
        if self.policy == 'add':
            return sum(x)
        elif self.policy == 'cat':
            return torch.cat(x, dim=1)
        else:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy)
            )


class Fusion(nn.Module):
    def __init__(self, feature_channels, policy):
        super().__init__()
        if policy not in ["add", "cat", "conv", "diff", "abs_diff"]:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(
                    policy
                )
            )
        self.feature_channels = feature_channels
        self.policy = policy
        if self.policy == 'conv':
            self.fusion_conv = nn.ModuleList()
            for i in range(len(feature_channels)):
                self.fusion_conv.append(nn.Sequential(
                    nn.Conv2d(feature_channels[i] * 2, feature_channels[i], 3, 1, 1),
                    nn.InstanceNorm2d(feature_channels[i]),
                    nn.GELU(),
                ))

    def forward(self, x1_list, x2_list):
        out = []
        if self.policy == 'add':
            for i in range(len(x1_list)):
                out.append(x1_list[i] + x2_list[i])
            return out
        elif self.policy == 'diff':
            for i in range(len(x1_list)):
                out.append(x1_list[i] - x2_list[i])
            return out
        elif self.policy == 'abs_diff':
            for i in range(len(x1_list)):
                out.append(torch.abs(x1_list[i] - x2_list[i]))
            return out
        elif self.policy == 'cat':
            for i in range(len(x1_list)):
                out.append(torch.cat([x1_list[i], x2_list[i]], dim=1))
            return out
        elif self.policy == 'conv':
            for i in range(len(x1_list)):
                out.append(self.fusion_conv[i](torch.cat([x1_list[i], x2_list[i]], dim=1)))
            return out
        else:
            raise ValueError(
                "`fusion_policy` must be one of: ['add', 'cat', 'conv', 'diff', 'abs_diff'], got {}".format(self.policy)
            )


class UPerNet(nn.Module):
    def __init__(
            self,
            backbone_name,
            psp_channels=512,
            psp_size=(1, 2, 3, 6),
            pyramid_channels=256,
            segmentation_channels=256,
            num_classes=7,
            dropout=0.,
            drop_path_rate=0.,
            merge_policy="add",
            fusion_form="cat",
            scse=False,
            pretrained=False
    ):
        super().__init__()

        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4

        self.backbone, encoder_channels = get_backbone(backbone_name, out_indices=[1, 2, 3, 4], pretrained=pretrained)
        self.fusion = Fusion(encoder_channels, fusion_form)

        if fusion_form == "cat":
            encoder_channels = [ch * 2 for ch in encoder_channels]

        self.psp = PSPModule(
            in_channels=encoder_channels[3],
            out_channels=psp_channels,
            sizes=psp_size,
            use_bathcnorm=True,
        )

        self.psp_last_conv = Conv2dReLU(
            in_channels=psp_channels * len((1, 2, 3, 6)) + encoder_channels[3],
            out_channels=pyramid_channels,
            kernel_size=1,
            use_batchnorm=True,
        )

        self.p4 = FPNBlock(pyramid_channels, encoder_channels[2], drop_path_rate, scse)
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[1], drop_path_rate, scse)
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[0], drop_path_rate, scse)

        self.merge = MergeBlock(merge_policy)

        self.conv_last = Conv2dReLU(self.out_channels, pyramid_channels, 1)

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        self.segmentation_head = SegmentationHead(
            in_channels=self.out_channels,
            out_channels=num_classes,
            kernel_size=1,
            upsampling=4,
            align_corners=False,
        )

    def forward(self, x1, x2):
        features1 = self.backbone(x1)
        features2 = self.backbone(x2)
        features = self.fusion(features1, features2)

        c2, c3, c4, c5 = features

        c5 = self.psp(c5)
        p5 = self.psp_last_conv(c5)

        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)

        output_size = p2.size()[2:]
        feature_pyramid = [nn.functional.interpolate(p, output_size,
                                                     mode='bilinear', align_corners=False) for p in
                           [p5, p4, p3, p2]]
        seg_out = self.merge(feature_pyramid)
        seg_out = self.conv_last(seg_out)
        seg_out = self.dropout(seg_out)
        seg_out = self.segmentation_head(seg_out)
        return seg_out


if __name__ == "__main__":
    x1 = torch.rand(4, 3, 512, 512).float()
    model = UPerNet(backbone_name='convnext_base_clip')
    print(model.backbone)
    seg_out = model(x1, x1)
    print(seg_out.size())
