import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Any

import torch.nn.functional as F
from engine.model import BUILD_NETWORK_REGISTRY

from ..utils import ConvModule, accuracy
from .base_head import BaseHead

__all__ = [
    'LaneHead',
    'LaneHeadAux'
]


class SpatialPyramidPooling(nn.Module):
    def __init__(self, levels=(1, 2, 4)):
        super(SpatialPyramidPooling, self).__init__()
        self.levels = levels

    def forward(self, x):
        num_samples, num_channels, height, width = x.size()
        all_features = []
        for level in self.levels:
            # 计算每个池化区域的大小和步长
            kernel_size = (int(height / level), int(width / level))
            stride = kernel_size
            # 进行最大池化操作
            pooled = nn.functional.max_pool2d(x, kernel_size=kernel_size, stride=stride)
            # 将池化后的特征展平
            flattened = pooled.view(num_samples, -1)
            all_features.append(flattened)

        # 拼接不同层级的池化特征
        output = torch.cat(all_features, dim=1)
        return output


@BUILD_NETWORK_REGISTRY.register()
class LaneHead(BaseHead):

    def __init__(self,
                 in_channels: List,
                 grid_num: int,
                 cls_num_per_lane: int,
                 num_lanes: int,
                 cls_channel: int = 2048,
                 pool_channel: int = 8,
                 spp_levels: Tuple = (1, 2, 4, 8),
                 loss_cfg: Optional[List] = None):
        super(LaneHead, self).__init__(loss_cfg)
        """
        in_size: h, w
        """
        self.grid_num = grid_num  # 不包括背景
        self.grid_cls = grid_num + 1
        self.num_lanes = num_lanes
        self.cls_num_per_lane = cls_num_per_lane
        self.cls_channel = cls_channel

        self.pool = nn.Sequential(
            ConvModule(in_channels[-1], pool_channel, 1, 0, bias=False, act_cfg=None, norm_cfg=None),
            SpatialPyramidPooling(spp_levels)
        )

        self.pool_channel = sum([l * l * pool_channel for l in spp_levels])
        self.cls = nn.Sequential(
            nn.Linear(self.pool_channel, self.cls_channel),
            nn.ReLU(),
            nn.Linear(self.cls_channel, self.grid_cls * self.num_lanes * self.cls_num_per_lane)
        )

        self.init_weights(self.cls)
        self.cls_group = (self.grid_cls, self.cls_num_per_lane, self.num_lanes)
        return

    def init_weights(self, models):
        for model in models:
            if isinstance(model, list):
                for m in model:
                    self.init_weights(m)
            else:
                if isinstance(model, torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(model.weight, nonlinearity='relu')
                    if model.bias is not None:
                        torch.nn.init.constant_(model.bias, 0)
                elif isinstance(model, torch.nn.Linear):
                    model.weight.data.normal_(0.0, std=0.01)
                elif isinstance(model, torch.nn.BatchNorm2d):
                    torch.nn.init.constant_(model.weight, 1)
                    torch.nn.init.constant_(model.bias, 0)
                elif isinstance(model, torch.nn.Module):
                    for m in model.children():
                        self.init_weights(m)
                else:
                    print('unkonwn module', model)

    def forward(self, x_stages, shape=None):
        """
        x_stages： 特征图有大到小
        """
        feat = self.pool(x_stages[-1])
        out = self.cls(feat)
        out = out.view(-1, *self.cls_group)

        if not self.training:
            out = torch.argmax(out, dim=1)

        return out

    def forward_train(self, x_stages, shape, gt_group_cls) -> torch.Tensor:
        x = self(x_stages, shape)

        loss_cfg = [
            (x, gt_group_cls),
            dict(logits=x),
            dict(logits=x),
        ]
        loss = self.loss_fn(dict(loss=loss_cfg))
        return loss


@BUILD_NETWORK_REGISTRY.register()
class LaneHeadAux(BaseHead):

    def __init__(self,
                 in_channels: List,
                 num_lanes: int,
                 aux_channel: Optional[int] = None,
                 loss_cfg: Optional[List] = None):
        super(LaneHeadAux, self).__init__(loss_cfg)
        self.aux_channel = aux_channel
        self.num_lanes = num_lanes
        self.model_list = nn.ModuleList()

        layers = len(in_channels)
        for i, in_channel in enumerate(in_channels):

            models = [
                ConvModule(in_channel, self.aux_channel, 3, 1, 1, bias=False, norm_cfg=dict(type="BatchNorm2d"))
            ]

            for _ in range(i, layers):
                models.append(ConvModule(self.aux_channel, self.aux_channel, 3, 1, 1, bias=False,
                                         norm_cfg=dict(type="BatchNorm2d")))

            self.model_list.append(nn.Sequential(*models))

        self.aux_combine = nn.Sequential(
            ConvModule(self.aux_channel * layers, self.aux_channel, 3, 2, 1, 2, bias=False,
                       norm_cfg=dict(type="BatchNorm2d")),
            ConvModule(self.aux_channel, self.aux_channel, 3, 2, 1, 2, bias=False,
                       norm_cfg=dict(type="BatchNorm2d")),
            ConvModule(self.aux_channel, self.aux_channel, 3, 2, 1, 2, bias=False,
                       norm_cfg=dict(type="BatchNorm2d")),
            ConvModule(self.aux_channel, self.aux_channel, 3, 4, 1, 4, bias=False,
                       norm_cfg=dict(type="BatchNorm2d")),
            ConvModule(self.aux_channel, self.num_lanes + 1, 1, 0, bias=False, act_cfg=None, norm_cfg=None),
        )

        return

    def forward(self, x_stages, shape=None):
        """
        x_stages： 特征图有大到小
        """

        assert len(self.model_list) == len(x_stages)

        aux_header = self.model_list[0](x_stages[0])
        aux_shape = aux_header.shape[2:]
        aux_headers = [aux_header]

        for i, model in enumerate(self.model_list[1:]):
            aux_header = model(x_stages[i + 1])
            aux_header = torch.nn.functional.interpolate(aux_header, size=aux_shape, mode='bilinear')
            aux_headers.append(aux_header)

        out = self.aux_combine(torch.cat(aux_headers, dim=1))
        out = F.interpolate(out, shape, mode='nearest')

        return out
