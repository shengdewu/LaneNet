import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from engine.loss import LOSS_ARCH_REGISTRY

from ..model.utils import differentiable_argmax

__all__ = [
    'SoftmaxFocalLoss',
    'SimilarityLoss',
    'StraightLoss',
    'SegStraightLoss'
]


@LOSS_ARCH_REGISTRY.register()
class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, lambda_weight=1.0, ignore_index=255):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_index)
        self.lambda_weight = lambda_weight
        return

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1. - scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss * self.lambda_weight


@LOSS_ARCH_REGISTRY.register()
class SimilarityLoss(nn.Module):
    def __init__(self, lambda_weight=1.0):
        super(SimilarityLoss, self).__init__()
        self.lambda_weight = lambda_weight
        return

    def forward(self, logits):
        n, c, h, w = logits.shape
        loss_all = []
        for i in range(0, h - 1):
            loss_all.append(logits[:, :, i, :] - logits[:, :, i + 1, :])
        # loss0 : n,c,w
        loss = torch.cat(loss_all)
        return torch.nn.functional.smooth_l1_loss(loss, torch.zeros_like(loss)) * self.lambda_weight


@LOSS_ARCH_REGISTRY.register()
class StraightLoss(nn.Module):
    def __init__(self, lambda_weight=1.0, use_all_pts=False):
        super(StraightLoss, self).__init__()
        self.l1 = torch.nn.L1Loss()
        # self.l1 = torch.nn.MSELoss()
        self.lambda_weight = lambda_weight
        self.use_all_pts = use_all_pts
        return

    def forward(self, logits):
        n, dim, num_rows, num_cols = logits.shape
        x = torch.nn.functional.softmax(logits[:, :dim - 1, :, :], dim=1)
        embedding = torch.Tensor(np.arange(dim - 1)).float().to(x.device).view(1, -1, 1, 1)
        pos = torch.sum(x * embedding, dim=1)

        diff_list1 = []
        end_loc = num_rows // 2  # 原始论文只关注了前半部分
        if self.use_all_pts:
            end_loc = num_rows - 1
        for i in range(0, end_loc):
            diff_list1.append(pos[:, i, :] - pos[:, i + 1, :])

        loss = 0
        for i in range(len(diff_list1) - 1):
            loss += self.l1(diff_list1[i], diff_list1[i + 1])
        loss /= len(diff_list1) - 1
        return loss * self.lambda_weight


@LOSS_ARCH_REGISTRY.register()
class SegStraightLoss(nn.Module):
    def __init__(self, lambda_weight=1.0):
        super(SegStraightLoss, self).__init__()
        self.l1 = torch.nn.L1Loss(reduction='mean')
        # self.l1 = torch.nn.MSELoss()
        self.lambda_weight = lambda_weight
        return

    def forward(self, logits, labels):
        bs, nc, h, w = logits.shape
        seg_logits = differentiable_argmax(logits, dim=1).to(torch.int64)

        loss = 0
        for b in range(bs):
            for c in range(1, nc):
                coord = torch.column_stack(torch.where(seg_logits[b] == c))
                if coord.shape[0] == 0:
                    continue
                coord = coord.to(torch.float)
                diff = coord[:, 1:] - coord[:, :-1]
                loss += self.l1(diff[1:], diff[:-1]) / (diff.shape[0] + 1)

        return loss * self.lambda_weight
