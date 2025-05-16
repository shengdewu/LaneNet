import abc

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Any

from engine.loss import LossKeyCompose


__all__ = [
    'BaseHead',
]


class BaseHead(nn.Module):
    def __init__(self,
                 loss_cfg: Optional[Dict] = None):
        super(BaseHead, self).__init__()
        self.with_loss = False
        if loss_cfg is not None:
            self.with_loss = True
            self.loss_fn = LossKeyCompose(loss_cfg)
        return

    @abc.abstractmethod
    def forward(self, x_stages, shape):
        raise NotImplemented('the forward of BaseHead2')

    def forward_train(self, x_stages, shape, gt_semantic_seg) -> torch.Tensor:
        x = self(x_stages, shape)
        loss = self.loss_fn(dict(loss=[(x, gt_semantic_seg)]))
        return loss

    def prepare_deploy(self):
        if self.with_loss:
            delattr(self, 'loss_fn')
            self.with_loss = False
        return
