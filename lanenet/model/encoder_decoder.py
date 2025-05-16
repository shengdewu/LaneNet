from engine.model.build import BUILD_NETWORK_REGISTRY
from engine.model.build import build_network
import torch.nn as nn
from torch import Tensor
import torch
from typing import Dict, Optional, Union, List, Tuple
from .utils import acc_topk

__all__ = [
    'EncoderDecoder'
]


@BUILD_NETWORK_REGISTRY.register()
class EncoderDecoder(nn.Module):
    def __init__(self, encoder: Dict,
                 decoder: Dict,
                 necker: Optional[Dict] = None,
                 auxiliary: Optional[Union[List[Dict], Dict]] = None,
                 ignore_index: int = None,
                 out_type=None):

        super(EncoderDecoder, self).__init__()
        self.encoder = build_network(encoder)
        self.decoder = build_network(decoder)
        self.with_neck = False
        if necker is not None:
            self.necker = build_network(necker)
            self.with_neck = True
        self.with_auxiliary = False
        if auxiliary is not None:
            self.with_auxiliary = True
            if isinstance(auxiliary, list):
                self.auxiliary = nn.ModuleList()
                for config in auxiliary:
                    self.auxiliary.append(build_network(config))
            else:
                self.auxiliary = build_network(auxiliary)
        self.ignore_index = ignore_index
        self.out_type = out_type
        return

    @property
    def model_name(self):
        return self.encoder.__class__.__name__ + '-' + self.decoder.__class__.__name__

    def extract_feature(self, img: Tensor):
        x = self.encoder(img)
        if self.with_neck:
            x = self.necker(x)
        return x

    def forward_train(self, img: Tensor, label: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.extract_feature(img)

        loss = self.decoder.forward_train(x, img.shape[2:], label['label'])

        result = dict()
        if self.with_auxiliary:
            if isinstance(self.auxiliary, nn.ModuleList):
                for idx, aux in enumerate(self.auxiliary):
                    loss_aux = aux.forward_train(x, img.shape[2:], label['aux_label'])
                    loss += loss_aux
            else:
                loss_aux = self.auxiliary.forward_train(x, img.shape[2:], label['aux_label'])
                loss += loss_aux

        result['loss'] = loss

        return result

    def forward_test(self, img: Tensor, label: Dict[str, Tensor]) -> Dict:
        x = self.extract_feature(img)
        logits = self.decoder(x, img.shape[2:])

        result = dict()
        result['logits'] = logits

        """
        车道线的坐标，在x轴上被分成101类（0-100），100 是背景
        """
        result['acc'] = acc_topk(torch.argmax(logits, dim=1), label['label'], logits.shape[1]-1)

        if self.with_auxiliary:
            if isinstance(self.auxiliary, nn.ModuleList):
                logits = self.auxiliary[-1](x, img.shape[2:])
            else:
                logits = self.auxiliary(x, img.shape[2:])

            result['aux_logits'] = logits
            result['aux_acc'] = acc_topk(torch.argmax(logits, dim=1), label['aux_label'], 0).item()

        return result

    def forward(self, img: Tensor):
        x = self.extract_feature(img)
        logits = self.decoder(x, img.shape[2:])
        return logits

    def prepare_deploy(self):
        if self.with_auxiliary:
            delattr(self, 'auxiliary')
            self.with_auxiliary = False
        self.decoder.prepare_deploy()
        return
