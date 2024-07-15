import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional

class Cls(nn.Module):

    def __init__(
            self,
            num_classes: int = 1000,
    ) -> None:
        super(Cls, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, num_classes)
    
    def _forward_impl(self, x: Tensor) -> Tensor:
        pooled_f = self.avgpool(x)
        pooled_f = torch.flatten(pooled_f, 1)
        cls = self.fc(pooled_f)
        return cls

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

class Cls2(nn.Module):

    def __init__(
            self,
            num_classes: int = 1000,
    ) -> None:
        super(Cls2, self).__init__()
        self.avgpool = nn.AvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, num_classes)
    
    def _forward_impl(self, x: Tensor) -> Tensor:
        pooled_f = self.avgpool(x)
        pooled_f = torch.flatten(pooled_f, 1)
        cls = self.fc(pooled_f)
        return cls

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def classification_loss(num_cls):
    if num_cls > 2:
        return nn.CrossEntropyLoss(), None
    else:
        return nn.BCELoss(), nn.Sigmoid()
