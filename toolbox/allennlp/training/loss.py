#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Callable, Optional

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss


class FocalLoss(_WeightedLoss):

    __constants__ = ['reduction']

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(FocalLoss, self).__init__(weight, size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.binary_focal_cross_entropy(input, target, reduction=self.reduction)

    @staticmethod
    def binary_focal_cross_entropy(inputs, targets, reduction='mean'):
        alpha = 0.25
        gamma = 2

        cond1 = alpha * torch.pow((1 - inputs), gamma) * torch.log(1 / inputs)
        cond2 = (1 - alpha) * torch.pow(inputs, gamma) * torch.log(1 / (1 - inputs))

        focal = targets * cond1 + (1.0 - targets) * cond2

        if reduction == 'sum':
            focal = torch.sum(focal)
        elif reduction == 'mean':
            if focal.ndim > 1:
                focal = torch.sum(focal, dim=-1)
            focal = torch.mean(focal)

        else:
            if focal.ndim > 1:
                focal = torch.sum(focal, dim=-1)
            focal = torch.mean(focal)

        return focal


if __name__ == '__main__':
    pass
