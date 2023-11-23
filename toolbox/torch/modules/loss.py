#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable


class ClassBalancedLoss(_Loss):
    """
    https://arxiv.org/abs/1901.05555
    """
    @staticmethod
    def demo1():
        batch_loss: torch.FloatTensor = torch.randn(size=(2, 1), dtype=torch.float32)
        targets: torch.LongTensor = torch.tensor([1, 2], dtype=torch.long)

        class_balanced_loss = ClassBalancedLoss(
            num_classes=3,
            num_samples_each_class=[300, 433, 50],
            reduction='mean',
        )
        loss = class_balanced_loss.forward(batch_loss=batch_loss, targets=targets)
        print(loss)
        return

    @staticmethod
    def demo2():
        inputs: torch.FloatTensor = torch.randn(size=(2, 3), dtype=torch.float32)
        targets: torch.LongTensor = torch.tensor([1, 2], dtype=torch.long)

        focal_loss = FocalLoss(
            num_classes=3,
            # reduction='mean',
            # reduction='sum',
            reduction='none',
        )
        batch_loss = focal_loss.forward(inputs, targets)
        print(batch_loss)

        class_balanced_loss = ClassBalancedLoss(
            num_classes=3,
            num_samples_each_class=[300, 433, 50],
            reduction='mean',
        )
        loss = class_balanced_loss.forward(batch_loss=batch_loss, targets=targets)
        print(loss)

        return

    def __init__(self,
                 num_classes: int,
                 num_samples_each_class: List[int],
                 beta: float = 0.999,
                 reduction: str = 'mean') -> None:
        super(ClassBalancedLoss, self).__init__(None, None, reduction)

        effective_num = 1.0 - np.power(beta, num_samples_each_class)
        weights = (1.0 - beta) / np.array(effective_num)
        self.weights = weights / np.sum(weights) * num_classes

    def forward(self, batch_loss: torch.FloatTensor, targets: torch.LongTensor):
        """
        :param batch_loss: shape=[batch_size, 1]
        :param targets: shape=[batch_size,]
        :return:
        """
        weights = list()
        targets = targets.numpy()
        for target in targets:
            weights.append([self.weights[target]])

        weights = torch.tensor(weights, dtype=torch.float32)
        batch_loss = weights * batch_loss

        if self.reduction == 'mean':
            loss = batch_loss.mean()
        elif self.reduction == 'sum':
            loss = batch_loss.sum()
        else:
            loss = batch_loss
        return loss


class EqualizationLoss(_Loss):
    """
    在图像识别中的, sigmoid 的多标签分类, 且 num_classes 类别数之外有一个 background 背景类别.
    Equalization Loss
    https://arxiv.org/abs/2003.05176
    Equalization Loss v2
    https://arxiv.org/abs/2012.08548
    """

    @staticmethod
    def demo1():
        logits: torch.FloatTensor = torch.randn(size=(3, 3), dtype=torch.float32)
        targets: torch.LongTensor = torch.tensor([1, 2, 3], dtype=torch.long)

        equalization_loss = EqualizationLoss(
            num_samples_each_class=[300, 433, 50],
            threshold=100,
            reduction='mean',
        )
        loss = equalization_loss.forward(logits=logits, targets=targets)
        print(loss)
        return

    def __init__(self,
                 num_samples_each_class: List[int],
                 threshold: int = 100,
                 reduction: str = 'mean') -> None:
        super(EqualizationLoss, self).__init__(None, None, reduction)
        self.num_samples_each_class = np.array(num_samples_each_class, dtype=np.int32)
        self.threshold = threshold

    def forward(self,
                logits: torch.FloatTensor,
                targets: torch.LongTensor
                ):
        """
        num_classes + 1 对应于背景类别 background.
        :param logits: shape=[batch_size, num_classes]
        :param targets: shape=[batch_size]
        :return:
        """
        batch_size, num_classes = logits.size()

        one_hot_targets = F.one_hot(targets, num_classes=num_classes + 1)
        one_hot_targets = one_hot_targets[:, :-1]

        exclude = self.exclude_func(
            num_classes=num_classes,
            targets=targets
        )
        is_tail = self.threshold_func(
            num_classes=num_classes,
            num_samples_each_class=self.num_samples_each_class,
            threshold=self.threshold,
        )

        weights = 1 - exclude * is_tail * (1 - one_hot_targets)

        batch_loss = F.binary_cross_entropy_with_logits(
            logits,
            one_hot_targets.float(),
            reduction='none'
        )

        batch_loss = weights * batch_loss

        if self.reduction == 'mean':
            loss = batch_loss.mean()
        elif self.reduction == 'sum':
            loss = batch_loss.sum()
        else:
            loss = batch_loss

        loss = loss / num_classes
        return loss

    @staticmethod
    def exclude_func(num_classes: int, targets: torch.LongTensor):
        """
        最后一个类别是背景 background.
        :param num_classes: int,
        :param targets: shape=[batch_size,]
        :return: weight, shape=[batch_size, num_classes]
        """
        batch_size = targets.shape[0]
        weight = (targets != num_classes).float()
        weight = weight.view(batch_size, 1).expand(batch_size, num_classes)
        return weight

    @staticmethod
    def threshold_func(num_classes: int, num_samples_each_class: np.ndarray, threshold: int):
        """
        :param num_classes: int,
        :param num_samples_each_class: shape=[num_classes]
        :param threshold: int,
        :return: weight, shape=[1, num_classes]
        """
        weight = torch.zeros(size=(num_classes,))
        weight[num_samples_each_class < threshold] = 1
        weight = torch.unsqueeze(weight, dim=0)
        return weight


class FocalLoss(_Loss):
    """
    https://arxiv.org/abs/1708.02002
    """
    @staticmethod
    def demo1(self):
        inputs: torch.FloatTensor = torch.randn(size=(2, 3), dtype=torch.float32)
        targets: torch.LongTensor = torch.tensor([1, 2], dtype=torch.long)

        focal_loss = FocalLoss(
            num_classes=3,
            reduction='mean',
            # reduction='sum',
            # reduction='none',
        )
        loss = focal_loss.forward(inputs, targets)
        print(loss)
        return

    def __init__(self,
                 num_classes: int,
                 alpha: List[float] = None,
                 gamma: int = 2,
                 reduction: str = 'mean',
                 inputs_logits: bool = True) -> None:
        """
        :param num_classes:
        :param alpha:
        :param gamma:
        :param reduction: (`none`, `mean`, `sum`) available.
        :param inputs_logits: if False, the inputs should be probs.
        """
        super(FocalLoss, self).__init__(None, None, reduction)
        if alpha is None:
            self.alpha = torch.ones(num_classes, 1)
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.gamma = gamma
        self.num_classes = num_classes
        self.inputs_logits = inputs_logits

    def forward(self,
                inputs: torch.FloatTensor,
                targets: torch.LongTensor):
        """
        :param inputs: logits, shape=[batch_size, num_classes]
        :param targets: shape=[batch_size,]
        :return:
        """
        batch_size, num_classes = inputs.shape

        if self.inputs_logits:
            probs = F.softmax(inputs, dim=-1)
        else:
            probs = inputs

        class_mask = inputs.data.new(batch_size, num_classes).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (probs*class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

        if self.reduction == 'mean':
            loss = batch_loss.mean()
        elif self.reduction == 'sum':
            loss = batch_loss.sum()
        else:
            loss = batch_loss
        return loss


class HingeLoss(_Loss):
    @staticmethod
    def demo1():
        inputs: torch.FloatTensor = torch.randn(size=(2, 3), dtype=torch.float32)
        targets: torch.LongTensor = torch.tensor([1, 2], dtype=torch.long)

        hinge_loss = HingeLoss(
            margin_list=[300, 433, 50],
            reduction='mean',
        )
        loss = hinge_loss.forward(inputs=inputs, targets=targets)
        print(loss)
        return

    def __init__(self,
                 margin_list: List[float],
                 max_margin: float = 0.5,
                 scale: float = 30.0,
                 weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean') -> None:
        super(HingeLoss, self).__init__(None, None, reduction)

        self.max_margin = max_margin
        self.scale = scale
        self.weight = weight

        margin_list = np.array(margin_list)
        margin_list = margin_list * (max_margin / np.max(margin_list))
        self.margin_list = torch.tensor(margin_list, dtype=torch.float32)

    def forward(self,
                inputs: torch.FloatTensor,
                targets: torch.LongTensor
                ):
        """
        :param inputs: logits, shape=[batch_size, num_classes]
        :param targets: shape=[batch_size,]
        :return:
        """
        batch_size, num_classes = inputs.shape
        one_hot_targets = F.one_hot(targets, num_classes=num_classes)
        margin_list = torch.unsqueeze(self.margin_list, dim=0)

        batch_margin = torch.sum(margin_list * one_hot_targets, dim=-1)
        batch_margin = torch.unsqueeze(batch_margin, dim=-1)
        inputs_margin = inputs - batch_margin

        # 将类别对应的 logits 值减小一点, 以形成 margin 边界.
        logits = torch.where(one_hot_targets > 0, inputs_margin, inputs)

        loss = F.cross_entropy(
            input=self.scale * logits,
            target=targets,
            weight=self.weight,
            reduction=self.reduction,
        )
        return loss


class LDAMLoss(_Loss):
    """
    https://arxiv.org/abs/1906.07413
    """
    @staticmethod
    def demo1():
        inputs: torch.FloatTensor = torch.randn(size=(2, 3), dtype=torch.float32)
        targets: torch.LongTensor = torch.tensor([1, 2], dtype=torch.long)

        ldam_loss = LDAMLoss(
            num_samples_each_class=[300, 433, 50],
            reduction='mean',
        )
        loss = ldam_loss.forward(inputs=inputs, targets=targets)
        print(loss)
        return

    def __init__(self,
                 num_samples_each_class: List[int],
                 max_margin: float = 0.5,
                 scale: float = 30.0,
                 weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean') -> None:
        super(LDAMLoss, self).__init__(None, None, reduction)

        margin_list = np.power(num_samples_each_class, -0.25)
        margin_list = margin_list * (max_margin / np.max(margin_list))

        self.num_samples_each_class = num_samples_each_class
        self.margin_list = torch.tensor(margin_list, dtype=torch.float32)
        self.scale = scale
        self.weight = weight

    def forward(self,
                inputs: torch.FloatTensor,
                targets: torch.LongTensor
                ):
        """
        :param inputs: logits, shape=[batch_size, num_classes]
        :param targets: shape=[batch_size,]
        :return:
        """
        batch_size, num_classes = inputs.shape
        one_hot_targets = F.one_hot(targets, num_classes=num_classes)
        margin_list = torch.unsqueeze(self.margin_list, dim=0)

        batch_margin = torch.sum(margin_list * one_hot_targets, dim=-1)
        batch_margin = torch.unsqueeze(batch_margin, dim=-1)
        inputs_margin = inputs - batch_margin

        # 将类别对应的 logits 值减小一点, 以形成 margin 边界.
        logits = torch.where(one_hot_targets > 0, inputs_margin, inputs)

        loss = F.cross_entropy(
            input=self.scale * logits,
            target=targets,
            weight=self.weight,
            reduction=self.reduction,
        )
        return loss


def demo1():
    HingeLoss.demo1()
    return


if __name__ == '__main__':
    demo1()
