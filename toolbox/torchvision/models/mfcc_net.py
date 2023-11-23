#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F


class MfccNet(nn.Module):
    def __init__(self, dropout=0.1, out_features=64):
        super().__init__()
        self.dropout = dropout
        self.out_features = out_features
        self.batch_norm = nn.BatchNorm2d(1, affine=False)

        self.conv1 = nn.Conv2d(1, 4, kernel_size=(11, 11), stride=1, padding=(0,))
        self.conv2 = nn.Conv2d(4, self.out_features, kernel_size=(7, 20), stride=1, padding=(0,))

    def forward(self, x):
        x = self.batch_norm(x)

        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # x.shape = [64, 4, 7, 20]
        x = self.conv2(x)
        x = F.relu(x)
        return x

    def get_input_dim(self):
        """[channel, height, width]"""
        result = (1, 25, 50)
        return result

    def get_output_dim(self):
        """[channel, height, width]"""
        result = (self.out_features, 1, 1)
        return result


if __name__ == '__main__':
    pass
