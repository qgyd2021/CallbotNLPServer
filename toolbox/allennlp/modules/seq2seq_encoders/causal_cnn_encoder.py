#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import List, Tuple, Union

from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.nn import Activation
import torch
import torch.nn as nn


class CausalConv1d(torch.nn.Module):
    """
    因果卷积
    https://zhuanlan.zhihu.com/p/422177151
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 batch_norm: bool = False,
                 activation: str = None,
                 dropout: float = None,
                 projection_dim: int = None,
                 device=None,
                 dtype=None
                 ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise AssertionError('kernel_size should be odd number. ')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.device = device
        self.dtype = dtype
        self.projection_dim = projection_dim or self.out_channels

        self.padding = (self.kernel_size - 1) * self.dilation

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(in_channels)
        else:
            self.batch_norm = None

        self.conv1d = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(self.kernel_size,),
            stride=(self.stride,),
            padding=self.padding,
            dilation=(self.dilation,),
            groups=self.groups,
            bias=self.bias,
            padding_mode=self.padding_mode,
            device=self.device,
            dtype=self.dtype,
        )

        if activation is None:
            self.activation = None
        else:
            self.activation = Activation.by_name(activation)()

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        if projection_dim is not None:
            self.projection_layer = torch.nn.Linear(
                in_features=self.out_channels * len(self.ngram_filter_sizes),
                out_features=self.projection_dim,
            )
        else:
            self.projection_layer = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, dim, seq_length]
        x = torch.transpose(input, dim0=-1, dim1=-2)

        if self.batch_norm is not None:
            x = self.batch_norm(x)

        x = self.conv1d.forward(x)

        if self.activation is not None:
            x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x)
        x = torch.transpose(x, dim0=-1, dim1=-2)

        if self.projection_layer is not None:
            x = self.projection_layer(x)

        if self.padding != 0:
            x = x[:, :-self.padding, :]

        return x


@Seq2SeqEncoder.register('causal_cnn')
class CausalCnnSeq2SeqEncoder(Seq2SeqEncoder):
    """
    因果卷积
    https://zhuanlan.zhihu.com/p/422177151
    """
    def __init__(self,
                 causal_conv1d_block_list: List[dict],
                 ):
        super().__init__()
        self.causal_conv1d_block_list = nn.ModuleList(modules=[
            CausalConv1d(**causal_conv1d_block)
            for causal_conv1d_block in causal_conv1d_block_list
        ])

    def get_output_dim(self) -> int:
        return self.conv1d_block_list[-1]['out_channel']

    def forward(self,
                inputs: torch.Tensor,
                mask: torch.Tensor = None,
                hidden_state: torch.Tensor = None
                ):
        x = inputs
        # x: [batch_size, channel, height, width]
        for causal_conv1d_block in self.causal_conv1d_block_list:
            x = causal_conv1d_block(x)
        return x


def demo1():
    import numpy as np

    # [batch_size, seq_length, dim]
    x = np.ones(shape=(2, 20, 100))
    x = torch.tensor(x, dtype=torch.float32)

    cnn = CausalCnnSeq2SeqEncoder(
        causal_conv1d_block_list=[
            {
                'batch_norm': True,
                'in_channels': 100,
                'out_channels': 100,
                'kernel_size': 3,
                'stride': 1,
                'activation': 'relu',
                'dropout': 0.1,
            },
            {
                'in_channels': 100,
                'out_channels': 100,
                'kernel_size': 3,
                'stride': 1,
                'activation': 'relu',
                'dropout': 0.1,
            },
        ]
    )
    print(cnn)

    x = cnn.forward(x)
    print(x.shape)
    return


if __name__ == '__main__':
    demo1()
