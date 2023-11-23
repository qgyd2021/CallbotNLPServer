#!/usr/bin/python3
# -*- coding: utf-8 -*-
import math
from typing import List, Tuple, Union

from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.nn import Activation
from overrides import overrides
import torch
import torch.nn as nn


class Conv1dBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ngram_filter_sizes: Union[int, List[int]],
                 stride: Tuple[int, int],
                 padding: str = 'same',
                 batch_norm: bool = False,
                 activation: str = None,
                 dropout: float = None,
                 projection_dim: int = None,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ngram_filter_sizes = ngram_filter_sizes

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(in_channels)
        else:
            self.batch_norm = None

        if isinstance(ngram_filter_sizes, int):
            ngram_filter_sizes = [ngram_filter_sizes]

        self.conv_list = nn.ModuleList(modules=[
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size,),
                stride=stride,
                padding=padding,
            ) for kernel_size in ngram_filter_sizes
        ])

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

    def forward(self, x):
        x = torch.transpose(x, dim0=-1, dim1=-2)

        if self.batch_norm is not None:
            x = self.batch_norm(x)

        x_list = list()
        for conv in self.conv_list:
            x_i = conv(x)
            x_list.append(x_i)
        x = torch.concat(x_list, dim=-2)

        if self.activation is not None:
            x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x)
        x = torch.transpose(x, dim0=-1, dim1=-2)

        if self.projection_layer is not None:
            x = self.projection_layer(x)
        return x


@Seq2SeqEncoder.register('cnn')
class CnnSeq2SeqEncoder(Seq2SeqEncoder):
    def __init__(self,
                 conv1d_block_list: List[dict],
                 ):
        super().__init__()
        self.conv1d_block_list = nn.ModuleList(modules=[
            Conv1dBlock(**conv1d_block)
            for conv1d_block in conv1d_block_list
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
        for conv1d_block in self.conv1d_block_list:
            x = conv1d_block(x)
        return x


def demo1():
    import numpy as np

    # [batch_size, seq_length, dim]
    x = np.ones(shape=(2, 20, 100))
    x = torch.tensor(x, dtype=torch.float32)

    cnn = CnnSeq2SeqEncoder(
        conv1d_block_list=[
            {
                'batch_norm': True,
                'in_channel': 100,
                'out_channel': 100,
                'kernel_size': [3,],
                'stride': 1,
                'padding': 'same',
                'activation': 'relu',
                'dropout': 0.1,
            },
            {
                'in_channel': 100,
                'out_channel': 100,
                'kernel_size': [3,],
                'stride': 1,
                'padding': 'same',
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
