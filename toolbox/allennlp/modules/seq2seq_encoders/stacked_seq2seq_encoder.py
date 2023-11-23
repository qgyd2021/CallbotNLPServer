#!/usr/bin/python3
# -*- coding: utf-8 -*-
import math
from typing import List, Tuple

from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.nn import Activation
from overrides import overrides
import torch
import torch.nn as nn


@Seq2SeqEncoder.register('stack')
class StackedSeq2SeqEncoder(Seq2SeqEncoder):
    def __init__(self,
                 seq2seq_encoder_list: List[Seq2SeqEncoder],
                 ):
        super().__init__()
        self.seq2seq_encoder_list = nn.ModuleList(seq2seq_encoder_list)

    def get_output_dim(self) -> int:
        return self.seq2seq_encoder_list[-1].get_output_dim()

    def forward(self,
                inputs: torch.Tensor,
                mask: torch.Tensor,
                hidden_state: torch.Tensor = None
                ):
        x = inputs
        for seq2seq_encoder in self.seq2seq_encoder_list:
            try:
                x = seq2seq_encoder.forward(x, mask, hidden_state)
            except TypeError:
                x = seq2seq_encoder.forward(x, mask)
        return x


if __name__ == '__main__':
    pass
