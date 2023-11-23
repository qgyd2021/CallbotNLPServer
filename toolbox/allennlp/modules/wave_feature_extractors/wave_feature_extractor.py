#!/usr/bin/python3
# -*- coding: utf-8 -*-
from allennlp.modules.encoder_base import _EncoderBase
from allennlp.common import Registrable


class WaveFeatureExtractor(_EncoderBase, Registrable):
    """
    从 1 维的音频数组中提取特征.
    """
    def get_input_dim(self) -> int:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError


if __name__ == '__main__':
    pass
