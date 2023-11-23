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
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.layer_norm import LayerNorm

from toolbox.allennlp.modules.seq2seq_encoders.multi_head_external_attention import MultiHeadExternalAttention


@Seq2SeqEncoder.register('stacked_external_attention')
class StackedExternalAttentionEncoder(Seq2SeqEncoder):
    def __init__(self,
                 num_layers: int,
                 input_dim: int,
                 feedforward_hidden_dim: int,
                 hidden_dim: int,
                 attention_dim: int,
                 num_attention_heads: int,
                 num_values: int,
                 output_projection_dim: int = None,
                 dropout_prob: float = 0.1,
                 residual_dropout_prob: float = 0.2,
                 attention_dropout_prob: float = 0.1
                 ):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.feedforward_hidden_dim = feedforward_hidden_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.num_attention_heads = num_attention_heads
        self.num_values = num_values
        self.output_dim = output_projection_dim or input_dim
        self.dropout_prob = dropout_prob
        self.residual_dropout_prob = residual_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob

        self._attention_layers: List[MultiHeadExternalAttention] = []
        self._feedfoward_layers: List[FeedForward] = []
        self._layer_norm_layers: List[LayerNorm] = []
        self._feed_forward_layer_norm_layers: List[LayerNorm] = []

        feedfoward_input_dim = input_dim
        for i in range(num_layers):
            feedfoward = FeedForward(
                feedfoward_input_dim,
                activations=[
                    Activation.by_name('relu')(),
                    Activation.by_name('linear')()
                ],
                hidden_dims=[feedforward_hidden_dim, hidden_dim],
                num_layers=2,
                dropout=dropout_prob
            )

            self.add_module(f"feedforward_{i}", feedfoward)
            self._feedfoward_layers.append(feedfoward)

            feedforward_layer_norm = LayerNorm(feedfoward.get_output_dim())
            self.add_module(f"feedforward_layer_norm_{i}", feedforward_layer_norm)
            self._feed_forward_layer_norm_layers.append(feedforward_layer_norm)

            self_attention = MultiHeadExternalAttention(
                num_heads=num_attention_heads,
                num_values=num_values,
                input_dim=hidden_dim,
                attention_dim=attention_dim,
                values_dim=attention_dim,
                attention_dropout_prob=attention_dropout_prob
            )
            self.add_module(f"self_attention_{i}", self_attention)
            self._attention_layers.append(self_attention)

            layer_norm = LayerNorm(self_attention.get_output_dim())
            self.add_module(f"layer_norm_{i}", layer_norm)
            self._layer_norm_layers.append(layer_norm)

            feedfoward_input_dim = hidden_dim

        self.output_projection = torch.nn.Linear(hidden_dim, self.output_dim)

        self.dropout = torch.nn.Dropout(residual_dropout_prob)

    def get_input_dim(self):
        return self.input_dim

    def get_output_dim(self):
        return self.output_dim

    @overrides
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor):
        if inputs.shape[-1] != self.get_input_dim():
            raise AssertionError()

        output = inputs
        for i in range(len(self._attention_layers)):
            attention = getattr(self, f"self_attention_{i}")
            feedforward = getattr(self, f"feedforward_{i}")
            feedforward_layer_norm = getattr(self, f"feedforward_layer_norm_{i}")
            layer_norm = getattr(self, f"layer_norm_{i}")

            cached_input = output

            # shape [batch_size, timesteps, input_size]
            feedforward_output = feedforward(output)
            feedforward_output = self.dropout(feedforward_output)
            if feedforward_output.size() == cached_input.size():
                feedforward_output = feedforward_layer_norm(feedforward_output + cached_input)
            # shape [batch_size, sequence_length, hidden_dim]
            attention_output = attention(feedforward_output, mask)
            output = layer_norm(self.dropout(attention_output) + feedforward_output)

        output = self.output_projection(output)
        return output


def demo1():
    inputs = torch.randn(50, 49, 512)
    mask = torch.ones(size=(50, 49))

    stacked_external_attention = StackedExternalAttentionEncoder(
        num_layers=2,
        input_dim=512,
        feedforward_hidden_dim=256,
        hidden_dim=128,
        attention_dim=512,
        num_values=128,
        num_attention_heads=4,
    )

    output = stacked_external_attention.forward(inputs, mask)
    return


if __name__ == '__main__':
    demo1()
