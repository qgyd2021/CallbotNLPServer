#!/usr/bin/python3
# -*- coding: utf-8 -*-
from overrides import overrides
import torch
from torch.nn import Dropout, Linear
from torch.nn import functional

from allennlp.nn.util import masked_softmax, weighted_sum
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder


@Seq2SeqEncoder.register("multi_head_external_attention")
class MultiHeadExternalAttention(Seq2SeqEncoder):
    """
    参考链接:
    https://github.com/xmu-xiaoma666/External-Attention-pytorch
    """
    def __init__(self,
                 num_heads: int,
                 num_values: int,
                 input_dim: int,
                 attention_dim: int,
                 values_dim: int,
                 output_projection_dim: int = None,
                 attention_dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_values = num_values
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.values_dim = values_dim
        self.output_dim = output_projection_dim or input_dim

        if attention_dim % num_heads != 0:
            raise ValueError(f"Key size ({attention_dim}) must be divisible by the number of "
                             f"attention heads ({num_heads}).")

        if values_dim % num_heads != 0:
            raise ValueError(f"Value size ({values_dim}) must be divisible by the number of "
                             f"attention heads ({num_heads}).")

        self.query = torch.nn.Linear(input_dim, attention_dim, bias=True)
        self.key = torch.nn.Linear(attention_dim // num_heads, num_values, bias=False)
        self.value = torch.nn.Linear(num_values, values_dim // num_heads, bias=False)

        self.attention_dropout = Dropout(attention_dropout_prob)

        self.output_projection = Linear(values_dim, self.output_dim)

    def get_input_dim(self):
        return self.input_dim

    def get_output_dim(self):
        return self.output_dim

    @overrides
    def is_bidirectional(self):
        return False

    @overrides
    def forward(self,
                inputs: torch.Tensor,
                mask: torch.LongTensor = None) -> torch.FloatTensor:
        # [batch_size, timesteps, attention_dim]
        query = self.query(inputs)
        # [batch_size, timesteps, num_heads, attention_dim // num_heads]
        query = torch.reshape(query, shape=(*inputs.shape[:-1], self.num_heads, -1))
        # [batch_size, timesteps, num_heads, attention_dim // num_heads]
        similarities = self.key(query)
        # [batch_size, timesteps, num_heads, attention_dim // num_heads]
        attention = functional.softmax(similarities, dim=-1)
        attention = self.attention_dropout(attention)

        # [batch_size, timesteps, num_heads, values_dim // num_heads]
        weighted_sum = self.value(attention)
        # [batch_size, timesteps, values_dim]
        outputs = torch.reshape(weighted_sum, shape=(*weighted_sum.shape[:-2], -1))

        outputs = self.output_projection(outputs)
        return outputs


def demo1():
    inputs = torch.randn(50, 49, 512)
    mask = torch.ones(size=(50, 49))
    multi_head_external_attention = MultiHeadExternalAttention(
        num_heads=4,
        num_values=128,
        input_dim=512,
        attention_dim=512,
        values_dim=128,
    )
    print(multi_head_external_attention)
    output = multi_head_external_attention.forward(inputs, mask)
    print(output.shape)
    return


if __name__ == '__main__':
    demo1()
