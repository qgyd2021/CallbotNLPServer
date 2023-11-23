#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
https://github.com/lucidrains/local-attention
"""
from overrides import overrides

import allennlp
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.nn.util import get_mask_from_sequence_lengths
from allennlp.nn.util import masked_softmax, weighted_sum
import torch
from torch.nn import Dropout, Linear


@Seq2SeqEncoder.register("multi_head_self_local_attention")
class MultiHeadSelfLocalAttention(Seq2SeqEncoder):
    def __init__(self,
                 left_size: int,
                 right_size: int,
                 num_heads: int,
                 input_dim: int,
                 attention_dim: int,
                 values_dim: int,
                 output_projection_dim: int = None,
                 attention_dropout_prob: float = 0.1
                 ):
        """

        """
        super().__init__()
        self._left_size = left_size
        self._right_size = right_size

        self._num_heads = num_heads
        self._input_dim = input_dim
        self._output_dim = output_projection_dim or input_dim
        self._attention_dim = attention_dim
        self._values_dim = values_dim

        if attention_dim % num_heads != 0:
            raise ValueError(f"Key size ({attention_dim}) must be divisible by the number of "
                             f"attention heads ({num_heads}).")

        if values_dim % num_heads != 0:
            raise ValueError(f"Value size ({values_dim}) must be divisible by the number of "
                             f"attention heads ({num_heads}).")

        self._combined_projection = Linear(input_dim, 2 * attention_dim + values_dim)

        self._scale = (input_dim // num_heads) ** 0.5
        self._output_projection = Linear(values_dim, self._output_dim)
        self._attention_dropout = Dropout(attention_dropout_prob)

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    @overrides
    def is_bidirectional(self):
        return False

    @overrides
    def forward(self,
                inputs: torch.Tensor,
                mask: torch.LongTensor = None) -> torch.FloatTensor:
        num_heads = self._num_heads

        batch_size, timesteps, _ = inputs.size()
        if mask is None:
            mask = inputs.new_ones(batch_size, timesteps, device=inputs.device)

        combined_projection = self._combined_projection(inputs)
        queries, keys, *values = combined_projection.split(self._attention_dim, -1)
        queries = queries.contiguous()
        keys = keys.contiguous()
        values = torch.cat(values, -1).contiguous()

        queries = torch.unsqueeze(queries, dim=2)

        keys = torch.concat(tensors=[
            torch.zeros(size=(batch_size, self._left_size, self._attention_dim), device=keys.device),
            keys,
            torch.zeros(size=(batch_size, self._right_size, self._attention_dim), device=keys.device),
        ], dim=1)
        keys_ = list()
        for i in range(timesteps):
            key = keys[:, i:1 + i + self._left_size + self._right_size, :]
            keys_.append(key)
        keys = torch.stack(keys_, dim=1)

        values = torch.concat(tensors=[
            torch.zeros(size=(batch_size, self._left_size, self._attention_dim), device=values.device),
            values,
            torch.zeros(size=(batch_size, self._right_size, self._attention_dim), device=values.device),
        ], dim=1)
        values_ = list()
        for i in range(timesteps):
            value = values[:, i:1 + i + self._left_size + self._right_size, :]
            values_.append(value)

        values = torch.stack(values_, dim=1)

        frame_mask = torch.concat(tensors=[
            torch.zeros(size=(batch_size, self._left_size), device=mask.device),
            mask,
            torch.zeros(size=(batch_size, self._right_size), device=mask.device),
        ], dim=1)
        frame_mask_ = list()
        for i in range(timesteps):
            m = frame_mask[:, i:1 + i + self._left_size + self._right_size]
            frame_mask_.append(m)

        frame_mask = torch.stack(frame_mask_, dim=1)

        win_size = self._left_size + 1 + self._right_size
        queries_per_head = queries.view(batch_size, timesteps, 1, num_heads, int(self._attention_dim/num_heads))
        keys_per_head = keys.view(batch_size, timesteps, win_size, num_heads, int(self._attention_dim/num_heads))
        values_per_head = values.view(batch_size, timesteps, win_size, num_heads, int(self._attention_dim/num_heads))

        queries_per_head = torch.transpose(queries_per_head, dim0=2, dim1=3)
        keys_per_head = torch.transpose(keys_per_head, dim0=2, dim1=3)
        values_per_head = torch.transpose(values_per_head, dim0=2, dim1=3)

        scaled_similarities = torch.sum(queries_per_head * keys_per_head, dim=-1)

        frame_mask = torch.unsqueeze(frame_mask, dim=-2)

        attention = masked_softmax(scaled_similarities,
                                   frame_mask,
                                   memory_efficient=True)

        attention = self._attention_dropout(attention)
        outputs = weighted_sum(values_per_head, attention)

        outputs = torch.reshape(outputs, shape=(batch_size, timesteps, self._values_dim))

        outputs = self._output_projection(outputs)

        return outputs


def demo1():
    inputs = torch.rand(size=(2, 35, 16), dtype=torch.float32)
    input_lengths = torch.tensor([12, 35], dtype=torch.long)
    max_length = torch.max(input_lengths)
    max_length = int(max_length)

    mask: torch.LongTensor = get_mask_from_sequence_lengths(input_lengths, max_length=max_length)

    multi_head_self_local_attention = MultiHeadSelfLocalAttention(
        left_size=3,
        right_size=3,
        num_heads=4,
        input_dim=16,
        attention_dim=32,
        values_dim=32,
        output_projection_dim=16,
    )
    result = multi_head_self_local_attention.forward(
        inputs=inputs,
        mask=mask,
    )
    print(result.shape)

    return


if __name__ == '__main__':
    demo1()
