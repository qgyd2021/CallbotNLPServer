#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Callable, Dict, List, Set, Tuple, TypeVar, Optional

import numpy as np
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure
import cv2 as cv

from toolbox.allennlp.modules.wave_feature_extractors.wave_feature_extractor import WaveFeatureExtractor


@Model.register("basic_wave_classifier")
class BasicWaveClassifier(Model):
    def __init__(self, vocab: Vocabulary,
                 seq2seq_encoder: Seq2SeqEncoder,
                 seq2vec_encoder: Seq2VecEncoder,
                 wave_feature_extractor: WaveFeatureExtractor = None,
                 dropout: float = None,
                 num_labels: int = None,
                 label_namespace: str = "labels",
                 max_wave_value: float = 1.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        self._seq2seq_encoder = seq2seq_encoder
        self._seq2vec_encoder = seq2vec_encoder
        self.wave_feature_extractor = wave_feature_extractor
        self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None

        self._label_namespace = label_namespace

        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)

        self._max_wave_value = max_wave_value

        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    @overrides
    def forward(self,
                array: torch.FloatTensor,
                mask: torch.LongTensor = None,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:

        if self.wave_feature_extractor is not None:
            array = array / self._max_wave_value
            array = self.wave_feature_extractor(array)

        batch_size, sequence_length, _ = array.size()

        if mask is None:
            mask = torch.ones(size=array.shape[:-1], device=array.device)
        else:
            mask = mask.numpy()
            mask = cv.resize(mask, dsize=(sequence_length, batch_size))
            mask = torch.tensor(mask, dtype=torch.long, device=array.device)

        array = self._seq2seq_encoder.forward(array, mask=mask)

        array = self._seq2vec_encoder.forward(array, mask=mask)

        if self._dropout:
            array = self._dropout(array)

        logits = self._classification_layer(array)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = (self.vocab.get_index_to_token_vocabulary(self._label_namespace)
                         .get(label_idx, str(label_idx)))
            classes.append(label_str)
        output_dict["label"] = classes
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {'accuracy': self._accuracy.get_metric(reset)}
        return metrics


if __name__ == '__main__':
    pass
