#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Optional

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


@Metric.register("threshold_accuracy")
class ThresholdAccuracy(Metric):
    def __init__(self, threshold: float = 0.5) -> None:
        if not 0.0 <= threshold < 1.0:
            raise ConfigurationError("threshold should between [0.0, 1.0]")
        self._threshold = threshold
        self.correct_count = 0.
        self.total_count = 0.

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        preds = torch.greater(predictions, self._threshold).type(dtype=torch.int)

        correct = preds.eq(gold_labels).float()

        if mask is not None:
            correct *= mask.view(-1, 1).float()
            self.total_count += mask.sum()
        else:
            self.total_count += gold_labels.numel()
        self.correct_count += correct.sum()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        if self.total_count > 1e-12:
            accuracy = float(self.correct_count) / float(self.total_count)
        else:
            accuracy = 0.0
        if reset:
            self.reset()
        return accuracy

    @overrides
    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0


def demo1():
    threshold_accuracy = ThresholdAccuracy()
    inputs = torch.tensor([1, 0, 0])
    targets = torch.tensor([1, 0, 1])
    threshold_accuracy(inputs, targets)
    ret = threshold_accuracy.get_metric()
    print(ret)
    return


if __name__ == '__main__':
    demo1()
