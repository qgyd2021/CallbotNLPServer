from collections import defaultdict
from typing import List, Optional

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric


class HarnnMetric(Metric):
    def __init__(self) -> None:
        self.correct_count = defaultdict(int)
        self.total_count = defaultdict(int)

    def __call__(self,
                 predictions: List[torch.Tensor],
                 gold_labels: List[torch.Tensor],
                 mask: Optional[torch.Tensor] = None):
        """
        :param predictions: list of tensor, shape=(batch_size, num_classes)
        :param gold_labels: list of tensor, shape= (batch_size,)
        :param mask:
        :return:
        """
        for i, (prediction, gold_label) in enumerate(zip(predictions, gold_labels)):
            _, indices = torch.max(prediction, dim=-1)
            correct = torch.sum(torch.eq(indices, gold_label).float())
            total = gold_label.numel()
            self.correct_count[i] += correct
            self.total_count[i] += total

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        result = dict()
        for k, correct in self.correct_count.items():
            total = self.total_count[k]
            if total > 1e-12:
                accuracy = float(correct) / float(total)
            else:
                accuracy = 0.0
            result['accuracy{}'.format(k)] = accuracy
        if reset:
            self.reset()
        return result

    @overrides
    def reset(self):
        self.correct_count = defaultdict(int)
        self.total_count = defaultdict(int)

