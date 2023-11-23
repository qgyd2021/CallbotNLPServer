from collections import defaultdict
from typing import List, Optional

import editdistance
from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric


class CTCWERMetric(Metric):
    def __init__(self) -> None:
        self.correct_count = defaultdict(int)
        self.total_count = defaultdict(int)

    def get_metric(self, reset: bool = False):
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

    def __call__(self,
                 predictions: List[torch.Tensor],
                 gold_labels: List[torch.Tensor],
                 mask: Optional[torch.Tensor] = None,
                 input_sizes=None,
                 target_sizes=None,
                 ):

        for i in range(len(predictions)):
            label = gold_labels[i][:target_sizes[i]]
            pred = []
            for j in range(len(predictions[i][:input_sizes[i]])):
                if predictions[i][j] == 0:
                    continue
                if j == 0:
                    pred.append(predictions[i][j])
                if j > 0 and predictions[i][j] != predictions[i][j-1]:
                    pred.append(predictions[i][j])
            self.correct_count[i] += editdistance.eval(label, pred)
            self.total_count[i] += len(label)

    @overrides
    def reset(self):
        self.correct_count = defaultdict(int)
        self.total_count = defaultdict(int)


def demo1():

    return


if __name__ == '__main__':
    demo1()
