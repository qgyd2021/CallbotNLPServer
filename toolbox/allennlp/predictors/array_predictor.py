#!/usr/bin/python3
# -*- coding: utf-8 -*-
from copy import deepcopy
from typing import List, Dict

from overrides import overrides
import numpy as np

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import LabelField


@Predictor.register('array_predictor')
class ArrayPredictor(Predictor):

    def predict(self, array: np.ndarray) -> JsonDict:
        return self.predict_json({
            'array': array,
        })

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        array = json_dict["array"]
        return self._dataset_reader.array_to_instance(array=array)

    @overrides
    def predictions_to_labeled_instances(self,
                                         instance: Instance,
                                         outputs: Dict[str, np.ndarray]) -> List[Instance]:
        new_instance = deepcopy(instance)
        label = np.argmax(outputs['probs'])
        new_instance.add_field('label', LabelField(int(label), skip_indexing=True))
        return [new_instance]


if __name__ == '__main__':
    pass
