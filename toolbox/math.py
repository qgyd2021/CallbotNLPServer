#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import List
import numpy as np


def softmax(x):
    x = np.array(x, dtype=np.float32)

    total = np.sum(np.exp(x))
    result = x / total
    return result


def score_transform2(x: float, stages: List[float], scores: List[float], ndigits: int = 4):
    """
    对 0 到 1 之间的 float 存转换.
    x, stages 和 scores 中的值都应该是 0-1 之间的 float.
    """
    stages = [1.0] + stages + [0.0]
    scores = [1.0] + scores + [0.0]
    last_stage = 1.0
    last_score = 1.0
    for stage, score in zip(stages, scores):
        if x >= stage:
            result = score + (x - stage) / (last_stage - stage + 1e-7) * (last_score - score)
            return round(result, ndigits)
        last_stage = stage
        last_score = score
    raise ValueError('values of x, stages and scores should between 0 and 1, '
                     'stages and scores should be same length and decreased. '
                     'x: {}, stages: {}, scores: {}'.format(x, stages, scores))


def score_transform(x: float, stages: List[float], scores: List[float], ndigits: int = 4):
    """
    对 0 到 1 之间的 float 存转换.
    x, stages 和 scores 中的值都应该是 0-1 之间的 float.
    """
    last_stage = stages[0]
    last_score = scores[0]
    stages = stages[1:]
    scores = scores[1:]
    for stage, score in zip(stages, scores):
        if x >= stage:
            result = score + (x - stage) / (last_stage - stage + 1e-7) * (last_score - score)
            return round(result, ndigits)
        last_stage = stage
        last_score = score
    raise ValueError('values of x, stages and scores should between 0 and 1, '
                     'stages and scores should be same length and decreased. '
                     'x: {}, stages: {}, scores: {}'.format(x, stages, scores))


def demo1():
    stages = [1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.0]
    scores = [1.0, 0.7, 0.4, 0.3, 0.2, 0.1, 0.0]

    for x in (1.0, 0.85, 0.0):
        result = score_transform(x, stages, scores)
        print(result)
    return


def demo2():
    stages = [1.0, 0.5, 0.3, 0.1, 0.0]
    scores = [1.2, 1.0, 0.7, 0.7, 0.0]

    for x in (1.0, 0.85, 0.1, 0.0):
        result = score_transform(x, stages, scores)
        print(result)
    return


if __name__ == '__main__':
    demo1()
    demo2()
