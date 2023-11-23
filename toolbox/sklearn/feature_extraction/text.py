#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import Union

import numpy as np
from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer


class TfidfTransformer(object):
    def __init__(self, *, ratio_tf: False, smooth_idf=False, sublinear_tf=False
                 ):
        self.ratio_tf = ratio_tf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

        self.idf = None

    def _tf(self, x: np.ndarray):
        if self.sublinear_tf:
            x = x + 1

        if self.ratio_tf:
            tf = x / (np.sum(x, axis=-1, keepdims=True) + 1e-7)
        else:
            tf = x

        return tf

    def _idf(self, x: np.ndarray):
        n_samples, n_features = x.shape

        x = np.where(x == 0, 0, 1)
        df = np.sum(x, axis=0, keepdims=True)

        if self.smooth_idf:
            df += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)

        idf = np.log(n_samples / df) + 1

        return idf

    def fit(self, x: Union[np.ndarray, csr_matrix]):
        if isinstance(x, csr_matrix):
            x = x.toarray()

        idf = self._idf(x)
        self.idf = idf
        return self.idf

    def transform(self, x: Union[np.ndarray, csr_matrix]):
        if isinstance(x, csr_matrix):
            x = x.toarray()

        tf = self._tf(x)
        tf_idf = tf * self.idf
        return tf_idf

    def fit_transform(self, x: Union[np.ndarray, csr_matrix]):
        if isinstance(x, csr_matrix):
            x = x.toarray()

        self.fit(x)
        return self.transform(x)


def demo1():
    data = ['TF-IDF 算法 的 主要 思想 是',
            '算法 一个 重要 特点 可以 脱离 语料库 背景',
            '如果 一个 网页 被 很多 其他 网页 链接 说明 网页 重要',
            'TF-IDF 算法 的 原理 很 简单',
            'TF-IDF 算法 的 应用 很 广泛']

    vectorizer = CountVectorizer(max_features=5)
    count_matrix = vectorizer.fit_transform(data)
    # count_matrix = count_matrix.toarray()
    print(vectorizer.vocabulary_)

    transformer = TfidfTransformer(
        # ratio_tf=True,
        ratio_tf=True,
        smooth_idf=False,
        sublinear_tf=False,
    )
    tf_idf = transformer.fit_transform(count_matrix)
    print(tf_idf)
    return


if __name__ == '__main__':
    demo1()
