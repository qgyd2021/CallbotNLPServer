#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
from typing import List

from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim import utils, matutils
import numpy as np

from project_settings import project_path


def demo1():
    word2vec = Word2Vec.load(os.path.join(project_path, 'model/word2vec_100_indonesian/idwiki_word2vec_100.model'))
    wv = word2vec.wv

    def cosine_similarity(tokens1: List[str], tokens2: List[str]):
        v1 = [wv[key] for key in tokens1 if key in wv]
        v2 = [wv[key] for key in tokens2 if key in wv]
        if len(v1) == 0 or len(v2) == 0:
            return 0.0

        result = np.dot(matutils.unitvec(np.array(v1).mean(axis=0)), matutils.unitvec(np.array(v2).mean(axis=0)))

        return result

    # print(wv)
    # print(wv.index_to_key)
    text1 = 'betul ini saya'
    text2 = 'betul ini'
    tokens1 = text1.split()
    tokens2 = text2.split()
    result = cosine_similarity(tokens1, tokens2)
    print(result)
    return


def demo2():
    wv = KeyedVectors.load(os.path.join(project_path, 'model/glove.6B/glove.6B.50d.txt'))
    print(wv)
    return


if __name__ == '__main__':
    # demo1()
    demo2()
