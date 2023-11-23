# -*- encoding=UTF-8 -*-
from collections import Counter, defaultdict
import json
import logging
import os
import random
import sys
import time
from typing import Dict, List

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))

from cacheout import Cache
from elasticsearch import Elasticsearch, helpers
import faiss
import numpy as np
import pandas as pd
import requests
from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import datasets, model_selection, naive_bayes
from sklearn.utils.validation import check_is_fitted
from sklearn.decomposition import PCA

import torch
from tqdm import tqdm

from nxtech.common.params import Params
from nxtech.database.mysql_connect import MySqlConnect
from nxtech.database.es_connect import ElasticSearchConnect
from nxtech.nlpbot.candidate import RecallCandidates, MysqlCandidatesIncludeIntentLib, ListCandidates
from nxtech.nlpbot.misc import Preprocess, Replacer, Filter
from nxtech.nlpbot.model import HttpAllenNlpPredictor
from nxtech.nlpbot.request_context import RequestContext
from nxtech.nlpbot.tokenizer import Tokenizer, JiebaTokenizer

logger = logging.getLogger('nxtech')


class Recall(Params):
    """
    召回前, 应对句子去除停用词, 以避免停用词匹配的召回.
    """
    def __init__(self):
        super().__init__()

    def recall(self, context: RequestContext) -> List[dict]:
        """
        召回的结果 dict 中应至少包含 text, score, score_key 三个键.

        score_key 在 Scorer 中发挥作用. 使不同的召回结果, 可以使用不同的算分方式.
        """
        raise NotImplementedError

    async def async_recall(self, context: RequestContext) -> List[dict]:
        return self.recall(context)


@Recall.register('elastic_search')
class ElasticSearchRecall(Recall):
    # """为每一个场景单独创建一个 index, 太多了, 让所有的场景都在同一个索引中, 通过 scene_id 过滤. """
    """为每一个场景单独创建一个 index. """

    mapping = {
        'properties': {
            'text': {
                'type': 'text',
                'analyzer': 'whitespace',
                'search_analyzer': 'whitespace'
            },
            'text_preprocessed': {
                'type': 'text',
                'analyzer': 'whitespace',
                'search_analyzer': 'whitespace'
            },
            'text_id': {
                'type': 'keyword'
            },
            'product_id': {
                'type': 'keyword'
            },
            'scene_id': {
                'type': 'keyword'
            },
            'parent_node_id': {
                'type': 'keyword'
            },
            'child_node_id': {
                'type': 'keyword'
            },
            'node_type': {
                'type': 'long'
            },
            'resource_id': {
                'type': 'keyword'
            },
        }
    }

    def __init__(self,
                 es_connect: ElasticSearchConnect,
                 product_id: str,
                 scene_id: str,
                 node_id: str,
                 env: str,
                 tokenizer: Tokenizer,
                 top_k: int = 30,
                 preprocess_list: List[Preprocess] = None,
                 replacer_list: List[Replacer] = None,
                 # filter_list: List[Filter] = None,
                 update_es: bool = False,
                 mysql_connect: MySqlConnect = None,
                 group_name: str = 'default',
                 ):
        super().__init__()
        self.es_connect = es_connect
        self.product_id = product_id
        self.scene_id = scene_id
        self.node_id = node_id
        self.env = env
        self.top_k = top_k
        self.preprocess_list = preprocess_list or list()
        self.tokenizer = tokenizer
        self.replacer_list = replacer_list or list()
        # 召回, 不要用停用词过滤. 最大限度召回.
        # self.filter_list = filter_list or list()
        self.update_es = update_es
        self.mysql_connect = mysql_connect
        self.group_name = group_name

        # self.index = 'nlpbot_{}_{}_{}'.format(self.product_id, self.env, self.group_name)
        # 每个场景有一个索引.
        self.index = 'nlpbot_{}_{}_{}_{}'.format(self.product_id, self.scene_id, self.env, self.group_name)

        if update_es:
            if self.mysql_connect is None:
                raise ValueError('mysql_connect is expected when update es.')
            self._update_es()
        elif not self.es_connect.es.indices.exists(index=self.index):
            self._update_es()
        else:
            pass

    def _update_es(self):
        # 刷新索引
        if not self.es_connect.es.indices.exists(index=self.index):
            # 设置索引最大数量据.
            body = {
                "settings": {
                    "index": {
                        "max_result_window": 100000000
                    }
                }
            }
            self.es_connect.es.indices.create(index=self.index, body=body)

            # 设置文档结构
            self.es_connect.es.indices.put_mapping(
                index=self.index,
                doc_type='_doc',
                body=self.mapping,
                params={"include_type_name": "true"}
            )

        # 查询现有数据
        # query = {
        #     'query': {
        #         'bool': {
        #             'filter': [
        #                 {'term': {'product_id': self.product_id}},
        #                 {'term': {'scene_id': self.scene_id}},
        #                 {'term': {'parent_node_id': self.node_id}},
        #
        #             ]
        #         },
        #     },
        #     'size': 65536
        # }
        # result = self.es_connect.es.search(
        #     index=self.index,
        #     body=query,
        # )
        # print(len(result['hits']['hits']))

        # 删掉旧的数据.
        query = {
            'query': {
                'bool': {
                    'filter': [
                        {'term': {'product_id': self.product_id}},
                        {'term': {'scene_id': self.scene_id}},
                        {'term': {'parent_node_id': self.node_id}},

                    ]
                },
            },
        }
        result = self.es_connect.es.delete_by_query(
            index=self.index,
            body=query,
        )
        # print(result)

        # 候选句子
        candidates: List[dict] = MysqlCandidatesIncludeIntentLib(
            product_id=self.product_id,
            scene_id=self.scene_id,
            node_id=self.node_id,
            mysql_connect=self.mysql_connect,
            resource_type_list=['similar_question'],
        ).get()

        # 写入新的数据
        rows = list()
        for idx, candidate in enumerate(candidates):
            row = {
                'product_id': candidate['product_id'],
                'scene_id': candidate['scene_id'],
                'parent_node_id': self.node_id,
                'child_node_id': candidate['node_id'],
                'node_type': candidate['node_type'],
                'node_desc': candidate['node_desc'],
                'resource_id': candidate['resource_id'],
                'text': candidate['text'],
                'text_preprocessed': ' '.join(self._split(candidate['text'])),
                'word_name': candidate['word_name'],
            }
            rows.append({
                '_op_type': 'index',
                '_index': self.index,
                # '_id': idx,
                # '_type': 'last',
                '_source': row
            })
            # result = self.es_connect.es.index(index=self.index, body=row, id=idx)
            # print(result)
        helpers.bulk(client=self.es_connect.es, actions=rows)

        # 刷新数据
        self.es_connect.es.indices.refresh(index=self.index)
        return

    def _split(self, text: str):
        for preprocess in self.preprocess_list:
            text = preprocess.process(text)

        outlst, _ = self.tokenizer.tokenize(text)

        for replacer in self.replacer_list:
            try:
                outlst = replacer.replace(outlst)
                outlst = outlst[0]
            except IndexError:
                return list()

        return outlst

    def recall(self, context: RequestContext) -> List[dict]:
        logger.debug('{} racall: '.format(self.register_name))
        context.append_row("""### {} recall detail""".format(self.register_name))

        text = context.text
        outlst = self._split(text)
        # es 召回, 都采用空隔分词, 用于查询.
        text_preprocessed = ' '.join(outlst)

        logging.debug('text_preprocessed: {}'.format(text_preprocessed))
        context.append_row("""user text: {}, preprocessed text: {}""".format(text, text_preprocessed))

        query = {
            'query': {
                'bool': {
                    'must': [{
                        'match': {
                            'text_preprocessed': text_preprocessed
                        }
                    }],
                    'filter': [
                        {'term': {'product_id': self.product_id}},
                        {'term': {'scene_id': self.scene_id}},
                        {'term': {'parent_node_id': self.node_id}},
                    ]
                },
            },
        }
        js = self.es_connect.search(
            index=self.index,
            size=self.top_k,
            body=query
        )
        return self.output_schema(js, context)

    async def async_recall(self, context: RequestContext) -> List[dict]:
        """
        category 是 t_dialog_node_info 表的 node_type 字段:
        ans_id 是  t_dialog_node_info 表的 node_id 字段.
        qst_id 是 t_dialog_resource_info 表的 res_id 字段.
        qst 是 t_dialog_resource_info 表的 word 字段:

        """
        logger.debug('{} racall: '.format(self.register_name))
        context.append_row("""### {} recall detail""".format(self.register_name))

        text = context.text
        outlst = self._split(text)
        # es 召回, 都采用空隔分词, 用于查询.
        text_preprocessed = ' '.join(outlst)

        logging.debug('text_preprocessed: {}'.format(text_preprocessed))
        context.append_row("""user text: {}, preprocessed text: {}""".format(text, text_preprocessed))

        query = {
            'query': {
                'bool': {
                    'must': [{
                        'match': {
                            'text_preprocessed': text_preprocessed
                        }
                    }],
                    'filter': [
                        {'term': {'product_id': self.product_id}},
                        {'term': {'scene_id': self.scene_id}},
                        {'term': {'parent_node_id': self.node_id}},
                    ]
                },
            },
        }
        js = await self.es_connect.async_search(
            index=self.index,
            size=self.top_k,
            body=query
        )
        return self.output_schema(js, context)

    def output_schema(self, js, context: RequestContext):
        hits = js['hits']['hits']

        logger.debug('{} recall, count: {}'.format(self.register_name, len(hits)))
        context.append_row("""recall count: {}""".format(len(hits)))

        result = list()
        for hit in hits:
            score = hit['_score']
            source = hit['_source']
            node_type = source['node_type']
            node_id = source['child_node_id']
            node_desc = source['node_desc']
            text = source['text']
            text_preprocessed = source['text_preprocessed']
            resource_id = source['resource_id']
            word_name = source['word_name']

            logger.debug('text: {}, text_preprocessed: {}, score: {}, node_id: {}, '
                         'node_desc: {}, word_name: {}'.format(
                text, text_preprocessed, score, node_id, node_desc, word_name)
            )

            result.append({
                'text': text,
                'word_name': word_name,
                'node_id': node_id,
                'node_desc': node_desc,
                'node_type': node_type,
                'resource_id': resource_id,
                'score': score,
                'scorer_key': node_id,
                'recaller': self.register_name,
            })

        context.append_table(result)
        return result


@Recall.register('faiss')
class FaissRecall(Recall):
    def __init__(self,
                 product_id: str,
                 scene_id: str,
                 node_id: str,
                 env: str,
                 tokenizer: Tokenizer,
                 delimiter: str = '',
                 top_k: int = 30,
                 dim: int = 128,
                 sentence_vector_key: str = 'sentence_vector',
                 sim_mode: str = 'cosine',
                 http_allennlp_predictor: HttpAllenNlpPredictor = None,
                 preprocess_list: List[Preprocess] = None,
                 replacer_list: List[Replacer] = None,
                 mysql_connect: MySqlConnect = None,
                 ):
        super().__init__()

        self.product_id = product_id
        self.scene_id = scene_id
        self.node_id = node_id
        self.env = env
        self.predictor = http_allennlp_predictor
        self.tokenizer = tokenizer
        self.delimiter = delimiter
        self.top_k = top_k
        self.dim = dim
        self.sentence_vector_key = sentence_vector_key
        self.sim_mode = sim_mode
        self.preprocess_list = preprocess_list or list()
        self.replacer_list = replacer_list or list()
        self.mysql_connect = mysql_connect
        self.vector_list = None
        self.text_info_list = None

        self.index = faiss.IndexFlatL2(self.dim)
        self._init_index()

    def _init_index(self):
        # 候选句子
        candidates: List[dict] = MysqlCandidatesIncludeIntentLib(
            product_id=self.product_id,
            scene_id=self.scene_id,
            node_id=self.node_id,
            mysql_connect=self.mysql_connect,
            resource_type_list=['similar_question'],
        ).get()

        if len(candidates) == 0:
            raise AssertionError(
                'no candidate wording for the node. scene_id: {}, node_id: {}'.format(
                    self.scene_id, self.node_id
                ))

        vector_list = list()
        text_info_list = list()

        for candidate in tqdm(candidates):
            text = candidate['text']
            tokens = self._split(text)
            text_for_vector = self.delimiter.join(tokens)
            output_dict = self.predictor.predict_json({'sentence': text_for_vector})
            vector = output_dict.get(self.sentence_vector_key)
            # vector: shape=(output_dim,)
            if vector is None:
                raise KeyError(
                    '{} not found in {}'.format(self.sentence_vector_key, output_dict.keys())
                )

            vector_list.append(vector)

            text_info_list.append({
                'product_id': candidate['product_id'],
                'scene_id': candidate['scene_id'],
                'parent_node_id': self.node_id,
                'child_node_id': candidate['node_id'],
                'node_type': candidate['node_type'],
                'node_desc': candidate['node_desc'],
                'resource_id': candidate['resource_id'],
                'text': text,
                'text_for_vector': text_for_vector,
                'word_name': candidate['word_name'],

            })

        vector_list = np.array(vector_list, dtype=np.float32)
        self.index.add(vector_list)
        self.vector_list = vector_list
        self.text_info_list = text_info_list
        return

    def _split(self, text: str):
        for preprocess in self.preprocess_list:
            text = preprocess.process(text)

        outlst, _ = self.tokenizer.tokenize(text)

        for replacer in self.replacer_list:
            try:
                outlst = replacer.replace(outlst)
                outlst = outlst[0]
            except IndexError:
                return list()

        return outlst

    async def async_recall(self, context: RequestContext) -> List[dict]:
        logger.debug('{} recall: '.format(self.register_name))
        context.append_row("""### {} recall detail""".format(self.register_name))

        text = context.text
        tokens = self._split(text)
        text_for_vector = self.delimiter.join(tokens)
        logger.debug('text: {}, text_for_vector: {}'.format(text, text_for_vector))
        output_dict = await self.predictor.async_predict_json({'sentence': text_for_vector})
        return self.output_schema(output_dict, context)

    def recall(self, context: RequestContext) -> List[dict]:
        logger.debug('{} recall: '.format(self.register_name))
        context.append_row("""### {} recall detail""".format(self.register_name))

        text = context.text

        tokens = self._split(text)
        text_for_vector = self.delimiter.join(tokens)
        logger.debug('text: {}, text_for_vector: {}'.format(text, text_for_vector))
        context.append_row("""text: {}, text_for_vector: {}""".format(text, text_for_vector))

        output_dict = self.predictor.predict_json({'sentence': text_for_vector})
        return self.output_schema(output_dict, context)

    def sim_score(self, vector1, vector2, sim_mode='cosine'):
        if sim_mode == 'cosine':
            sim = np.sum(vector1 * vector2, axis=-1)
        elif sim_mode == 'probs':
            sim = np.sum(np.sqrt(vector1 + 1e-7) * np.sqrt(vector2 + 1e-7), axis=-1)
        else:
            sim = np.sum(vector1 * vector2, axis=-1)
        return sim

    def output_schema(self, output_dict, context: RequestContext):
        vector = output_dict.get(self.sentence_vector_key)
        # vector: shape=(output_dim,)
        if vector is None:
            raise KeyError(
                '{} not found in {}'.format(self.sentence_vector_key, output_dict.keys())
            )

        vector = np.array([vector], dtype=np.float32)
        D, I = self.index.search(vector, 20)

        result = list()
        for idx in I[0]:
            text_info = self.text_info_list[idx]
            idx_vector = self.vector_list[idx]

            sim = self.sim_score(
                vector1=vector,
                vector2=np.array([idx_vector], dtype=np.float32),
                sim_mode=self.sim_mode,
            )

            result.append({
                'text': text_info['text'],
                'word_name': text_info['word_name'],
                'node_id': text_info['child_node_id'],
                'node_desc': text_info['node_desc'],
                'node_type': text_info['node_type'],
                'resource_id': text_info['resource_id'],
                'score': round(float(sim), 4),
                'scorer_key': text_info['child_node_id'],
                'recaller': self.register_name,
            })
        context.append_table(result)
        return result


@Recall.register('deep_knn')
class DeepKnnRecall(Recall):
    def __init__(self,
                 product_id: str,
                 scene_id: str,
                 node_id: str,
                 env: str,
                 tokenizer: Tokenizer,
                 delimiter: str = '',
                 top_k: int = 30,
                 dim: int = 256,
                 sentence_vector_key: str = 'query_vectors',
                 http_allennlp_predictor: HttpAllenNlpPredictor = None,
                 preprocess_list: List[Preprocess] = None,
                 replacer_list: List[Replacer] = None,
                 mysql_connect: MySqlConnect = None,
                 ):
        super().__init__()

        self.product_id = product_id
        self.scene_id = scene_id
        self.node_id = node_id
        self.env = env

        self.predictor = http_allennlp_predictor
        self.tokenizer = tokenizer
        self.delimiter = delimiter
        self.top_k = top_k
        self.dim = dim
        self.sentence_vector_key = sentence_vector_key
        self.preprocess_list = preprocess_list or list()
        self.replacer_list = replacer_list or list()
        self.mysql_connect = mysql_connect
        #
        self.index = faiss.IndexFlatL2(self.dim)
        self.vector_list = None
        self.text_info_list = None

        self._init_index()

    def _init_index(self):
        # 候选句子
        candidates: List[dict] = MysqlCandidatesIncludeIntentLib(
            product_id=self.product_id,
            scene_id=self.scene_id,
            node_id=self.node_id,
            mysql_connect=self.mysql_connect,
            resource_type_list=['similar_question'],
        ).get()

        if len(candidates) == 0:
            raise AssertionError(
                'no candidate wording for the node. scene_id: {}, node_id: {}'.format(
                    self.scene_id, self.node_id
                ))

        vector_list = list()
        text_info_list = list()

        for candidate in tqdm(candidates):
            text = candidate['text']
            tokens = self._split(text)
            text_for_vector = self.delimiter.join(tokens)
            output_dict = self.predictor.predict_json(
                inputs={
                    'task': {
                        'query': [
                            text_for_vector,
                        ]
                    }
                },
            )
            vector = output_dict.get(self.sentence_vector_key)
            # vector: shape=(1, output_dim)
            if vector is None:
                raise KeyError(
                    '{} not found in {}'.format(self.sentence_vector_key, output_dict.keys())
                )
            vector_list.append(vector[0])

            text_info_list.append({
                'product_id': candidate['product_id'],
                'scene_id': candidate['scene_id'],
                'parent_node_id': self.node_id,
                'child_node_id': candidate['node_id'],
                'node_type': candidate['node_type'],
                'node_desc': candidate['node_desc'],
                'resource_id': candidate['resource_id'],
                'text': text,
                'text_for_vector': text_for_vector,
                'word_name': candidate['word_name'],
            })

        vector_list = np.array(vector_list, dtype=np.float32)
        self.index.add(vector_list)
        self.vector_list = vector_list
        self.text_info_list = text_info_list
        return

    def _split(self, text: str):
        for preprocess in self.preprocess_list:
            text = preprocess.process(text)

        outlst, _ = self.tokenizer.tokenize(text)

        for replacer in self.replacer_list:
            try:
                outlst = replacer.replace(outlst)
                outlst = outlst[0]
            except IndexError:
                return list()

        return outlst

    async def async_recall(self, context: RequestContext) -> List[dict]:
        logger.debug('{} recall: '.format(self.register_name))
        context.append_row("""### {} recall detail""".format(self.register_name))

        text = context.text

        tokens = self._split(text)
        text_for_vector = self.delimiter.join(tokens)
        logger.debug('text: {}, text_for_vector: {}'.format(text, text_for_vector))
        output_dict = self.predictor.predict_json(
            inputs={
                'task': {
                    'query': [
                        text_for_vector,
                    ]
                }
            },
        )
        return self.output_schema(output_dict, context)

    def recall(self, context: RequestContext) -> List[dict]:
        logger.debug('{} recall: '.format(self.register_name))
        context.append_row("""### {} recall detail""".format(self.register_name))

        text = context.text

        tokens = self._split(text)
        text_for_vector = self.delimiter.join(tokens)
        logger.debug('text: {}, text_for_vector: {}'.format(text, text_for_vector))
        output_dict = self.predictor.predict_json(
            inputs={
                'task': {
                    'query': [
                        text_for_vector,
                    ]
                }
            },
        )
        return self.output_schema(output_dict, context)

    def knn_score(self, vector, k_vector, k_label):
        label2score = defaultdict(float)
        label2max_prob = defaultdict(float)
        probs = [0] * len(k_vector)

        vector = torch.tensor(vector, dtype=torch.float32)

        # query 与每个 support 计算分数, 求和作为类别分数.
        if len(k_label) > 0:
            k_vector = torch.tensor(k_vector)
            # k_vector = torch.unsqueeze(k_vector, dim=1)

            distance = torch.square(vector - k_vector)
            distance = torch.sum(distance, dim=-1)
            logits = - torch.log(distance + 1e-7)
            probs = torch.softmax(logits, dim=-1)
            probs = probs.tolist()
            for l, p in zip(k_label, probs):
                label2score[l] += p

                if p > label2max_prob[l]:
                    label2max_prob[l] = p

        return label2score, label2max_prob, probs

    def output_schema(self, output_dict, context: RequestContext):
        vector = output_dict.get(self.sentence_vector_key)
        # vector: shape=(1, output_dim)
        if vector is None:
            raise KeyError(
                '{} not found in {}'.format(self.sentence_vector_key, output_dict.keys())
            )

        vector = np.array(vector, dtype=np.float32)
        D, I = self.index.search(vector, self.top_k)

        k_text, k_label, k_vector = list(), list(), list()
        for _, idx in zip(D[0], I[0]):
            text_info = self.text_info_list[idx]

            i_label = text_info['child_node_id']
            i_vector = self.vector_list[idx]

            k_text.append(text_info['text'])
            k_label.append(i_label)
            k_vector.append(i_vector)

        label2score, label2max_prob, probs = self.knn_score(
            vector=vector,
            k_label=k_label,
            k_vector=k_vector,
        )

        result = list()
        for idx1, idx2 in enumerate(I[0]):
            text_info = self.text_info_list[idx2]
            i_label = text_info['child_node_id']
            score = label2score.get(i_label, 0.0)

            prob = probs[idx1]
            max_score = label2max_prob.get(i_label, 1.0)
            score = score * prob / max_score

            result.append({
                'text': text_info['text'],
                'word_name': text_info['word_name'],
                'node_id': text_info['child_node_id'],
                'node_desc': text_info['node_desc'],
                'node_type': text_info['node_type'],
                'resource_id': text_info['resource_id'],
                'score': round(float(score), 4),
                'scorer_key': text_info['child_node_id'],
                'recaller': self.register_name,

            })

        context.append_table(result)
        return result


@Recall.register('jaccard')
class JaccardRecall(Recall):
    """字面召回"""
    @staticmethod
    def demo1():
        recall = JaccardRecall(
            candidates=ListCandidates(
                candidates=[
                    {'text': '暂时还不了'},
                    {'text': '不想还了'},
                    {'text': '没钱还'},
                    {'text': '农民没钱'},
                ]
            ),
            top_k=2
        )

        text = '可以先不还吧'

        result = recall.recall(RequestContext(text))
        print(result)
        return

    def __init__(self,
                 candidates: RecallCandidates,
                 top_k: int = 30,
                 preprocess_list: List[Preprocess] = None,
                 tokenizer: Tokenizer = None,
                 replacer_list: List[Replacer] = None,
                 filter_list: List[Filter] = None,
                 ):
        super().__init__()

        self.candidates = candidates
        self._candidates = self.candidates.get()
        self.top_k = top_k

        self.preprocess_list = preprocess_list or list()
        self.tokenizer = tokenizer or JiebaTokenizer()
        self.replacer_list = replacer_list or list()
        self.filter_list = filter_list or list()

        self.jaccard_candidates = self._init_jaccard_candidates(self._candidates)

    def _split(self, text: str):
        for preprocess in self.preprocess_list:
            text = preprocess.process(text)

        outlst, _ = self.tokenizer.tokenize(text)

        for replacer in self.replacer_list:
            outlst = replacer.replace(outlst)
            outlst = outlst[0]

        for filter_ in self.filter_list:
            outlst = filter_.filter(outlst)

        return outlst

    def _init_jaccard_candidates(self, candidates: List[dict]):
        result = list()
        for candidate in candidates:
            text = candidate['text']
            outlst = self._split(text)
            result.append({
                'tokens': outlst,
                'unique_tokens': set(outlst),
            })
        return result

    def recall(self, context: RequestContext) -> List[dict]:
        logger.debug('{} recall: '.format(self.register_name))
        context.append_row("""### {} recall detail""".format(self.register_name))

        text = context.text

        outlst = self._split(text)
        tokens = set(outlst)

        result = list()
        for candidate, jaccard_candidate in zip(self._candidates, self.jaccard_candidates):
            unique_tokens = jaccard_candidate['unique_tokens']
            intersection = tokens.intersection(unique_tokens)
            union = tokens.union(unique_tokens)
            score = len(intersection) / (len(union) + 1e-7)
            if score < 1e-7:
                continue

            result.append({
                **candidate,
                'score': score,
                'scorer_key': text,
                'recaller': self.register_name,

            })

        context.append_table(result)
        result = list(sorted(result, key=lambda x: x['score'], reverse=True))
        result = result[:self.top_k]

        context.append_table(result)
        return result


@Recall.register('each_node_one')
class EachNodeOneRecall(Recall):
    """
    从每个可选的分支中分别召回一条样本.
    """
    @staticmethod
    def demo1():
        from nxtech.nlpbot.node import RequestContext

        mysql_connect = MySqlConnect(
            host='10.20.251.13',
            port=3306,
            user='nx_prd',
            password='wm%msjngbtmheh3TdqYbmgg3s@nxprd230417',
            database='callbot_ppe',
        )
        recall = EachNodeOneRecall(
            product_id='callbot',
            scene_id='2vmbh5ex8u2q',
            node_id='51def72a-b086-4f6b-a3d5-45165d02dc10',
            env='ppe',
            mysql_connect=mysql_connect,
        )
        context = RequestContext(
            product_id='callbot',
            scene_id='2vmbh5ex8u2q',
            node_id='51def72a-b086-4f6b-a3d5-45165d02dc10',
            env='ppe',
            text='我需要啊',
        )
        candidates = recall.recall(context=context)

        node_type_list = [candidate['node_type'] for candidate in candidates]
        print(list(set(node_type_list)))

        node_id_list = [candidate['node_id'] for candidate in candidates]
        print(list(set(node_id_list)))

        return

    def __init__(self,
                 product_id: str,
                 scene_id: str,
                 node_id: str,
                 env: str,
                 mysql_connect: MySqlConnect = None,
                 resource_type_list: List[str] = None,
                 priority_list: List[str] = None,
                 ):
        super().__init__()

        self.product_id = product_id
        self.scene_id = scene_id
        self.node_id = node_id
        self.env = env
        self.mysql_connect = mysql_connect
        self.resource_type_list = resource_type_list or ['similar_question', 'white_regex', 'black_regex']
        self.priority_list = priority_list

        self.candidates = self._init_candidate()

    def _init_candidate(self):
        # 候选句子
        candidates: List[dict] = MysqlCandidatesIncludeIntentLib(
            product_id=self.product_id,
            scene_id=self.scene_id,
            node_id=self.node_id,
            mysql_connect=self.mysql_connect,
            resource_type_list=self.resource_type_list,
            priority_list=self.priority_list,
        ).get()

        result = list()
        unique_node_id = set()

        for candidate in candidates:
            node_id = candidate['node_id']
            node_desc = candidate['node_desc']
            node_type = candidate['node_type']
            resource_id = candidate['resource_id']

            if node_id in unique_node_id:
                continue
            unique_node_id.add(node_id)

            result.append({
                'text': 'dummy text',
                'node_id': node_id,
                'node_desc': node_desc,
                'node_type': node_type,
                'resource_id': resource_id,
                'score': 0.0,
                'scorer_key': node_id,
                'recaller': self.register_name,
            })
        return result

    def recall(self, context: RequestContext) -> List[dict]:
        logger.debug('{} recall: '.format(self.register_name))
        context.append_row("""### {} recall detail""".format(self.register_name))

        node_type_list = [candidate['node_type'] for candidate in self.candidates]
        node_desc_list = [candidate['node_desc'] for candidate in self.candidates]
        logger.debug('node_type_list: {}, node_desc_list: {}'.format(node_type_list, node_desc_list))

        context.append_table(self.candidates)
        return self.candidates


@Recall.register('random')
class RandomRecall(Recall):
    def __init__(self, candidates: List[dict], top_k: int = None):
        super().__init__()
        self.candidates = candidates
        self.top_k = top_k

    def recall(self, context: RequestContext) -> List[dict]:
        logger.debug('{} recall: '.format(self.register_name))

        candidates = self.candidates
        random.shuffle(candidates)
        if self.top_k is not None:
            candidates = candidates[:self.top_k]

        result = list()
        for candidate in candidates:
            candidate['recaller'] = self.register_name
            result.append(candidate)
        return result


def demo1():
    # JaccardRecall.demo1()
    # ElasticSearchRecall.demo1()
    EachNodeOneRecall.demo1()
    return


def demo2():
    recall = Recall.from_json(
        params={
            'type': 'jaccard',
            'candidates': {
                'type': 'list',
                'candidates': [
                     {'text': '暂时还不了'},
                     {'text': '不想还了'},
                     {'text': '没钱还'},
                     {'text': '可以分期吗'},
                ]
            },
            'top_k': 2,
            'tokenizer': {
                'type': 'jieba',
            },
            'replacer': {
                'type': 'synonym',
                'synonyms': [
                    {
                        'entity': '分期',
                        'synonyms': ['分气', '芬期']
                    },
                ]
            },
            'filter': {
                'type': 'stopwords',
                'stopwords': ['分期', '可以', '吗']

            }


        },
        global_params={}
    )
    print(recall)
    text = '可以先不还或者芬期吗'

    result = recall.recall(text)
    print(result)
    return


def demo3():
    config = {
        'product_id': 'callbot',
        'scene_id': 'x9o2opcr7n',
        'node_id': 'start_acwdg9pqjiozwqfgehtot',
        'mysql_connect': {
            'host': '10.20.251.13',
            'port': 3306,
            'user': 'callbot',
            'password': 'NxcloudAI2021!',
            'database': 'callbot_dev',
            'charset': 'utf8'
        }
    }
    recall = Recall.from_json(
        params={
            'type': 'jaccard',
            'candidates': {
                'type': 'mysql',
            },
            'top_k': 10,
            'tokenizer': {
                'type': 'jieba',
            },
        },
        global_params=config
    )
    print(recall)
    text = '可以先不还或者芬期吗'

    result = recall.recall(text)
    print(result)
    return


def demo4():
    es = ElasticSearchRecall.from_json(
        params={
            "update_es": False,
            "tokenizer": {
                "type": "jieba"
            },
        },
        global_params={
            "env": "tianxing",
            "product_id": "callbot",
            "scene_id": "x9o2opcr7n",
            "node_id": "start_acwdg9pqjiozwqfgehtot",
            "node_type": 1,
            "node_desc": "测试",
            "mysql_connect": {
                "host": "10.20.251.13",
                "port": 3306,
                "user": "callbot",
                "password": "NxcloudAI2021!",
                "database": "callbot_dev",
                "charset": "utf8"
            },
            "es_connect": {
                "hosts": ["10.20.251.8"],
                "http_auth": ["elastic", "ElasticAI2021!"],
                "port": 9200
            },
        }
    )

    text_list = [
        '现在不行',
        '没打算还钱',
    ]

    for text in text_list:
        result = es.recall(text)
        print(result)
    return


def demo5():
    es = ElasticSearchRecall.from_json(
        params={
            "tokenizer": {
                "type": "jieba"
            },
            "update_es": True
        },
        global_params={
            "env": "tianxing",
            "product_id": "callbot",
            "scene_id": "x9o2opcr7n",
            "node_id": "start_acwdg9pqjiozwqfgehtot",
            "node_type": 1,
            "node_desc": "测试",
            "mysql_connect": {
                "host": "10.20.251.13",
                "port": 3306,
                "user": "callbot",
                "password": "NxcloudAI2021!",
                "database": "callbot_dev",
                "charset": "utf8"
            },
            "es_connect": {
                "hosts": ["10.20.251.8"],
                "http_auth": ["elastic", "ElasticAI2021!"],
                "port": 9200
            },
        }
    )

    text_list = [
        '现在不行',
        '没打算还钱',
        '逾期不怕'
    ]

    for text in text_list:
        result = es.recall(text)
        print(result)
    return


if __name__ == '__main__':
    demo1()
    # demo2()
    # demo3()
    # demo4()
    # demo5()
