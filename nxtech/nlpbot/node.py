#!/usr/bin/python3
# -*- coding: utf-8 -*-
import logging
from typing import List, Tuple

from nxtech.nlpbot.recall import Recall
from nxtech.nlpbot.scorer import Scorer
from nxtech.common.params import Params
from nxtech.nlpbot.request_context import RequestContext

logger = logging.getLogger('nxtech')


class Searcher(Params):
    """
    Node 节点包含 Searcher,
    每个判别器单独包含 Recall, Scorer.
    给出的结果形式为 List[dict], 每个 dict 表示其匹配的分支.
    以及相应的分数.

    可专门设置一个 Searcher 用于判断答非所问.
    """
    def __init__(self,
                 node_id: str,
                 node_type: str,
                 node_desc: str,
                 recall: Recall,
                 scorer: Scorer,
                 top_k: int = 10,
                 ):
        super().__init__()

        self.node_id = node_id
        self.node_type = node_type
        self.node_desc = node_desc

        self.recall = recall
        self.scorer = scorer
        self.top_k = top_k

    def search(self, context: RequestContext) -> List[dict]:
        candidate_list = self.recall.recall(context)

        scores = self.scorer.score(
            context=context,
            candidate_list=candidate_list
        )
        return self.output_schema(candidate_list, scores)

    async def async_search(self, context: RequestContext) -> List[dict]:
        candidate_list = await self.recall.async_recall(context)

        scores = await self.scorer.async_score(
            context=context,
            candidate_list=candidate_list
        )
        return self.output_schema(candidate_list, scores)

    def output_schema(self, candidate_list, scores):
        """scores 可能为空列表, 则此函数会返回空列表. """
        result = list()
        for candidate, score in zip(candidate_list, scores):
            result.append({
                **candidate,
                **score
            })
        # 只返回有 scorer 签名的项.
        result = list(filter(lambda x: x['scorer'] is not None, result))
        result = list(sorted(result, key=lambda x: x['score'], reverse=True))[:self.top_k]
        return result


class Node(Params):
    def __init__(self,
                 product_id: str,
                 scene_id: str,
                 env: str,
                 node_id: str,
                 node_type: str,
                 node_desc: str,
                 searcher_list: List[Searcher],
                 top_k: int = 10,
                 valid_score_threshold: float = 0.35,
                 ):
        super().__init__()
        self.product_id = product_id
        self.scene_id = scene_id
        self.env = env
        self.node_id = node_id
        self.node_type = node_type
        self.node_desc = node_desc
        self.top_k = top_k
        self.valid_score_threshold = valid_score_threshold

        self.searcher_list = searcher_list

    def recommend(self, context: RequestContext) -> RequestContext:
        context.append_row("""## intent search detail""")

        if len(context.text) == 0:
            return context

        result = list()
        for searcher in self.searcher_list:
            result = searcher.search(context)
            if len(result) != 0:
                break

        context.result = result
        return context

    async def async_recommend(self, context: RequestContext) -> RequestContext:
        context.append_row("""## intent search detail""")

        if len(context.text) == 0:
            return context

        for searcher in self.searcher_list[:-1]:
            searcher_result = await searcher.async_search(context)
            valid_result = [match for match in searcher_result if match['score'] > self.valid_score_threshold]
            # 前面的 search 得到符合条件的结果时, 跳出.
            if len(valid_result) != 0:
                result = searcher_result
                break
        else:
            # 最后一个 searcher 保底.
            searcher = self.searcher_list[-1]
            result = await searcher.async_search(context)

        context.result = result
        return context


def demo1():
    global_config = {
        "product_id": "callbot",
        "scene_id": "mzkev2wvfz",
        "env": "ppe",
        "node_id": "1d48edc8-f7cb-46c3-9261-908a270062b2",
        "node_type": 1,
        "node_desc": "施压",
        "mysql_connect": {
            "host": "10.20.251.13",
            "port": 3306,
            "user": "callbot",
            "password": "NxcloudAI2021!",
            "database": "callbot_ppe",
            "charset": "utf8"
        },
        "es_connect": {
            "hosts": [
                "10.20.251.8"
            ],
            "http_auth": [
                "elastic",
                "ElasticAI2021!"
            ],
            "port": 9200
        },
    }

    node_config = {
        "searcher_list": [
            {
                "recall": {
                    "type": "node_recall"
                },
                "scorer": {
                    "type": "regex",
                    "filename": "server/callbot_nlp_server/config/scenes/mzkev2wvfz/regex_config.json",
                    "preprocess_list": [
                        {
                            "type": "string2number"
                        }
                    ]
                }
            },
            {
                "recall": {
                    "type": "faiss",
                    "http_allennlp_predictor": {
                        "host_port": "http://10.20.251.5:1070"
                    },
                    "tokenizer": {
                        "type": "jieba"
                    },
                    "top_k": 30,
                    "dim": 128,
                    "preprocess_list": [
                        {
                            "type": "do_lowercase"
                        },
                        {
                            "type": "strip"
                        }
                    ]
                },
                "scorer": {
                    "type": "scale",
                    "stages": [0.95, 0.85, 0.79, 0.72, 0.50],
                    "scores": [0.85, 0.75, 0.50, 0.35, 0.20]
                }
            }
        ]
    }

    import pickle

    node = Node.from_json(node_config, global_config)
    print(node)

    with open('test.pkl', 'wb') as f:
        pickle.dump(node, f)
    return


if __name__ == '__main__':
    demo1()
