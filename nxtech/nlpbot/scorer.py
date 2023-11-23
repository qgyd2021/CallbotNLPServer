# -*- encoding=UTF-8 -*-
from collections import defaultdict
import json
import logging
import os
import re
import sys
from typing import Dict, List, Tuple

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))

import editdistance
import graphviz
from gensim.models import Word2Vec
from gensim import matutils
import numpy as np
import pandas as pd
from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import model_selection
from sklearn.utils.validation import check_is_fitted

from nxtech.common.params import Params
from nxtech.database.mysql_connect import MySqlConnect
from nxtech.nlpbot.candidate import RecallCandidates, MysqlCandidatesIncludeIntentLib
from nxtech.nlpbot.misc import Preprocess, Replacer, Filter
from nxtech.nlpbot.model import HttpAllenNlpPredictor, TextClassifierPredictor, BasicIntentClassifierPredictor
from nxtech.nlpbot.request_context import RequestContext
from nxtech.nlpbot.tokenizer import Tokenizer, JiebaTokenizer
from nxtech.table_lib.t_dialog_node_info import TDialogNodeInfo
from project_settings import project_path
from toolbox.design_patterns.singleton import ParamsSingleton
from toolbox.math import score_transform
from toolbox.sklearn.feature_extraction.text import TfidfTransformer

logger = logging.getLogger("nxtech")


class Scorer(Params):
    """
    给定两个文本, Scorer 为其打一个 0-1 之前的分数.

    需要一个基于规则的打分器, 不然都要训练模型, 没那么多数据.
    """
    def __init__(self):
        super().__init__()

    def score(self, context: RequestContext, candidate_list: List[dict]) -> List[dict]:
        """
        因为使用了正则匹配的分类器来打分. 因此 Scorer 会在结果中添加 scorer 的 key 来签名. 没有匹配正则表达式的, 则不会有签名.
        而一般的相似度打分器, 都会有签名的.

        结果中的 label 用于给出一些额外信息, 如分类器打分时, 会给出其标签, 正则匹配时, 会给出其匹配的模式.
        :param text: 当前文本
        :param candidate_list: 包含字典的列表. 候选的文本, 字典中至少包含 `text`, `scorer_key` 两个 key.
        `text` 是候选文本. `scorer_key` 用来为 `text` 分组,
        在 ScorerWrapper 中会根据 `scorer_key` 的不同选用不同的算分器.
        :return: 返回包含字典的列表, 字典中至少有 `score`, `metric` 两个 key.
        `score` 是归一化到 0-1 之间的相似度分数; `metric` 是原始的度量值.
        示例:
        [
            {
                'score': 0.9123,
                'metric': 1.2314,
                'scorer': scorer_name,
                'label': None
            },
            ...
        ]
        """
        raise NotImplementedError

    async def async_score(self, context: RequestContext, candidate_list: List[dict]):
        return self.score(context, candidate_list)


@Scorer.register('zero')
class ZeroScorer(Scorer):
    def score(self, context: RequestContext, candidate_list: List[dict]) -> List[dict]:
        logger.debug('{} score: '.format(self.register_name))
        context.append_row("""### {} score detail""".format(self.register_name))

        l = len(candidate_list)

        result = [{
            'score': 0.0,
            'metric': 0.0,
            'scorer': self.register_name,

        }] * l

        context.append_table(result)
        return result


@Scorer.register('scale')
class ScaleScorer(Scorer):
    """只是将 recall 的分数进行 scale 缩放. """
    def __init__(self,
                 stages: List[float] = None,
                 scores: List[float] = None,
                 ):
        super().__init__()

        self.stages = stages or [1.0, 0.95, 0.85, 0.75, 0.65, 0.5, 0.0]
        self.scores = scores or [1.0, 0.85, 0.75, 0.35, 0.25, 0.1, 0.0]

    def score(self, context: RequestContext, candidate_list: List[dict]) -> List[dict]:
        logger.debug('{} score: '.format(self.register_name))
        context.append_row("""### {} score detail""".format(self.register_name))

        result = list()
        for candidate in candidate_list:
            metric = candidate['score']
            score = score_transform(
                x=metric,
                stages=self.stages,
                scores=self.scores
            )
            result.append({
                'score': round(score, 4),
                'metric': metric,
                'scorer': self.register_name,
            })

        context.append_table(result)
        return result


@Scorer.register('black_white_regex')
class BlackWhiteRegexScorer(Scorer):
    """
    ?(white).*?(需要).*?(black).*?(不需要|不用|不要|不贷款).*??

    # black 可为空.
    ?(white).*?(需要).*?(black)?

    # 无效, white 必须有.
    ?(white)(black).*?(不需要|不用|不要|不贷款).*??

    """

    pattern = r'\?\(white\)(.+)\(black\)(.*)\?'

    @staticmethod
    def demo1():
        from nxtech.nlpbot.recall import EachNodeOneRecall

        scene_id = '9qtb38k09cpb'
        node_id = '51def72a-b086-4f6b-a3d5-45165d02dc10'

        mysql_connect = MySqlConnect(
            host='10.20.251.13',
            port=3306,
            user='callbot',
            password='NxcloudAI2021!',
            database='callbot_ppe',
        )
        recall = EachNodeOneRecall(
            product_id='callbot',
            scene_id=scene_id,
            node_id=node_id,
            env='ppe',
            mysql_connect=mysql_connect,
        )
        scorer = BlackWhiteRegexScorer(
            product_id='callbot',
            scene_id=scene_id,
            node_id=node_id,
            env='ppe',
            mysql_connect=mysql_connect,
            main_take_precedence=False,
        )

        context = RequestContext(
            product_id='callbot',
            scene_id=scene_id,
            node_id=node_id,
            env='ppe',
            text='不需要啊',
        )
        candidate_list = recall.recall(context)
        result = scorer.score(context=context, candidate_list=candidate_list)
        print(result[0]['resource_id'])
        return

    def __init__(self,
                 product_id: str,
                 scene_id: str,
                 node_id: str,
                 env: str,
                 mysql_connect: MySqlConnect,
                 preprocess_list: List[Preprocess] = None,
                 main_take_precedence: bool = True,
                 ):
        super().__init__()
        self.product_id = product_id
        self.scene_id = scene_id
        self.node_id = node_id
        self.mysql_connect = mysql_connect
        self.main_take_precedence = main_take_precedence

        self.preprocess_list = preprocess_list or list()
        self.env = env

        self.regex_list = self._init_regex_list()

    def _init_regex_list(self):
        candidates: List[dict] = MysqlCandidatesIncludeIntentLib(
            product_id=self.product_id,
            scene_id=self.scene_id,
            node_id=self.node_id,
            mysql_connect=self.mysql_connect,
            # 3 对应白名单正则
            resource_type_list=['white_regex'],
        ).get()

        candidates = sorted(candidates, key=lambda x: x['node_id'], reverse=not self.main_take_precedence)
        candidates = sorted(candidates, key=lambda x: x['node_type'], reverse=not self.main_take_precedence)

        regex_list = list()
        for candidate in candidates:
            resource_id = candidate['resource_id']
            resource_type = candidate['resource_type']
            node_id = candidate['node_id']
            node_desc = candidate['node_desc']
            node_type = candidate['node_type']
            text = candidate['text']

            text = str(text).strip()

            match = re.match(self.pattern, text, flags=re.IGNORECASE)
            if match is None:
                logger.warning('invalid regular expression: {}'.format(text))
                continue
            white_pattern = match.group(1)
            # black_pattern 可以为空字符串.
            black_pattern = match.group(2)
            try:
                white_pattern = re.compile(white_pattern, flags=re.IGNORECASE)
            except Exception as e:
                logger.warning('invalid regular expression, white_pattern: {}'.format(white_pattern))
                continue

            if len(black_pattern) == 0:
                black_pattern = None
            else:
                try:
                    black_pattern = re.compile(black_pattern, flags=re.IGNORECASE)
                except Exception as e:
                    logger.warning('invalid regular expression, black_pattern: {}'.format(black_pattern))
                    continue

            regex_list.append({
                'node_id': node_id,
                'node_desc': node_desc,
                'node_type': node_type,
                'resource_id': resource_id,
                'resource_type': resource_type,
                'original_pattern': text,

                'white_pattern': white_pattern,
                'black_pattern': black_pattern,
            })

        return regex_list

    def score(self, context: RequestContext, candidate_list: List[dict] = None):
        logger.debug('{} score: '.format(self.register_name))
        context.append_row("""### {} score detail""".format(self.register_name))

        text = context.text

        for preprocess in self.preprocess_list:
            text = preprocess.process(text)

        node_desc_list = [candidate['node_desc'] for candidate in candidate_list]
        node_desc_list = list(set(node_desc_list))
        print('node_id_list: {}'.format([candidate['node_id'] for candidate in candidate_list]))
        logger.debug('node_desc_list: {}'.format(node_desc_list))

        result = list()
        for regex in self.regex_list:
            node_id = regex['node_id']
            node_desc = regex['node_desc']
            node_type = regex['node_type']
            resource_id = regex['resource_id']
            original_pattern = regex['original_pattern']

            white_pattern = regex['white_pattern']
            black_pattern = regex['black_pattern']

            if node_desc not in node_desc_list:
                logger.debug('continued, regex: {}'.format(regex))
                continue
            else:
                logger.debug('regex: {}'.format(regex))

            if black_pattern and re.match(black_pattern, text):
                continue

            if white_pattern and re.match(white_pattern, text):
                result.append({
                    'text': original_pattern,
                    'node_id': node_id,
                    'node_desc': node_desc,
                    'node_type': node_type,
                    'resource_id': resource_id,

                    'score': 1.0,
                    'metric': 1.0,
                    'scorer': self.register_name,
                    'label': white_pattern
                })
                break

        context.append_table(result)
        return result


@Scorer.register('intent_and_entity')
class IntentAndEntityScorer(Scorer):
    """
    # 意图标签: intent, 识别方法: basic.
    # 实体识别: entity, 识别方法: regex.
    text = '?[{
                "usage": "intent_and_entity",
                "intent_label": "否定(不需要)",
                "intent_method": "basic",
                "entity_label": ".*不需要.*",
                "entity_method": "regex"
            }]'

    """

    pattern = r'\?\[(.+)\]'

    @staticmethod
    def demo1():
        from nxtech.nlpbot.recall import EachNodeOneRecall
        mysql_connect = MySqlConnect(
            # gz
            # host='10.20.251.13',
            # hk
            host='10.52.66.41',
            port=3306,
            user='callbot',
            password='NxcloudAI2021!',
            database='callbot_ppe',
        )
        recall = EachNodeOneRecall(
            product_id='callbot',
            scene_id='ad6e2oq406',
            node_id='51def72a-b086-4f6b-a3d5-45165d02dc10',
            env='ppe',
            mysql_connect=mysql_connect,
        )

        # curl -X POST http://127.0.0.1:13070/BasicIntent -d '{"key": "chinese", "text": "C++的BERT分词器实现"}'
        # curl -X POST http://10.52.66.97:13070/BasicIntent -d '{"key": "chinese", "text": "C++的BERT分词器实现"}'
        scorer = IntentAndEntityScorer(
            product_id='callbot',
            scene_id='ad6e2oq406',
            node_id='51def72a-b086-4f6b-a3d5-45165d02dc10',
            env='ppe',
            mysql_connect=mysql_connect,
            intent_method={
                'basic': BasicIntentClassifierPredictor(
                    url='http://10.52.66.97:13070/BasicIntent',
                    language='chinese'
                )
            },
            sort_reverse=True,
        )

        context = RequestContext(
            product_id='callbot',
            scene_id='ad6e2oq406',
            node_id='51def72a-b086-4f6b-a3d5-45165d02dc10',
            env='ppe',
            text='没有这个需要',
        )
        candidate_list = recall.recall(context)
        result = scorer.score(context=context, candidate_list=candidate_list)

        print(len(result))
        # print(result[0]['resource_id'])
        return

    @staticmethod
    def demo2():
        import asyncio
        from nxtech.nlpbot.recall import EachNodeOneRecall
        mysql_connect = MySqlConnect(
            # gz
            # host='10.20.251.13',
            # hk
            host='10.52.66.41',
            port=3306,
            user='callbot',
            password='NxcloudAI2021!',
            database='callbot_ppe',
        )
        recall = EachNodeOneRecall(
            product_id='callbot',
            scene_id='ad6e2oq406',
            node_id='51def72a-b086-4f6b-a3d5-45165d02dc10',
            env='ppe',
            mysql_connect=mysql_connect,
        )

        # curl -X POST http://127.0.0.1:13070/BasicIntent -d '{"key": "chinese", "text": "C++的BERT分词器实现"}'
        # curl -X POST http://10.52.66.97:13070/BasicIntent -d '{"key": "chinese", "text": "C++的BERT分词器实现"}'
        scorer = IntentAndEntityScorer(
            product_id='callbot',
            scene_id='ad6e2oq406',
            node_id='51def72a-b086-4f6b-a3d5-45165d02dc10',
            env='ppe',
            mysql_connect=mysql_connect,
            intent_method={
                'basic': BasicIntentClassifierPredictor(
                    url='http://10.52.66.97:13070/BasicIntent',
                    language='chinese'
                )
            },
            sort_reverse=True,
        )

        context = RequestContext(
            product_id='callbot',
            scene_id='ad6e2oq406',
            node_id='51def72a-b086-4f6b-a3d5-45165d02dc10',
            env='ppe',
            # text='没有这个需要',
            text=r'\u4f60\u597d, \u4f60\u597d\u5417? \u55ef. ',
        )
        candidate_list = recall.recall(context)

        async def task():
            result = await scorer.async_score(context=context, candidate_list=candidate_list)
            print(len(result))
            # print(result[0]['resource_id'])
            return

        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            task()
        )
        return

    @staticmethod
    def demo3():
        import asyncio
        from toolbox.aiohttp import async_requests

        url = 'http://10.52.66.97:13070/BasicIntent'

        headers = {
            "Content-Type": "application/json"
        }
        # data = {'key': 'chinese', 'text': '你好, 你好吗? 嗯. '}
        data = {'key': 'chinese', 'text': r'\u4f60\u597d, \u4f60\u597d\u5417? \u55ef. '}

        async def task():
            text, status_code = await async_requests.requests(
                'POST',
                url,
                headers=headers,
                data=json.dumps(data, ensure_ascii=False),
                # data=json.dumps(data),
                timeout=2,
            )
            if status_code == 200:
                # js = json.loads(text)
                # print(js)
                print(text)
            else:
                print(text)
            return

        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            task()
        )
        return

    def __init__(self,
                 product_id: str,
                 scene_id: str,
                 node_id: str,
                 env: str,
                 mysql_connect: MySqlConnect,
                 intent_method: Dict[str, TextClassifierPredictor],
                 entity_method: Dict[str, HttpAllenNlpPredictor] = None,
                 preprocess_list: List[Preprocess] = None,
                 sort_reverse: bool = False
                 ):
        super(IntentAndEntityScorer, self).__init__()
        self.product_id = product_id
        self.scene_id = scene_id
        self.node_id = node_id
        self.mysql_connect = mysql_connect

        self.intent_method = intent_method
        self.entity_method = entity_method or dict()

        self.sort_reverse = sort_reverse

        self.preprocess_list = preprocess_list or list()
        self.env = env

        self.branch_to_intent_and_entity_list = self._init_branch_to_intent_and_entity_list()

    def _init_branch_to_intent_and_entity_list(self):
        candidates: List[dict] = MysqlCandidatesIncludeIntentLib(
            product_id=self.product_id,
            scene_id=self.scene_id,
            node_id=self.node_id,
            mysql_connect=self.mysql_connect,
            resource_type_list=['white_regex'],
        ).get()

        branch_to_intent_and_entity_list: Dict[str, List[dict]] = dict()
        for candidate in candidates:
            resource_id = candidate['resource_id']
            resource_type = candidate['resource_type']
            node_id = candidate['node_id']
            node_desc = candidate['node_desc']
            node_type = candidate['node_type']
            text = candidate['text']

            match = re.match(self.pattern, text, flags=re.IGNORECASE)
            if match is None:
                # logger.warning('invalid intent and entity expression: {}'.format(text))
                continue

            json_string = match.group(1)
            json_string = '{{{}}}'.format(json_string)

            js = json.loads(json_string)
            usage = js.get('usage')
            if usage != 'intent_and_entity':
                continue

            if node_id not in branch_to_intent_and_entity_list:
                branch_to_intent_and_entity_list[node_id] = list()

            branch_to_intent_and_entity_list[node_id].append({
                'node_id': node_id,
                'node_desc': node_desc,
                'node_type': node_type,
                'resource_id': resource_id,
                'resource_type': resource_type,
                'original_pattern': text,

                **js
            })
        return branch_to_intent_and_entity_list

    def score(self, context: RequestContext, candidate_list: List[dict]) -> List[dict]:
        logger.debug('{} score: '.format(self.register_name))
        context.append_row("""### {} score detail""".format(self.register_name))

        text = context.text

        for preprocess in self.preprocess_list:
            text = preprocess.process(text)

        # predictor = self.intent_method['basic']
        # outputs = predictor.predict(text=text)
        # outputs = predictor.predict(text='你好')
        # print(outputs)

        candidate_list = sorted(candidate_list, key=lambda x: x['node_id'], reverse=self.sort_reverse)
        candidate_list = sorted(candidate_list, key=lambda x: x['node_type'], reverse=self.sort_reverse)

        logger.debug('candidate num: {}'.format(len(candidate_list)))

        result = list()
        for candidate in candidate_list:
            node_id = candidate['node_id']

            intent_and_entity_list: List[dict] = self.branch_to_intent_and_entity_list.get(node_id)

            if intent_and_entity_list is None:
                continue

            logger.debug('{} intent_and_entity for node_id {}'.format(len(intent_and_entity_list), node_id))

            for intent_and_entity in intent_and_entity_list:
                node_id = intent_and_entity['node_id']
                node_desc = intent_and_entity['node_desc']
                node_type = intent_and_entity['node_type']
                resource_id = intent_and_entity['resource_id']
                original_pattern = intent_and_entity['original_pattern']

                # intent
                intent_label = intent_and_entity['intent_label']
                intent_method = intent_and_entity.get('intent_method', 'basic')
                intent_min_score = intent_and_entity.get('intent_min_score', 0.0)

                # entity
                black_entity_label = intent_and_entity.get('black_entity_label')
                black_entity_method = intent_and_entity.get('black_entity_method', 'regex')

                white_entity_label = intent_and_entity.get('white_entity_label')
                white_entity_method = intent_and_entity.get('white_entity_method', 'regex')

                # black entity (不能匹配到黑的)
                if black_entity_label is not None and black_entity_method is not None:
                    if black_entity_method == 'regex':
                        match = re.match(black_entity_label, text, flags=re.IGNORECASE)
                        if match is not None:
                            continue
                    elif black_entity_method in self.entity_method.keys():
                        raise NotImplementedError
                    else:
                        logger.warning('invalid entity_method: {}'.format(black_entity_method))
                        continue

                # white entity (必须匹配到白的)
                if white_entity_label is not None and white_entity_method is not None:
                    if white_entity_method == 'regex':
                        match = re.match(white_entity_label, text, flags=re.IGNORECASE)
                        if match is None:
                            continue
                    elif white_entity_method in self.entity_method.keys():
                        raise NotImplementedError
                    else:
                        logger.warning('invalid entity_method: {}'.format(white_entity_method))
                        continue

                # intent
                if intent_method not in self.intent_method.keys():
                    continue

                predictor = self.intent_method[intent_method]
                outputs = predictor.predict(text=text)

                label = outputs['label']
                prob = outputs['prob']

                logger.debug('text: {}, label: {}, prob: {}'.format(text, label, prob))

                if prob < intent_min_score:
                    continue

                suffix = '_{}'.format(intent_label)
                if not label.endswith(suffix):
                    logger.debug('not endswith {}'.format(suffix))
                    continue

                metric = round(prob, 4)

                result.append({
                    'text': original_pattern,
                    'node_id': node_id,
                    'node_desc': node_desc,
                    'node_type': node_type,
                    'resource_id': resource_id,

                    'score': metric,
                    'metric': metric,
                    'scorer': self.register_name,
                    'label': intent_label
                })

        context.append_table(result)
        return result

    async def async_score(self, context: RequestContext, candidate_list: List[dict]):
        logger.debug('{} score: '.format(self.register_name))
        context.append_row("""### {} score detail""".format(self.register_name))

        text = context.text

        for preprocess in self.preprocess_list:
            text = preprocess.process(text)

        # predictor = self.intent_method['basic']
        # outputs = predictor.predict(text=text)
        # outputs = predictor.predict(text='你好')
        # print(outputs)

        candidate_list = sorted(candidate_list, key=lambda x: x['node_id'], reverse=self.sort_reverse)
        candidate_list = sorted(candidate_list, key=lambda x: x['node_type'], reverse=self.sort_reverse)

        logger.debug('candidate num: {}'.format(len(candidate_list)))

        result = list()
        for candidate in candidate_list:
            node_id = candidate['node_id']

            intent_and_entity_list: List[dict] = self.branch_to_intent_and_entity_list.get(node_id)

            if intent_and_entity_list is None:
                continue

            logger.debug('{} intent_and_entity for node_id {}'.format(len(intent_and_entity_list), node_id))

            for intent_and_entity in intent_and_entity_list:
                node_id = intent_and_entity['node_id']
                node_desc = intent_and_entity['node_desc']
                node_type = intent_and_entity['node_type']
                resource_id = intent_and_entity['resource_id']
                original_pattern = intent_and_entity['original_pattern']

                # intent
                intent_label = intent_and_entity['intent_label']
                intent_method = intent_and_entity.get('intent_method', 'basic')
                intent_min_score = intent_and_entity.get('intent_min_score', 0.0)

                # entity
                black_entity_label = intent_and_entity.get('black_entity_label')
                black_entity_method = intent_and_entity.get('black_entity_method', 'regex')

                white_entity_label = intent_and_entity.get('white_entity_label')
                white_entity_method = intent_and_entity.get('white_entity_method', 'regex')

                # black entity (不能匹配到黑的)
                if black_entity_label is not None and black_entity_method is not None:
                    if black_entity_method == 'regex':
                        match = re.match(black_entity_label, text, flags=re.IGNORECASE)
                        if match is not None:
                            continue
                    elif black_entity_method in self.entity_method.keys():
                        raise NotImplementedError
                    else:
                        logger.warning('invalid entity_method: {}'.format(black_entity_method))
                        continue

                # white entity (必须匹配到白的)
                if white_entity_label is not None and white_entity_method is not None:
                    if white_entity_method == 'regex':
                        match = re.match(white_entity_label, text, flags=re.IGNORECASE)
                        if match is None:
                            continue
                    elif white_entity_method in self.entity_method.keys():
                        raise NotImplementedError
                    else:
                        logger.warning('invalid entity_method: {}'.format(white_entity_method))
                        continue

                # intent
                if intent_method not in self.intent_method.keys():
                    continue

                predictor = self.intent_method[intent_method]
                outputs = await predictor.async_predict(text=text)

                label = outputs['label']
                prob = outputs['prob']

                logger.debug('text: {}, label: {}, prob: {}'.format(text, label, prob))

                if prob < intent_min_score:
                    continue

                suffix = '_{}'.format(intent_label)
                if not label.endswith(suffix):
                    logger.debug('not endswith {}'.format(suffix))
                    continue

                metric = round(prob, 4)

                result.append({
                    'text': original_pattern,
                    'node_id': node_id,
                    'node_desc': node_desc,
                    'node_type': node_type,
                    'resource_id': resource_id,

                    'score': metric,
                    'metric': metric,
                    'scorer': self.register_name,
                    'label': intent_label
                })

        context.append_table(result)
        return result


@Scorer.register('decision_tree')
class DecisionTreeClassifierScorer(Scorer):
    """
    现存的问题:
    1. 无法识别无关领域的句子. 决策树一定会输出一个类别.
    解决办法: 决策树在输出概率, 提高对概率的要求.
    2. 能否让决策树放弃对样本打标记, 让后续的语义模型来做判断.
    解决办法:
    """
    def __init__(self,
                 product_id: str,
                 scene_id: str,
                 node_id: str,
                 candidates: RecallCandidates,
                 tokenizer: Tokenizer,
                 preprocess_list: List[Preprocess] = None,
                 replacer_list: List[Replacer] = None,
                 filter_list: List[Filter] = None,
                 label_map: Dict[str, str] = None,
                 export_report: bool = False,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 export_path: str = None
                 ):
        """
        :param label_map: 模板中有 `用户答非所问1`, `用户答非所问2`, `用户答非所问3` 三个不同的节点,
        我们做分类, 将其都映射为 `用户答非所问1`
        """
        super().__init__()

        self.product_id = product_id
        self.scene_id = scene_id
        self.node_id = node_id
        self.candidates = candidates
        self.tokenizer = tokenizer
        self.preprocess_list = preprocess_list or list()
        self.replacer_list = replacer_list or list()
        self.filter_list = filter_list or list()
        self.label_map = label_map or self._init_label_map()
        self.export_report = export_report

        if export_path is None:
            self.export_path = os.path.join(project_path, 'temp', self.register_name)
        else:
            self.export_path = os.path.join(project_path, export_path)

        self.vectorizer = CountVectorizer(
            # max_features=500,
            max_df=0.9,
            # min_df=0.01,
            binary=True
        )

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.tree = DecisionTreeClassifier(
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=0,
            # class_weight='balanced'
        )

        self.classes = None
        self._init_tree()

    def _init_label_map(self) -> dict:
        if isinstance(self.candidates, MysqlCandidatesIncludeIntentLib):
            t_dialog_node_info = TDialogNodeInfo(
                scene_id=self.scene_id,
                mysql_connect=self.candidates.mysql_connect
            )
            df = t_dialog_node_info.data
            df = df[df['product_id'] == self.candidates.product_id]
            df = df[df['scene_id'] == self.candidates.scene_id]

            node_desc2node_id = dict()
            for i, row in df.iterrows():
                node_id = row['node_id']
                node_desc = row['node_desc']
                node_desc = node_desc.lower().strip()
                if node_desc in (
                    'user is unclear 1',
                    'user is unclear 2',
                    'user is unclear 3',
                    '用户不清楚1',
                    '用户不清楚2',
                    '用户不清楚3',
                ):
                    node_desc2node_id[node_desc] = node_id

            label_map = dict()
            for k, v in [
                ('user is unclear 2', 'user is unclear 1'),
                ('user is unclear 3', 'user is unclear 1'),
                ('用户不清楚2', '用户不清楚1'),
                ('用户不清楚3', '用户不清楚1'),
            ]:
                k = node_desc2node_id.get(k)
                v = node_desc2node_id.get(v)
                if k is None or v is None:
                    continue
                label_map[k] = v
            result = label_map
        else:
            result = dict()
        return result

    def _export_report(self, x, y, candidates, tokens_data):
        """导出测试报告"""
        export_path = os.path.join(self.export_path, self.product_id, self.scene_id, self.node_id)
        os.makedirs(export_path, exist_ok=True)

        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.25, random_state=0)
        tree = DecisionTreeClassifier(random_state=0)
        tree.fit(x_train, y_train)

        train_accuracy = tree.score(x_train, y_train)
        validate_accuracy = tree.score(x_test, y_test)

        result = {
            'train_accuracy': train_accuracy,
            'validate_accuracy': validate_accuracy,
        }

        filename = os.path.join(export_path, 'metrics.json')
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        predicts = tree.predict(x)

        result = list()
        for predict, candidate, tokens in zip(predicts, candidates, tokens_data):
            text = candidate['text']
            label = candidate['node_id']

            if label in self.label_map:
                label = self.label_map[label]

            result.append({
                'text': text,
                'label': label,
                'tokens': json.dumps(tokens, ensure_ascii=False),
                'predict': predict,
                'flag': 1 if predict == label else 0,
            })

        filename = os.path.join(export_path, 'result.csv')
        result = pd.DataFrame(result)
        result.to_csv(filename, index=False, encoding='utf_8_sig')

        # 可视化决策树
        feature_names = [k for k, v in sorted(self.vectorizer.vocabulary_.items(), key=lambda x: x[1])]
        dot_data = export_graphviz(
            self.tree,
            class_names=self.classes,
            feature_names=feature_names,
            out_file=None)
        graph = graphviz.Source(dot_data)

        filename = os.path.join(export_path, 'DecisionTreeClassifierScorer')
        graph.render(filename)
        return

    def _init_tree(self):
        candidates = self.candidates.get()
        x = list()
        y = list()
        tokens_data = list()
        for candidate in candidates:
            text = candidate['text']
            if len(text.strip()) == 0:
                continue
            label = candidate['node_id']
            if label in self.label_map:
                label = self.label_map[label]
            tokens = self._split(text)

            tokens_data.append(tokens)
            x.append(' '.join(tokens))
            y.append(label)

        x = self.vectorizer.fit_transform(x)
        x = x.toarray()

        self.tree.fit(x, y)
        self.classes = list(self.tree.classes_)

        if self.export_report:
            self._export_report(
                x=x,
                y=y,
                candidates=candidates,
                tokens_data=tokens_data,
            )
        return

    def text2array(self, text: str):
        tokens = self._split(text)
        x = self.vectorizer.transform([' '.join(tokens)])
        x = x.toarray()
        return x

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

    def predict_proba(self, x):
        """
        DecisionTreeClassifier 不使用 class_weight 时,
        export_graphviz 决策树可视化 PDF 中, 每个节点会显示, 分类到当前节点时, 各类别样本的数量.
        如: value = [36, 41, 224, 71, 48, 49, 94, 224, 30, 75]

        使用 class_weight 时, 可视化中的 value 会有小数, 各类别的开始时的均衡样本数量是一样的.
        但是使用 class_weight='balanced' 时, 即使精准匹配的文本, 也可能得不到很高的 value 值.
        这里不采用.

        在 tree.predict_proba 可以看出其原理.
        以下部分为 copy 其代码.

        默认代码是按各类样本的数量计算其概率.
        但是我们想, 只在确定性能高时, 才输出高概率.
        因此, 将小于 threshold 的概率都认为不明确.
        """
        check_is_fitted(self.tree, 'tree_')
        x = self.tree._validate_X_predict(x, check_input=True)
        proba = self.tree.tree_.predict(x)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer
        return proba

    def score(self, context: RequestContext, candidate_list: List[dict]) -> List[dict]:
        logger.debug('{} score: '.format(self.register_name))
        context.append_row("""### {} score detail""".format(self.register_name))

        text = context.text

        array = self.text2array(text)

        result = list()
        if np.sum(array) == 0:
            for _ in candidate_list:
                result.append({
                    'score': 0.0,
                    'metric': 0.0,
                    'scorer': self.register_name,
                    'label': None
                })
        else:
            # probs = self.tree.predict_proba(array)
            probs = self.predict_proba(array)
            for candidate in candidate_list:
                node_id = candidate['node_id']
                if node_id in self.label_map:
                    node_id = self.label_map[node_id]
                idx = self.classes.index(node_id)
                prob = probs[0][idx]
                prob = score_transform(
                    x=prob,
                    # stages=[0.9, 0.8, 0.7, 0.5, 0.3],
                    # scores=[0.7, 0.4, 0.3, 0.2, 0.1],
                    stages=[1.0, 0.8, 0.70, 0.50, 0.4, 0.3, 0.0],
                    scores=[1.0, 0.8, 0.65, 0.35, 0.2, 0.1, 0.0],
                )

                result.append({
                    'score': round(prob, 4),
                    'metric': round(prob, 4),
                    'scorer': self.register_name,
                    'label': None
                })

        context.append_table(result)
        return result


@Scorer.register('tfidf')
class TfIdfScorer(Scorer):
    """
    计算各文档的 TF-IDF 值.
    对用户查询中的词, 在 TF-IDF 中查找对应的 tf-idf 值, 求和. 做为该查询与文档的相似度 metric
    """
    def __init__(self,
                 candidates: RecallCandidates,
                 tokenizer: Tokenizer,
                 preprocess_list: List[Preprocess] = None,
                 replacer_list: List[Replacer] = None,
                 filter_list: List[Filter] = None,
                 label_map: Dict[str, str] = None,
                 max_df: float = 0.8,
                 stages: List[float] = None,
                 scores: List[float] = None,
                 ):
        super().__init__()
        self.candidates = candidates
        self.tokenizer = tokenizer
        self.preprocess_list = preprocess_list or list()
        self.replacer_list = replacer_list or list()
        self.filter_list = filter_list or list()
        self.label_map = label_map or self._init_label_map()
        self.max_df = max_df
        self.stages = stages or [1.0, 0.80, 0.65, 0.40, 0.30, 0.20, 0.0]
        self.scores = scores or [1.0, 0.95, 0.75, 0.55, 0.45, 0.35, 0.0]

        self.vectorizer = CountVectorizer(
            # max_features=500,
            token_pattern=r"(?u)\b\w+\b",
            max_df=self.max_df,
            # min_df=0.01,
            binary=False
        )
        self.tfidf_transformer = TfidfTransformer(
            ratio_tf=True,
            smooth_idf=False,
            sublinear_tf=False,
        )
        self.tfidf: Dict[str, Dict[str, float]] = self._init_tfidf()

    def _init_label_map(self) -> dict:
        if isinstance(self.candidates, MysqlCandidatesIncludeIntentLib):
            t_dialog_node_info = TDialogNodeInfo(
                scene_id=self.scene_id,
                mysql_connect=self.candidates.mysql_connect
            )
            df = t_dialog_node_info.data
            df = df[df['product_id'] == self.candidates.product_id]
            df = df[df['scene_id'] == self.candidates.scene_id]

            node_desc2node_id = dict()
            for i, row in df.iterrows():
                node_id = row['node_id']
                node_desc = row['node_desc']
                node_desc = node_desc.lower().strip()
                if node_desc in (
                    'user is unclear 1',
                    'user is unclear 2',
                    'user is unclear 3',
                    '用户不清楚1',
                    '用户不清楚2',
                    '用户不清楚3',
                ):
                    node_desc2node_id[node_desc] = node_id

            label_map = dict()
            for k, v in [
                ('user is unclear 2', 'user is unclear 1'),
                ('user is unclear 3', 'user is unclear 1'),
                ('用户不清楚2', '用户不清楚1'),
                ('用户不清楚3', '用户不清楚1'),
            ]:
                k = node_desc2node_id.get(k)
                v = node_desc2node_id.get(v)
                if k is None or v is None:
                    continue
                label_map[k] = v
            result = label_map
        else:
            result = dict()
        return result

    def _init_tfidf(self):
        candidates = self.candidates.get()

        docs = defaultdict(list)
        for candidate in candidates:
            text = candidate['text']
            if len(text) == 0:
                continue
            label = candidate['node_id']
            if label in self.label_map:
                label = self.label_map[label]
            tokens = self._split(text)

            docs[label].extend(tokens)

        x = list()
        y = list()
        for k, v in docs.items():
            x.append(' '.join(v))
            y.append(k)
        x = self.vectorizer.fit_transform(x)

        tfidf = self.tfidf_transformer.fit_transform(x)
        if isinstance(tfidf, csr_matrix):
            tfidf = tfidf.toarray()

        feature_names = self.vectorizer.get_feature_names()

        result = defaultdict(lambda: defaultdict(float))
        for label, row in zip(y, tfidf):
            for feature_name, col in zip(feature_names, row):
                result[feature_name][label] = float(col)

        return result

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

    def score(self, context: RequestContext, candidate_list: List[dict]) -> List[dict]:
        logger.debug('{} score: '.format(self.register_name))
        context.append_row("""### {} score detail""".format(self.register_name))

        text = context.text

        tokens = self._split(text)

        label2metric = defaultdict(float)

        for token in tokens:
            label2value = self.tfidf[token]
            if len(label2value) == 0:
                continue
            for k, v in label2value.items():
                label2metric[k] += v

        total = sum(label2metric.values())
        label2score = dict()
        for k, v in label2metric.items():
            label2score[k] = v / total

        result = list()
        for candidate in candidate_list:
            node_id = candidate['node_id']
            if node_id in self.label_map:
                node_id = self.label_map[node_id]

            prob = label2score.get(node_id, 0.0)

            metric = round(prob, 4)

            score = score_transform(
                x=metric,
                stages=self.stages,
                scores=self.scores
            )

            result.append({
                'score': score,
                'metric': metric,
                'scorer': self.register_name,
                'label': None
            })

        context.append_table(result)
        return result


@Scorer.register('gensim_word2vec')
class Word2VecScorer(Scorer):
    def __init__(self,
                 filename: str,
                 tokenizer: Tokenizer,
                 preprocess_list: List[Preprocess] = None,
                 replacer_list: List[Replacer] = None,
                 filter_list: List[Filter] = None,
                 ):
        super().__init__()

        self.filename = filename
        self.tokenizer = tokenizer
        self.preprocess_list = preprocess_list
        self.replacer_list = replacer_list
        self.filter_list = filter_list
        self.word2vec = Word2Vec.load(filename)

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

    def _get_vectors_by_words(self, tokens: List[str]):
        vectors = [self.word2vec.wv[token] for token in tokens if token in self.word2vec.wv]
        result = matutils.unitvec(np.array(vectors).mean(axis=0))
        return result

    def score(self, context: RequestContext, candidate_list: List[dict]) -> List[dict]:
        logger.debug('{} score: '.format(self.register_name))
        context.append_row("""### {} score detail""".format(self.register_name))

        text = context.text

        tokens = self._split(text)
        if len(tokens) == 0:
            result = [{
                'score': 0.0,
                'metric': 0.0,
                'scorer': None,
                'label': None
            }] * len(candidate_list)
            return result
        v1 = self._get_vectors_by_words(tokens)

        result = list()
        for candidate in candidate_list:
            candidate_text = candidate['text']
            candidate_tokens = self._split(candidate_text)

            if len(candidate_tokens) == 0:
                result.append({
                    'score': 0.0,
                    'metric': 0.0,
                    'scorer': None,
                    'label': None
                })
            else:
                v2 = self._get_vectors_by_words(candidate_tokens)
                metric = np.dot(v1, v2)
                metric = round(float(metric), 4)

                score = 0.5 + 0.5 * metric

                result.append({
                    'score': score,
                    'metric': metric,
                    'scorer': self.register_name,
                    'label': None
                })

        context.append_table(result)
        return result


@Scorer.register('http_bert_whitening')
class HttpBertWhiteningScorer(Scorer):
    """
    给定语料, 全局初始化 kernel, bias.

    给定两个句子, 计算句子向量间的高斯距离.

    """
    def __init__(self,
                 tokenizer: Tokenizer,
                 preprocess_list: List[Preprocess] = None,
                 replacer_list: List[Replacer] = None,
                 filter_list: List[Filter] = None,
                 stages: List[float] = None,
                 scores: List[float] = None,
                 scale: float = 1.0
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.preprocess_list = preprocess_list or list()
        self.replacer_list = replacer_list or list()
        self.filter_list = filter_list or list()
        self.stages = stages or [1.0, 0.95, 0.85, 0.75, 0.65, 0.5, 0.0]
        self.scores = scores or [1.0, 0.85, 0.75, 0.35, 0.25, 0.1, 0.0]
        self.scale = scale

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

    def score(self, context: RequestContext, candidate_list: List[dict]) -> List[dict]:
        logger.debug('{} score: '.format(self.register_name))
        context.append_row("""### {} score detail""".format(self.register_name))

        text = context.text

        result = list()

        tokens1 = self._split(text)
        tokens1 = set(tokens1)
        for candidate in candidate_list:
            # bert_whitening 服务返回的召回结果中有 0-1 之间的余弦相似度分数.
            metric = candidate['score']
            text2 = candidate['text']

            score = score_transform(
                x=metric,
                stages=self.stages,
                scores=self.scores
            )
            tokens2 = self._split(text2)
            tokens2 = set(tokens2)
            if len(tokens1.intersection(tokens2)) == 0:
                score = self.scale * score
            result.append({
                'score': round(score, 4),
                'metric': round(metric, 4),
                'scorer': self.register_name,
                'label': None
            })

        context.append_table(result)
        return result


@Scorer.register('knn')
class KnnScorer(Scorer):
    def __init__(self,
                 max_distance: float = None,
                 stages: List[float] = None,
                 scores: List[float] = None,
                 ):
        super().__init__()

        self.max_distance = max_distance
        self.stages = stages or [1.0, 0.95, 0.85, 0.75, 0.65, 0.5, 0.0]
        self.scores = scores or [1.0, 0.85, 0.75, 0.35, 0.25, 0.1, 0.0]

    def score(self, context: RequestContext, candidate_list: List[dict]) -> List[dict]:
        logger.debug('{} score: '.format(self.register_name))
        context.append_row("""### {} score detail""".format(self.register_name))

        result = list()

        label2count = defaultdict(int)
        total = 0
        for candidate in candidate_list:
            metric = candidate['score']
            node_id = candidate['node_id']

            if metric > self.max_distance:
                continue

            label2count[node_id] += 1
            total += 1

        for candidate in candidate_list:
            metric = candidate['score']
            node_id = candidate['node_id']

            if metric > self.max_distance:
                continue

            count = label2count.get(node_id, 0)
            score = count / total

            score = score_transform(
                x=score,
                stages=self.stages,
                scores=self.scores
            )

            result.append({
                'score': round(score, 4),
                'metric': round(metric, 4),
                'scorer': self.register_name,
                'label': None
            })

        context.append_table(result)
        return result


@Scorer.register('jaccard')
class JaccardScorer(Scorer):
    """
    jaccard 字面匹配度.
    """
    def __init__(self,
                 jaccard_score: List[Tuple[Tuple[int, int], float]] = None,
                 preprocess_list: List[Preprocess] = None,
                 tokenizer: Tokenizer = None,
                 replacer_list: List[Replacer] = None,
                 filter_list: List[Filter] = None,
                 ):
        """

        :param jaccard_score: ((min_length, max_length), threshold).
        :param preprocess_list:
        :param tokenizer:
        :param replacer_list:
        :param filter_list:
        """
        super().__init__()
        self.jaccard_score = jaccard_score or [((0, 1e3), 0.9)]
        self.preprocess_list = preprocess_list or list()
        self.tokenizer = tokenizer or JiebaTokenizer()
        self.replacer_list = replacer_list or list()
        self.filter_list = filter_list or list()

    def _split(self, text: str):
        for preprocess in self.preprocess_list:
            text = preprocess.process(text)

        outlst, _ = self.tokenizer.tokenize(text)

        for replacer in self.replacer_list:
            outlst = replacer.replace(outlst)
            outlst = outlst[0]

        # 去除停用词后, 有可能返回空列表.
        for filter_ in self.filter_list:
            outlst = filter_.filter(outlst)

        return outlst

    def score(self, context: RequestContext, candidate_list: List[dict]):
        logger.debug('{} score: '.format(self.register_name))
        context.append_row("""### {} score detail""".format(self.register_name))

        # 按不同长度的 text 字符数, 查 min_jaccard_score 阈值.
        text = context.text

        l = len(text)
        min_jaccard_score = -1
        for (min_length, max_length), threshold in self.jaccard_score:
            if min_length <= l < max_length:
                min_jaccard_score = threshold
                break

        tokens1 = set(self._split(text))

        logger.debug('{} score, count: {}'.format(self.register_name, len(candidate_list)))
        result = list()
        for candidate in candidate_list:
            text2 = candidate['text']
            metric = candidate['score']

            tokens2 = set(self._split(text2))

            intersection = tokens1.intersection(tokens2)
            union = tokens1.union(tokens2)
            score = len(intersection) / (len(union) + 1e-7)

            logger.debug('text1: {}, text2: {}, score: {}'.format(text, text2, score))

            if score < min_jaccard_score:
                continue

            result.append({
                'score': round(score, 4),
                'metric': round(metric, 4),
                'scorer': self.register_name,
                'label': None
            })

        context.append_table(result)
        return result


@Scorer.register('edit_distance')
class EditDistanceScorer(Scorer):
    def __init__(self,
                 max_edit_distance_list: List[Tuple[Tuple[int, int], Tuple[float, float]]] = None,
                 preprocess_list: List[Preprocess] = None,
                 tokenizer: Tokenizer = None,
                 replacer_list: List[Replacer] = None,
                 filter_list: List[Filter] = None,
                 ):
        super().__init__()
        # 少于 3 个字符, 不通过
        self.max_edit_distance_list = max_edit_distance_list or [((0, 3), (-1, -1)), ((3, 5), (1, 1)), ((5, 8), (2, 2)), ((8, 15), (3, 3)), ((15, 20), (4, 4)), (20, 200), (5, 5)]
        self.preprocess_list = preprocess_list or list()
        self.tokenizer = tokenizer or JiebaTokenizer()
        self.replacer_list = replacer_list or list()
        self.filter_list = filter_list or list()

    def _split(self, text: str):
        for preprocess in self.preprocess_list:
            text = preprocess.process(text)

        outlst, _ = self.tokenizer.tokenize(text)

        for replacer in self.replacer_list:
            outlst = replacer.replace(outlst)
            outlst = outlst[0]

        # 去除停用词后, 有可能返回空列表.
        for filter_ in self.filter_list:
            outlst = filter_.filter(outlst)

        return outlst

    def _get_threshold_by_length(self, l1: float, l2: float):
        max_edit_distance = -1
        for (min_length, max_length), (low_threshold, high_threshold) in self.max_edit_distance_list:
            if min_length <= l1 < max_length:
                if l1 > l2:
                    max_edit_distance = low_threshold
                else:
                    max_edit_distance = high_threshold
                break

        return max_edit_distance

    def score(self, context: RequestContext, candidate_list: List[dict]):
        logger.debug('{} score: '.format(self.register_name))
        context.append_row("""### {} score detail""".format(self.register_name))

        text = context.text

        tokens1 = self._split(text)
        l1 = len(tokens1) + 1e-7

        logger.debug('{} score, count: {}'.format(self.register_name, len(candidate_list)))
        result = list()
        for candidate in candidate_list:
            text2 = candidate['text']
            metric = candidate['score']

            tokens2 = self._split(text2)
            l2 = len(tokens2) + 1e-7

            distance = editdistance.distance(tokens1, tokens2)

            threshold = self._get_threshold_by_length(l1, l2)

            if distance > threshold:
                score = 0.0
                result.append({
                    'score': round(score, 4),
                    'metric': round(metric, 4),
                    # scorer 为 None 的, 最终会被丢弃
                    'scorer': None,
                    'label': None
                })
            else:
                score = 1.0 - distance / l1
                result.append({
                    'score': round(score, 4),
                    'metric': round(metric, 4),
                    'scorer': self.register_name,
                    'label': None
                })
            logger.debug('text1: {}, text2: {}, distance: {}, score: {}, '
                         'threshold: {}'.format(
                text, text2, distance, score, threshold
            ))
            context.append_row("""text1: {}, text2: {}, distance: {}, score: {}, 
            threshold: {}""".format(
                text, text2, distance, score, threshold
            ))

        context.append_table(result)
        return result


class SingletonWeightWordScorer(ParamsSingleton):
    def __init__(self,
                 language: str,
                 synonyms_filename: str,
                 addition: dict = None,
                 ):
        if not self._initialized:
            addition = addition or dict()
            self.synonyms_filename = synonyms_filename

            if language == 'cn':
                from toolbox.string.pos_tokenizer import PyLTPPosTokenizer

                ltp_data_path = addition['ltp_data_path']

                synonyms_filename = os.path.join(project_path, synonyms_filename)
                ltp_data_path = os.path.join(project_path, ltp_data_path)

                self.word_weight_scorer = WeightWordScorer(
                    synonyms_filename=synonyms_filename,
                    post_pos_tokenizer=PyLTPPosTokenizer(
                        ltp_data_path=ltp_data_path,
                    ),
                    fast_tokenizer_splitter_name='by_char_splitter_v1',

                )
            elif language == 'th':
                from toolbox.string.pos_tokenizer import PyThaiNLPPosTokenizer

                synonyms_filename = os.path.join(project_path, synonyms_filename)

                self.word_weight_scorer = WeightWordScorer(
                    synonyms_filename=synonyms_filename,
                    post_pos_tokenizer=PyThaiNLPPosTokenizer(),
                    fast_tokenizer_splitter_name='list_encoder_one_splitter',
                )
            elif language == 'id':
                from toolbox.string.pos_tokenizer import IndonesianPosTokenizer

                model_file = addition['model_file']

                synonyms_filename = os.path.join(project_path, synonyms_filename)
                model_file = os.path.join(project_path, model_file)

                self.word_weight_scorer = WeightWordScorer(
                    synonyms_filename=synonyms_filename,
                    post_pos_tokenizer=IndonesianPosTokenizer(
                        model_file=model_file,
                    ),
                    fast_tokenizer_splitter_name='by_char_splitter_v1',
                )
            else:
                raise NotImplementedError
            self._initialized = True

    def score(self, *args, **kwargs):
        return self.word_weight_scorer.score(*args, **kwargs)


def demo1():
    import os
    from project_settings import project_path

    scorer = Scorer.from_json(
        params={
            'type': 'classifier_vector',
            'predictor': {
                'model_path': os.path.join(project_path, 'model/cls_20211021'),
                'predictor_name': 'text_classifier',
                'module_list': [
                    'toolbox.allennlp.data.dataset_readers.text_classification_json_utf8',
                    'toolbox.allennlp.models.basic_classifier_two_projection',
                ]
            },
            'sentence_vector_key': 'sentence_vector',

        }
    )
    print(scorer)
    result = scorer.score('你好', candidate_list=[
        {
            'text': '您好'
        }
    ])
    print(result)
    return


def demo2():
    import os
    from project_settings import project_path

    filename = os.path.join(project_path, 'data/wordings/chinese_wording_labeling.csv')
    scorer = TfIdfScorer.from_json(
        params={
            'filename': filename,
            'predictor': {
                'model_path': os.path.join(project_path, 'model/cls_20211021'),
                'predictor_name': 'text_classifier',
                'module_list': [
                    'toolbox.allennlp.data.dataset_readers.text_classification_json_utf8',
                    'toolbox.allennlp.models.basic_classifier_two_projection',
                ]
            },
        }
    )

    candidate_list = [
        {'text': '我很忙'},
        {'text': '正在忙'}
    ]
    result = scorer.score(text='我很忙', candidate_list=candidate_list)
    print(result)
    return


def demo3():
    import os
    from project_settings import project_path

    os.environ['NLTK_DATA'] = os.path.join(project_path, 'data/nltk_data')

    tree = DecisionTreeClassifierScorer.from_json(
        params={
            'candidates': {
                'type': 'mysql',
            },
            'tokenizer': {
                'type': 'nltk',
            },
            'preprocess_list': [
                {
                    'type': 'contraction'
                }
            ],
            'replacer_list': [
                {
                    'type': 'wordnet_lemma'
                }
            ],
            'filter_list': [
                {
                    'type': 'stopwords',
                    'filename': os.path.join(project_path, 'server/callbot_server/config/stopwords/english_stopwords.txt')
                }
            ],
            'label_map': {
                '3qczocvjb5': 'r01c5fucpi',
                'l324nf7hs3': 'r01c5fucpi',
            },
            'export_report': True
        },
        global_params={
            "env": "tianxing",
            "product_id": "callbot",
            "scene_id": "lbfk90nfs7",
            "node_id": "3e6b648e-5322-4b4f-a6a4-b2539d507a91",
            "node_type": 1,
            "node_desc": "测试",
            "mysql_connect": {
                "host": "10.20.251.13",
                "port": 3306,
                "user": "callbot",
                "password": "NxcloudAI2021!",
                "database": "callbot_ppe",
                "charset": "utf8"
            },
            "es_connect": {
                "hosts": ["10.20.251.8"],
                "http_auth": ["elastic", "ElasticAI2021!"],
                "port": 9200
            },
        }
    )
    print(tree)

    text = 'yes'
    array = tree.text2array(text)

    result = tree.predict_proba(array)
    print(result)
    return


def demo4():
    import os
    from project_settings import project_path

    os.environ['NLTK_DATA'] = os.path.join(project_path, 'data/nltk_data')

    scorer = TfIdfScorer.from_json(
        params={
            'candidates': {
                'type': 'mysql',
            },
            'tokenizer': {
                'type': 'nltk',
            },
            'preprocess_list': [
                {
                    'type': 'contraction'
                }
            ],
            'replacer_list': [
                {
                    'type': 'wordnet_lemma'
                }
            ],
            'filter_list': [
                {
                    'type': 'stopwords',
                    'filename': os.path.join(project_path, 'server/callbot_server/config/stopwords/english_stopwords.txt')
                }
            ],
        },
        global_params={
            "env": "tianxing",
            "product_id": "callbot",
            "scene_id": "lbfk90nfs7",
            "node_id": "3e6b648e-5322-4b4f-a6a4-b2539d507a91",
            "node_type": 1,
            "node_desc": "测试",
            "mysql_connect": {
                "host": "10.20.251.13",
                "port": 3306,
                "user": "callbot",
                "password": "NxcloudAI2021!",
                "database": "callbot_ppe",
                "charset": "utf8"
            },
            "es_connect": {
                "hosts": ["10.20.251.8"],
                "http_auth": ["elastic", "ElasticAI2021!"],
                "port": 9200
            },
        }
    )
    print(scorer)
    print(scorer.label_map)

    text = 'but i failed on withdraw'
    scorer.score(text, candidate_list=[])
    return


def demo5():
    import os
    from project_settings import project_path

    os.environ['NLTK_DATA'] = os.path.join(project_path, 'data/nltk_data')

    scorer = TfIdfScorer.from_json(
        params={
            'candidates': {
                'type': 'mysql',
            },
            'tokenizer': {
                'type': 'nltk',
            },
            'preprocess_list': [
                {
                    'type': 'contraction'
                }
            ],
            'replacer_list': [
                {
                    'type': 'wordnet_lemma'
                }
            ],
            'filter_list': [
                {
                    'type': 'stopwords',
                    'filename': os.path.join(project_path, 'server/callbot_server/config/stopwords/english_stopwords.txt')
                }
            ],
        },
        global_params={
            "env": "tianxing",
            "product_id": "callbot",
            "scene_id": "lbfk90nfs7",
            "node_id": "3e6b648e-5322-4b4f-a6a4-b2539d507a91",
            "node_type": 1,
            "node_desc": "测试",
            "mysql_connect": {
                "host": "10.20.251.13",
                "port": 3306,
                "user": "callbot",
                "password": "NxcloudAI2021!",
                "database": "callbot_ppe",
                "charset": "utf8"
            },
            "es_connect": {
                "hosts": ["10.20.251.8"],
                "http_auth": ["elastic", "ElasticAI2021!"],
                "port": 9200
            },
        }
    )
    print(scorer)
    print(scorer.label_map)

    text = 'but i failed on withdraw'
    scorer.score(text, candidate_list=[])
    return


def demo6():
    # BlackWhiteRegexScorer.demo1()
    IntentAndEntityScorer.demo2()
    return


if __name__ == '__main__':
    # demo1()
    # demo2()
    # demo3()
    # demo4()
    # demo5()
    demo6()
