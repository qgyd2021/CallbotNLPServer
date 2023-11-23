#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
基于 category 类别, entity 实体, synonym 同义词, weight 权重, 的短句子相似度算分.
"""
from collections import defaultdict
import copy
import os
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from project_settings import project_path
from toolbox.string.pos_tokenizer import FastPosTokenizer, PyLTPPosTokenizer, PyThaiNLPPosTokenizer, PosTokenizer, _SubPosTokenizer


class Pair(object):
    def __init__(self, category: str, intent: str, standard: str, synonym: str,
                 weight: float,
                 category_weight: float, intent_weight: float, standard_weight: float,
                 ):
        self.category = category
        self.intent = intent
        self.standard = standard
        self.synonym = synonym
        self.weight = weight
        self.category_weight = category_weight
        self.intent_weight = intent_weight
        self.standard_weight = standard_weight

    def __repr__(self):
        # result = '<{}.{}> word: {} flag: {} weight: {}'.format(
        #     self.__module__, self.__class__.__name__,
        #     self.word, self.flag, self.weight
        # )
        result = '{}/{}/{}/{}/{}'.format(
            self.category, self.intent, self.standard, self.synonym, round(self.weight, 3)
        )
        return result

    def __str__(self):
        result = '{}'.format(
            self.synonym
        )
        return result


class PairList(object):
    def __init__(self, pair_list: List[Pair]):
        self.pair_list = pair_list

    def __add__(self, other):
        pair_list = self.pair_list + other.pair_list
        result = PairList(
            pair_list=pair_list,
        )
        return result
    #
    # def sum(self):
    #     result = 0.0
    #     for pair in self.pair_list:
    #         result += pair.weight
    #     return result

    def __repr__(self):
        # result = '<{}.{} length: {} synonym_factor: {}>'.format(
        #     self.__module__, self.__class__.__name__,
        #     len(self.pair_list), self.synonym_factor
        # )
        # result = '<{}.{} {}>'.format(
        #     self.__module__, self.__class__.__name__,
        #     self.pair_list
        # )
        result = '<{} {}>'.format(
            self.__class__.__name__,
            self.pair_list
        )
        return result

    def jaccard_score(self, other: "PairList"):
        pair_list1 = copy.deepcopy(other.pair_list)
        pair_list2 = copy.deepcopy(self.pair_list)
        # print(pair_list1)
        # print(pair_list2)
        pair_list1_matched = list()
        pair_list2_matched = list()

        match = 0.0
        # synonym
        for i, pair1 in enumerate(pair_list1):
            if i in pair_list1_matched:
                continue
            for j, pair2 in enumerate(pair_list2):
                if j in pair_list2_matched:
                    continue
                if pair1.category == pair2.category and pair1.intent == pair2.intent and pair1.standard == pair2.standard and pair1.synonym == pair2.synonym:
                    pair_list1_matched.append(i)
                    pair_list2_matched.append(j)
                    match += pair1.weight
                    # match += pair2.weight
                    break

        # standard
        for i, pair1 in enumerate(pair_list1):
            if i in pair_list1_matched:
                continue
            for j, pair2 in enumerate(pair_list2):
                if j in pair_list2_matched:
                    continue
                if pair1.category == pair2.category and pair1.intent == pair2.intent and pair1.standard == pair2.standard:
                    pair_list1_matched.append(i)
                    pair_list2_matched.append(j)
                    match += (pair1.weight * pair1.standard_weight)
                    # match += (pair2.weight * pair2.standard_weight)
                    break

        # intent
        for i, pair1 in enumerate(pair_list1):
            if i in pair_list1_matched:
                continue
            for j, pair2 in enumerate(pair_list2):
                if j in pair_list2_matched:
                    continue
                if pair1.category == pair2.category and pair1.intent == pair2.intent:
                    pair_list1_matched.append(i)
                    pair_list2_matched.append(j)
                    match += (pair1.weight * pair1.intent_weight)
                    # match += (pair2.weight * pair2.intent_weight)
                    break

        # category
        for i, pair1 in enumerate(pair_list1):
            if i in pair_list1_matched:
                continue
            for j, pair2 in enumerate(pair_list2):
                if j in pair_list2_matched:
                    continue
                if pair1.category == pair2.category:
                    pair_list1_matched.append(i)
                    pair_list2_matched.append(j)
                    match += (pair1.weight * pair1.category_weight)
                    # match += (pair2.weight * pair2.category_weight)
                    break

        total = sum([pair1.weight for pair1 in pair_list1]) + sum([pair2.weight for i, pair2 in enumerate(pair_list2) if i not in pair_list2_matched])
        # print(match)
        # print(total)
        score = match / (total + 1e-7)
        return round(score, 4)


class WeightWordScorer(object):
    @staticmethod
    def demo1():
        text1 = '我已经有了'
        text2 = '有啊'

        filename = os.path.join(project_path, '../NlpBot', 'server/callbot_nlp_server/config/weighted_word_xlsx/weighted_word_cn.xlsx')
        ltp_data_path = os.path.join(project_path, 'model/pyltp/ltp_data_v3.4.0')

        post_pos_tokenizer = PyLTPPosTokenizer(
            ltp_data_path=ltp_data_path,
        )
        word_weight_scorer = WeightWordScorer(
            synonyms_filename=filename,
            post_pos_tokenizer=post_pos_tokenizer,
            fast_tokenizer_splitter_name='by_char_splitter_v1',

        )
        matches = word_weight_scorer.score(text1=text1, text2=text2)
        for match in matches:
            print(match)
        return

    @staticmethod
    def demo2():
        text1 = 'พูดว่าอะไรนะครับ'
        text2 = 'พูดว่าอะไรนะ'

        filename = os.path.join(project_path, '../NlpBot', 'server/callbot_nlp_server/config/weighted_word_xlsx/weighted_word_th.xlsx')

        post_pos_tokenizer = PyThaiNLPPosTokenizer()
        word_weight_scorer = WeightWordScorer(
            synonyms_filename=filename,
            post_pos_tokenizer=post_pos_tokenizer,
            fast_tokenizer_splitter_name='list_encoder_one_splitter',
        )

        matches = word_weight_scorer.score(text1=text1, text2=text2)
        for match in matches:
            print(match)
        return

    def _warn_up(self):
        """第一次执行时很慢, init 时先调用一次. """
        text1 = 'pass'
        text2 = 'pass'
        self.score(text1=text1, text2=text2)
        return

    def __init__(self, synonyms_filename: str, post_pos_tokenizer: _SubPosTokenizer, unknown_weight: float = 0.3,
                 fast_tokenizer_splitter_name: str = 'by_char_splitter_v1', unknown_pos_tag_key: str = 'UNK',
                 ):
        """
        :param synonyms_filename: xlsx 同义词表.
        :param post_pos_tokenizer:
        :param unknown_weight:
        :param fast_tokenizer_splitter_name: 取值: by_char_splitter_v1, list_encoder_one_splitter.
        """
        self.synonyms_filename = synonyms_filename
        self.unknown_weight = unknown_weight
        self.fast_tokenizer_splitter_name = fast_tokenizer_splitter_name
        self.unknown_pos_tag_key = unknown_pos_tag_key

        self.unknown_word = 'unknown'
        self.unknown_weights = '0;0;0'

        self.post_pos_tokenizer: PosTokenizer = post_pos_tokenizer
        self.pos_tokenizer: PosTokenizer = None
        self.tag_to_row: Dict[str, dict] = None
        self._initialize()
        self._warn_up()

    def _initialize(self):
        fast_pos_tokenizer = FastPosTokenizer(
            splitter=self.fast_tokenizer_splitter_name,
        )
        tag_to_row: Dict[str, dict] = dict()

        df = pd.read_excel(self.synonyms_filename)

        for i, row in tqdm(df.iterrows(), total=len(df)):
            key = row['key']
            category = row['category']
            intent = row['intent']
            standard = row['standard']
            synonyms = row['synonyms']
            weight = row['weight']
            weights = row['weights']

            if any([pd.isna(key), pd.isna(category), pd.isna(intent), pd.isna(standard), pd.isna(synonyms)]):
                continue

            tag_to_row[key] = dict(row)

            # init fast_pos_tokenizer
            synonym_list = str(synonyms).split(';')

            for synonym in synonym_list:
                synonym = str(synonym).strip()
                if len(synonym) == 0:
                    continue

                fast_pos_tokenizer.insert(
                    text=synonym,
                    tag=key
                )

        pos_tokenizer = PosTokenizer(sub_pos_tokenizers=[
            fast_pos_tokenizer,
            self.post_pos_tokenizer,
        ])

        if tag_to_row.get(self.unknown_pos_tag_key) is None:
            raise AssertionError('row for unknown pos tag is required. key: `{}`'.format(self.unknown_pos_tag_key))
        self.pos_tokenizer = pos_tokenizer
        self.tag_to_row = tag_to_row
        return fast_pos_tokenizer, tag_to_row

    def score(self, text1: str, text2: str):
        token_list, tags_list = self.pos_tokenizer.posseg(text=text1)
        # print(token_list)
        # print(tags_list)
        source_pair_list_list: List[PairList] = self._make_candidate_pairs(token_list, tags_list)

        matches = list()
        token_list, tags_list = self.pos_tokenizer.posseg(text=text2)
        target_pair_list_list: List[PairList] = self._make_candidate_pairs(token_list, tags_list)
        for target_pair_list in target_pair_list_list:
            for source_pair_list in source_pair_list_list:
                score = source_pair_list.jaccard_score(target_pair_list)
                match = {
                    'source_pair_list': source_pair_list,
                    'target_pair_list': target_pair_list,
                    'score': score,
                }
                matches.append(match)

        # 只返回分数最大的.
        matches = [max(matches, key=lambda x: x['score'])]
        return matches

    def _make_candidate_pairs(self, token_list, tags_list) -> List[PairList]:
        candidates: List[PairList] = list()
        for token, tags in zip(token_list, tags_list):
            # if isinstance(tags, bool):
            #     continue
            tmp = list()
            if len(candidates) == 0:
                if tags is False:
                    category = self.unknown_word
                    intent = self.unknown_word
                    standard = self.unknown_word
                    synonym = token
                    weight = self.unknown_weight
                    weights = self.unknown_weights

                    category_weight, intent_weight, standard_weight = weights.split(';')
                    pair = Pair(category=category, intent=intent, standard=standard, synonym=synonym,
                                weight=weight,
                                category_weight=float(category_weight),
                                intent_weight=float(intent_weight),
                                standard_weight=float(standard_weight),
                                )
                    pair_list = PairList(
                        pair_list=[pair],
                    )
                    candidates.append(pair_list)
                else:
                    for tag in tags:
                        row = self.tag_to_row.get(tag, None)
                        if row is None:
                            row = self.tag_to_row[self.unknown_pos_tag_key]

                        weight = row['weight']
                        category = row['category']
                        intent = row['intent']
                        standard = row['standard']
                        synonym = token
                        weights = row['weights']
                        category_weight, intent_weight, standard_weight = weights.split(';')

                        pair = Pair(category=category, intent=intent, standard=standard, synonym=synonym,
                                    weight=weight,
                                    category_weight=float(category_weight),
                                    intent_weight=float(intent_weight),
                                    standard_weight=float(standard_weight),
                                    )
                        pair_list = PairList(
                            pair_list=[pair],
                        )
                        candidates.append(pair_list)
            else:
                for c in candidates:
                    if tags is False:
                        category = self.unknown_word
                        intent = self.unknown_word
                        standard = self.unknown_word
                        synonym = token
                        weight = self.unknown_weight
                        weights = self.unknown_weights
                        category_weight, intent_weight, standard_weight = weights.split(';')

                        pair = Pair(category=category, intent=intent, standard=standard, synonym=synonym,
                                    weight=weight,
                                    category_weight=float(category_weight),
                                    intent_weight=float(intent_weight),
                                    standard_weight=float(standard_weight),
                                    )
                        pair_list = PairList(
                            pair_list=[pair],
                        )
                        tmp.append(c + pair_list)
                    else:
                        for tag in tags:
                            row = self.tag_to_row.get(tag, None)
                            if row is None:
                                row = self.tag_to_row[self.unknown_pos_tag_key]

                            weight = row['weight']
                            category = row['category']
                            intent = row['intent']
                            standard = row['standard']
                            synonym = token
                            weights = row['weights']
                            category_weight, intent_weight, standard_weight = weights.split(';')

                            pair = Pair(category=category, intent=intent,
                                        standard=standard, synonym=synonym,
                                        weight=weight,
                                        category_weight=float(category_weight),
                                        intent_weight=float(intent_weight),
                                        standard_weight=float(standard_weight),
                                        )
                            pair_list = PairList(
                                pair_list=[pair],
                            )

                            tmp.append(c + pair_list)
                    candidates = tmp
        return candidates


def demo1():
    """句子相似度比较"""
    WeightWordScorer.demo1()
    WeightWordScorer.demo2()

    return


if __name__ == '__main__':
    demo1()
