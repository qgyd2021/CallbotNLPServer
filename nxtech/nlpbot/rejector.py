#!/usr/bin/python3
# -*- coding: utf-8 -*-
import re
from typing import List

from nxtech.nlpbot.misc import Preprocess
from nxtech.common.params import Params
from nxtech.database.mysql_connect import MySqlConnect
from toolbox.string.tokenization import RegularQuickFindTokenizer


class RejectorCandidates(Params):
    def __init__(self):
        super().__init__()

    def get(self) -> List[dict]:
        raise NotImplementedError


@RejectorCandidates.register('list')
class ListCandidates(RejectorCandidates):
    def __init__(self, candidates: List[dict]):
        super().__init__()

        self.candidates = candidates

    def get(self) -> List[dict]:
        return self.candidates


class Rejector(Params):
    def __init__(self):
        super().__init__()

    def reject(self, text: str):
        """
        拒答返回 True
        不拒答返回 False
        无法判断返回 None
        """
        raise NotImplementedError


@Rejector.register('text_list')
class TextListRejector(Rejector):
    def __init__(self,
                 candidates: RejectorCandidates,
                 preprocess_list: List[Preprocess] = None,
                 ):
        super().__init__()

        self.candidates = candidates
        self.preprocess_list = preprocess_list or list()
        self._candidates = self._init_candidates()

    def _init_candidates(self) -> List[str]:
        result = list()
        for candidate in self.candidates.get():
            text = candidate['text']
            for preprocess in self.preprocess_list:
                text = preprocess.process(text)
            result.append(text)
        return result

    def reject(self, text: str):
        for preprocess in self.preprocess_list:
            text = preprocess.process(text)
        if text in self.candidates:
            return True
        return None


class PatternRejector(Rejector):
    def __init__(self,
                 candidates: RejectorCandidates,
                 preprocess_list: List[Preprocess] = None,
                 ):
        super().__init__()

        self.candidates = candidates
        self.preprocess_list = preprocess_list

        self.string2pattern = dict()
        self.regular_quick_finder = RegularQuickFindTokenizer()
        self._pattern_candidates: List[tuple] = self._init_pattern_candidates()

    def _init_pattern_quick_finder(self):
        finder = RegularQuickFindTokenizer()
        result = list()
        for candidate in self.candidates.get():
            white = candidate['white']
            black = candidate['black']

            finder.insert(white)
            finder.insert(black)

            result.append((white, black))

        return result

    def _init_pattern_candidates(self):
        result = list()
        for candidate in self.candidates.get():
            white = candidate['white']
            black = candidate['black']
            try:
                white = re.compile(white, flags=re.IGNORECASE)
            except Exception as e:
                continue
            try:
                black = re.compile(black, flags=re.IGNORECASE)
            except Exception as e:
                continue
            result.append((white, black))
        return result

    def reject(self, text: str):
        for preprocess in self.preprocess_list:
            text = preprocess.process(text)
        return


if __name__ == '__main__':
    pass
