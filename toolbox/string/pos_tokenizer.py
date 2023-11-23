#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
词性标注.
"""
from collections import defaultdict
from copy import deepcopy
import json
import os
from typing import *

from jieba.posseg import POSTokenizer
from pyltp import Postagger
from pyltp import Segmentor

from toolbox.string.tokenization import FastTokenizer, Splitter


class Pair(object):
    def __init__(self, word, flag):
        self.word = word
        self.flag = flag

    def __repr__(self):
        return 'pair(%r, %r)' % (self.word, self.flag)


class _SubPosTokenizer(object):
    def __init__(self, do_lower_case=True):
        self.do_lower_case = do_lower_case

    def posseg(self, text: str):
        raise NotImplementedError


class JiebaPosTokenizer(_SubPosTokenizer):
    """jieba 的词性标注太慢了. """
    def __init__(self, jieba_pos_convert_dict: dict = None, word2pos_list_dict: dict = None, do_lower_case=True):
        super().__init__(do_lower_case=do_lower_case)
        self.jieba_pos_convert_dict = jieba_pos_convert_dict or dict()
        self.word2pos_list_dict = word2pos_list_dict or dict()
        self.tokenizer = POSTokenizer()

    def posseg(self, text: str):
        if self.do_lower_case:
            text = text.lower()
        pairs = self.tokenizer.lcut(text)
        outlst, iswlst = list(), list()
        for pair in pairs:
            outlst.append(pair.word)
            iswlst.append(self._get_pos_by_pair(pair))

        return outlst, iswlst

    def _get_pos_by_pair(self, pair):
        pos_list = self.word2pos_list_dict.get(pair.word)
        if pos_list is None:
            pos_list = [self.jieba_pos_convert_dict.get(pair.flag, pair.flag)]
        return pos_list


class PyLTPPosTokenizer(_SubPosTokenizer):
    def __init__(self,
                 ltp_data_path: str,
                 pyltp_pos_convert_dict: dict = None,
                 word2pos_list_dict: dict = None,
                 do_lower_case: bool = True
                 ):
        super().__init__(do_lower_case=do_lower_case)
        self.ltp_data_path = ltp_data_path
        self.pyltp_pos_convert_dict = pyltp_pos_convert_dict or dict()
        self.word2pos_list_dict = word2pos_list_dict or dict()

        self.cws_model_path = os.path.join(ltp_data_path, 'cws.model')
        self.pos_model_path = os.path.join(ltp_data_path, 'pos.model')
        if not os.path.exists(self.cws_model_path):
            raise AssertionError
        if not os.path.exists(self.pos_model_path):
            raise AssertionError

        self.segmentor = Segmentor()
        self.segmentor.load(self.cws_model_path)

        self.pos_tagger = Postagger()
        self.pos_tagger.load(self.pos_model_path)

    def __del__(self):
        self.segmentor.release()
        self.pos_tagger.release()

    def __getstate__(self):
        result = {
            **self.__dict__
        }
        result.pop('segmentor')
        result.pop('pos_tagger')
        return result

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

        segmentor = Segmentor()
        segmentor.load(self.cws_model_path)
        setattr(self, 'segmentor', segmentor)

        pos_tagger = Postagger()
        pos_tagger.load(self.pos_model_path)
        setattr(self, 'pos_tagger', pos_tagger)
        return self

    def posseg(self, text: str):
        if self.do_lower_case:
            text = text.lower()
        words = self.segmentor.segment(text)

        postags = self.pos_tagger.postag(words)

        outlst, iswlst = list(), list()
        for word, postag in zip(words, postags):
            outlst.append(word)
            iswlst.append(self._get_pos_by_pair(word, postag))

        return outlst, iswlst

    def _get_pos_by_pair(self, word: str, postag: str):
        pos_list = self.word2pos_list_dict.get(word)
        if pos_list is None:
            pos_list = [self.pyltp_pos_convert_dict.get(postag, postag)]
        return pos_list


class PyThaiNLPPosTokenizer(_SubPosTokenizer):
    @staticmethod
    def demo1():
        text = 'พูดว่าอะไรนะ'

        tokenizer = PyThaiNLPPosTokenizer(
            pos_convert_dict={
                'JSBR': '连词',
                'PNTR': '代词',
                'NCMN': '助词',
            },
            word2pos_list_dict={},

        )

        outlst, iswlst = tokenizer.posseg(text)
        print(outlst)
        print(iswlst)

        return

    def __init__(self,
                 pos_convert_dict: dict = None,
                 word2pos_list_dict: dict = None,
                 do_lower_case=True
                 ):
        """
        :param pos_convert_dict: 将词性标签映射成其它标签. 可用于标签合并等.
        :param word2pos_list_dict: 为特定词指定词性.
        :param do_lower_case:
        """
        super().__init__(do_lower_case=do_lower_case)
        self.pos_convert_dict = pos_convert_dict or dict()
        self.word2pos_list_dict = word2pos_list_dict or dict()

        from pythainlp.tag import pos_tag
        from pythainlp.tokenize import word_tokenize

        self.pos_tag = pos_tag
        self.word_tokenize = word_tokenize

    def posseg(self, text: str):
        tokens = self.word_tokenize(text)
        token_tag_list = self.pos_tag(tokens)

        outlst, iswlst = list(), list()
        for word, postag in token_tag_list:
            outlst.append(word)
            iswlst.append(self._get_pos_by_pair(word, postag))

        return outlst, iswlst

    def _get_pos_by_pair(self, word: str, postag: str):
        pos_list = self.word2pos_list_dict.get(word)
        if pos_list is None:
            pos_list = [self.pos_convert_dict.get(postag, postag)]
        return pos_list


class IndonesianPosTokenizer(_SubPosTokenizer):
    """
    词性标注模型来源.
    https://github.com/wbwb33/pos-tagger-indonesian-api
    """
    @staticmethod
    def demo1():
        import os
        from project_settings import project_path

        text = 'apa dalemnya'

        model_file = os.path.join(project_path, '../NlpBot', 'third_party_data/pos_tag/indonesian/all_indo_man_tag_corpus_model.crf.tagger')
        print(model_file)
        tokenizer = IndonesianPosTokenizer(
            model_file=model_file,
            pos_convert_dict={
                'WH': '疑问代词',
                'RB': '副词',
            },
            word2pos_list_dict={},
        )

        outlst, iswlst = tokenizer.posseg(text)
        print(outlst)
        print(iswlst)

        return

    def __init__(self,
                 model_file: str,
                 pos_convert_dict: dict = None,
                 word2pos_list_dict: dict = None,
                 do_lower_case=True
                 ):
        super().__init__(do_lower_case=do_lower_case)
        self.model_file = model_file
        self.pos_convert_dict = pos_convert_dict or dict()
        self.word2pos_list_dict = word2pos_list_dict or dict()

        from nltk.tag import CRFTagger
        from nltk.tokenize import _treebank_word_tokenizer

        crf_tagger = CRFTagger()
        crf_tagger.set_model_file(model_file)

        self.pos_tag = crf_tagger
        self.word_tokenize = _treebank_word_tokenizer

    def posseg(self, text: str):
        tokens = self.word_tokenize.tokenize(text)
        token_tag_list = self.pos_tag.tag_sents([tokens])
        token_tag_list = token_tag_list[0]

        outlst, iswlst = list(), list()
        for word, postag in token_tag_list:
            outlst.append(word)
            iswlst.append(self._get_pos_by_pair(word, postag))

        return outlst, iswlst

    def _get_pos_by_pair(self, word: str, postag: str):
        pos_list = self.word2pos_list_dict.get(word)
        if pos_list is None:
            pos_list = [self.pos_convert_dict.get(postag, postag)]
        return pos_list


class IndivisibleTokenizer(_SubPosTokenizer):
    def __init__(self, do_lower_case=True):
        super().__init__(do_lower_case=do_lower_case)
        self.tokenizer = FastTokenizer(case_sensitive=not self.do_lower_case)
        self.text2segments: Dict[str, Dict[str, List]] = dict()

    def insert(self, text: str, segments: dict) -> None:
        if self.do_lower_case:
            text = text.lower()
        self.text2segments[text] = segments
        self.tokenizer.insert(text)

    def posseg(self, text: str):
        outlst, iswlst = self.tokenizer.tokenize(text)

        outlst2, iswlst2 = list(), list()
        for out, isw in zip(outlst, iswlst):
            if isw is True:
                segments = self.text2segments[out]
                outlst2.extend(segments['tokens'])
                iswlst2.extend(segments['tags'])
            else:
                outlst2.append(out)
                iswlst2.append(isw)
        return outlst2, iswlst2


class FastPosTokenizer(_SubPosTokenizer):
    def __init__(self, splitter: Optional[Union[Splitter, str]] = None, do_lower_case=True):
        super().__init__(do_lower_case=do_lower_case)
        self.tokenizer = FastTokenizer(splitter=splitter, case_sensitive=not self.do_lower_case)
        self.word2tags = defaultdict(set)

    def insert(self, text: str, tag: str) -> None:
        self.tokenizer.insert(text)

        if self.do_lower_case:
            text = text.lower()
        self.word2tags[text].add(tag)

    def posseg(self, text: str):
        outlst, iswlst = self.tokenizer.tokenize(text)

        outlst2, iswlst2 = list(), list()
        for out, isw in zip(outlst, iswlst):
            if isw is True:
                if self.do_lower_case:
                    key = out.lower()
                else:
                    key = out
                tags: set = self.word2tags[key]
                outlst2.append(out)
                iswlst2.append(list(tags))
            else:
                outlst2.append(out)
                iswlst2.append(isw)
        return outlst2, iswlst2


class SimpleRegularPosTokenizer(_SubPosTokenizer):
    @staticmethod
    def demo1():
        tokenizer = SimpleRegularPosTokenizer()

        pattern_tag_list = [
            ('[\d+\.]+', '数值d'),
        ]
        for pattern, tag in pattern_tag_list:
            tokenizer.insert(pattern=pattern, tag=tag)

        text = '5.5还是很高啊'
        result = tokenizer.posseg(text)
        print(result)
        return

    def __init__(self, pattern_list: List[str], tag: str, do_lower_case=True):
        super().__init__(do_lower_case=do_lower_case)
        self.pattern_list = pattern_list
        self.tag = tag

    def posseg(self, text: str):
        outlst, iswlst = self.tokenizer.tokenize(text)

        outlst2, iswlst2 = list(), list()
        for out, isw in zip(outlst, iswlst):
            print(out)
            print(isw)
            if isw is True:
                if self.do_lower_case:
                    key = out.lower()
                else:
                    key = out
                tags: set = self.pattern2tags[key]
                outlst2.append(out)
                iswlst2.append(list(tags))
            else:
                outlst2.append(out)
                iswlst2.append(isw)
        return outlst2, iswlst2


class PosTokenizer(object):
    def __init__(self, sub_pos_tokenizers: List[_SubPosTokenizer]):
        self.sub_pos_tokenizers = sub_pos_tokenizers

    def posseg(self, text: str):
        outlst, iswlst = list(), list()
        for i, sub_pos_tokenizer in enumerate(self.sub_pos_tokenizers):
            outlst_tmp, iswlst_tmp = list(), list()
            if i == 0:
                outlst_tmp, iswlst_tmp = sub_pos_tokenizer.posseg(text)
            else:
                for out, isw in zip(outlst, iswlst):
                    if isw is False:
                        a, b = sub_pos_tokenizer.posseg(out)
                        outlst_tmp.extend(a)
                        iswlst_tmp.extend(b)
                    else:
                        outlst_tmp.append(out)
                        iswlst_tmp.append(isw)

            outlst = deepcopy(outlst_tmp)
            iswlst = deepcopy(iswlst_tmp)
        return outlst, iswlst


def _make_candidate_tag_list(iswlst):
    candidate = list()
    for i, isw in enumerate(iswlst):
        if i == 0:
            for o in isw:
                candidate.append([o])
        else:
            c2 = list()
            for c in candidate:
                for o in isw:
                    c2.append(c + [o])
            candidate = c2
    return candidate


def _make_pos_strings(tokens, tags):
    candidate = list()
    for token, tag in zip(tokens, tags):
        tmp = list()
        if len(candidate) == 0:
            for t in tag:
                add = '<{}/{}>'.format(token, t)
                candidate.append(add)
        else:
            for c in candidate:
                for t in tag:
                    add = '<{}/{}>'.format(token, t)
                    tmp.append(c + add)
                candidate = tmp
    return candidate


def demo1():
    with open('jieba_pos_conv.json', 'r', encoding='utf-8') as f:
        jieba_pos_convert_dict = json.load(f)
    tokenizer = JiebaPosTokenizer(
        jieba_pos_convert_dict=jieba_pos_convert_dict,
        word2pos_list_dict={
            '需要': ['表需要动词v']
        }
    )
    text = '你不需要再和谁去争奇斗艳'
    result = tokenizer.posseg(text)
    print(result)
    return


def demo2():
    with open('indivisible_dict2.json', 'r', encoding='utf-8') as f:
        indivisible_dict = json.load(f)

    tokenizer = IndivisibleTokenizer()
    for k, v in indivisible_dict.items():
        tokenizer.insert(k, v)

    text = '天空好蓝, 我好想你'
    result = tokenizer.posseg(text)
    print(result)
    return


def demo3():
    tokenizer = FastPosTokenizer(

    )
    tokenizer.insert('你好', '敬语l')

    text = '你好, 我好想你'
    result = tokenizer.posseg(text)
    print(result)
    return


def demo4():
    jieba_pos_tokenizer = JiebaPosTokenizer(
        jieba_pos_convert_dict={
            "a": "形容词a",
            "c": "连词c",
            "d": "副词d",
            "eng": "英文字符eng",
            "i": "成语i",
            "k": "后接成分k",
            "l": "习用语l",
            "m": "数词m",
            "nr": "人名nr",
            "ns": "地名ns",
            "q": "量词q",
            "s": "处所词s",
            "v": "动词v",
            "vn": "动词v",
            "y": "语气词y",

            "r": "代词r",
            "t": "时间词t",
            "n": "名词n",
            "p": "介词p",
            "ul": "助词u",
            "uj": "助词u",

            "z": "状态词z",
            "zg": "状态词zg",
            "x": "其它词x"
        },
        word2pos_list_dict={
            '我': ['人称代词r']
        }
    )

    indivisible_dict = {
        "好想你": {
            "tokens": ["好", "想", "你"],
            "tags": [
                ["形容词a"],
                ["动词v"],
                ["人称代词r"]
            ]
        }
    }
    indivisible_tokenizer = IndivisibleTokenizer()
    for k, v in indivisible_dict.items():
        indivisible_tokenizer.insert(k, v)

    fast_pos_tokenizer = FastPosTokenizer()
    fast_pos_tokenizer.insert('你好', '敬语l')
    fast_pos_tokenizer.insert('你好', '招呼语l')

    pos_tokenizer = PosTokenizer(
        sub_pos_tokenizers=[
            indivisible_tokenizer,
            fast_pos_tokenizer,
            jieba_pos_tokenizer
        ]
    )
    # text = '你好, 我好想你'
    # text = '你好, 颐和园'
    text = '我已经有了'

    outlst, iswlst = pos_tokenizer.posseg(text)
    print(outlst, iswlst)

    result = _make_candidate_tag_list(iswlst)
    print(result)

    result = _make_pos_strings(outlst, iswlst)
    print(result)
    return


def demo5():
    # PyThaiNLPPosTokenizer.demo1()
    IndonesianPosTokenizer.demo1()
    return


if __name__ == '__main__':
    # demo1()
    # demo2()
    # demo3()
    # demo4()
    demo5()
