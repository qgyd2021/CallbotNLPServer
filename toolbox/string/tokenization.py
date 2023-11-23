#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
分词器
"""
from collections import defaultdict
import json
import logging
import re
from typing import *
import unicodedata

from tqdm import tqdm

from toolbox.string.character import Character, LowerCase, Pattern

logger = logging.getLogger(__file__)


_DEFAULT_SPLITTER_NAME = 'unknown'


class Splitter(object):
    def __init__(self, name=_DEFAULT_SPLITTER_NAME):
        self.name = name

    def split(self, text: str) -> List[str]:
        raise NotImplementedError()

    def post_process(self, tokens: List[List[str]]):
        return tokens


class ByCharSplitterV1(Splitter):
    def __init__(self, name=_DEFAULT_SPLITTER_NAME):
        super().__init__(name=name)

    def split(self, text: str) -> List[str]:
        return self._split(text)

    @staticmethod
    def _split(text: str) -> List[str]:
        flag = Character.f_unknown
        sep = '[sep]'
        ret = ''
        for c in text:
            if Character.is_hyphens(c):
                ret += c
                flag = Character.f_is_hyphens
            elif Character.is_punctuation(c) or Character.is_cjk_character(c) or Character.is_jap_character(c):
                if flag != Character.f_is_hyphens:
                    c = sep + c
                ret += c
                flag = Character.f_is_punctuation
            elif Character.is_space(c):
                # 连续的多个空隔, 不能合并为 1 个. 合并后, 分出的词不等于原来输入的别名.
                if flag != Character.f_is_space:
                    c = sep + c
                ret += c
                flag = Character.f_is_space
            elif Character.is_alpha(c):
                if flag not in (Character.f_is_alpha, Character.f_is_hyphens):
                    c = sep + c
                ret += c
                flag = Character.f_is_alpha
            elif Character.is_num(c):
                if flag not in (Character.f_is_num, Character.f_is_hyphens):
                    c = sep + c
                ret += c
                flag = Character.f_is_num
            else:
                if flag not in (Character.f_unknown, Character.f_is_hyphens):
                    c = sep + c
                ret += c
                flag = Character.f_unknown

        ret = ret.split(sep)
        ret = [ch for ch in ret if ch != '']

        if len(''.join(ret)) != len(text):
            raise AssertionError('this method should not change the char num. '
                                 'text: {}, ret: {}'.format(text, ''.join(ret)))
        return ret


class ByCharSplitterV2(Splitter):
    """
    在正则表达式的锚点识别时, `3000-3999` 应能分割出 `000`, 因此, 连续的数字须视作一个 token.
    于是定义了此类, 以区别于将连续的数字被识别为多个 token.
    ByCharSplitterV1 中, 连续的数字如 `3000` 将被分割为 ['3', '0', '0', '0']
    """
    def __init__(self, name=_DEFAULT_SPLITTER_NAME):
        super().__init__(name=name)

    def split(self, text: str) -> List[str]:
        return self._split(text)

    @staticmethod
    def _split(text: str) -> List[str]:
        """将 text 分割为 token list, 然后再按 token 到 trie 树匹配, 分词. """
        flag = Character.f_unknown
        sep = '[sep]'
        ret = ''
        for c in text:
            if Character.is_hyphens(c):
                # 3000-3999 应能分割出 000, 因此, 连字符不能生效.
                c = sep + c
                ret += c
                flag = Character.f_is_hyphens
            elif Character.is_punctuation(c) or Character.is_cjk_character(c) or Character.is_jap_character(c):
                if flag != Character.f_is_hyphens:
                    c = sep + c
                ret += c
                flag = Character.f_is_punctuation
            elif Character.is_space(c):
                # 连续的多个空隔, 不能合并为 1 个. 合并后, 分出的词不等于原来输入的别名.
                if flag != Character.f_is_space:
                    c = sep + c
                ret += c
                flag = Character.f_is_space
            elif Character.is_alpha(c):
                if flag not in (Character.f_is_alpha, Character.f_is_hyphens):
                    c = sep + c
                ret += c
                flag = Character.f_is_alpha
            elif Character.is_num(c):
                # 3000-3999 应能分割出 000, 因此, 连续的数字视作一个 token.
                if flag not in (Character.f_is_hyphens,):
                    c = sep + c
                ret += c
                flag = Character.f_is_num
            else:
                if flag not in (Character.f_unknown, Character.f_is_hyphens):
                    c = sep + c
                ret += c
                flag = Character.f_unknown

        ret = ret.split(sep)
        ret = [ch for ch in ret if ch != '']

        if len(''.join(ret)) != len(text):
            raise AssertionError('this method should not change the char num. '
                                 'text: {}, ret: {}'.format(text, ''.join(ret)))
        return ret


class ListSplitter(Splitter):
    def split(self, text: str):
        return list(text)


class ListEncodeOneSplitter(Splitter):
    def split(self, text: str):
        result = list()

        for c in text:
            dummy = '[{}]'.format(ord(c))
            result.append(dummy)
        return result

    def post_process(self, tokens: List[List[str]]):
        tokens_ = list()
        for token in tokens:
            token_ = list()
            for t in token:
                idx = t[1:-1]
                t = chr(int(idx))
                token_.append(t)
            tokens_.append(token_)

        return tokens_


_DEFAULT_SPLITTER_NAME_TO_SPLITTER = {
    'by_char_splitter_v1': ByCharSplitterV1(),
    'by_char_splitter_v2': ByCharSplitterV2(),
    'list_splitter': ListSplitter(),
    'list_encoder_one_splitter': ListEncodeOneSplitter(),
}


_DEFAULT_TOKENIZER_NAME = 'unknown'


class Tokenizer(object):
    """Abstract"""
    @staticmethod
    def lowercase(string: str) -> str:
        string = LowerCase.lowercase(string)
        return string

    def __init__(self, name=_DEFAULT_TOKENIZER_NAME, case_sensitive=False):
        self.name = name
        self.case_sensitive = case_sensitive

    def insert(self, word: str) -> None:
        raise NotImplementedError()

    def insert_from_list(self, words: Iterable[Any]) -> None:
        words = cast(List[Any], words)
        if len(words) == 0:
            return None
        for word in tqdm(words):
            self.insert(word)

    def insert_black(self, word: str) -> None:
        raise NotImplementedError()

    def insert_black_from_list(self, words: Iterable[Any]) -> None:
        words = cast(List[Any], words)
        if len(words) == 0:
            return None
        for word in tqdm(words):
            self.insert_black(word)

    def tokenize(self, text: str, full_mode: bool = False) -> Tuple[List[str], List[bool]]:
        raise NotImplementedError()

    @staticmethod
    def _merge_tokens(tokens: List[str], isword_list: List[bool]) -> Tuple[List[str], List[bool]]:
        """
        在 tokenize 分词后, 由于应用了黑名单, 有些分割出的词被标记为 False,
        这导致结果中出现连续的两个 False.
        在 segmenter 中, 多个分词器选后执行, 连续的两个 False 应合并, 以优化后面的分词的效果.
        这里, 只合并连续的两个 False, 不处理其它符号.
        """
        tokens2, isword_list2 = list(), list()
        false_token = ''
        for token, isword in zip(tokens, isword_list):
            if isword is False:
                false_token += str(token)
                continue

            if false_token != '':
                tokens2.append(false_token)
                isword_list2.append(False)

            tokens2.append(token)
            isword_list2.append(isword)
            false_token = ''
        else:
            if false_token != '':
                tokens2.append(false_token)
                isword_list2.append(False)
        return tokens2, isword_list2


class TrieNode(object):
    """建立词典的Trie树节点"""

    def __init__(self, t_word=None):
        self.t_word = t_word
        self.children = dict()

    def add_children(self, k, v):
        self.children[k] = v

    @property
    def text(self):
        if self.t_word is None:
            return None
        return ''.join(self.t_word)

    @property
    def isword(self):
        if self.t_word is None:
            return False
        return True

    def __repr__(self):
        return '<{}.{} t_word={}>'.format(self.__module__, self.__class__.__name__, self.t_word)


class FastTokenizer(Tokenizer):

    @staticmethod
    def demo1():
        fast = FastTokenizer()
        fast.insert('我要退款')
        fast.insert('色彩显示')
        fast.insert('我要')
        fast.insert('退款')
        fast.insert('eid')
        fast.insert('手机')
        fast.insert('机不')
        text = '手机不错我要退款'

        c = fast.tokenize(text, full_mode=True)

        print(c)
        return

    @staticmethod
    def demo2():
        fast = FastTokenizer(splitter=ListEncodeOneSplitter())
        # fast.insert('พูดว่')
        fast.insert('พูดว่า')
        fast.insert('นะ')
        fast.insert('พูดถึง')
        fast.insert('คำพูด')
        fast.insert('บอ')
        text = 'พูดว่าอะไรนะ'

        c = fast.tokenize(text, full_mode=False)

        print(c)
        return

    @staticmethod
    def token_list_to_string_list(token_list: List[List[str]]) -> List[str]:
        """因为 spliter 是将句子分割为 List[str], tokenize 是将列表中的子字符串合并为词. """
        ret = list()
        for l in token_list:
            ret.append(''.join(l))
        return ret

    def __init__(self, splitter: Optional[Union[Splitter, str]] = None, name=_DEFAULT_TOKENIZER_NAME, case_sensitive=False):
        if isinstance(splitter, str):
            splitter = _DEFAULT_SPLITTER_NAME_TO_SPLITTER[splitter]
        self.splitter = splitter or ByCharSplitterV1()
        self.trie = TrieNode()
        self._black_list: List[str] = list()
        super(FastTokenizer, self).__init__(name=name, case_sensitive=case_sensitive)

    def insert(self, word: str) -> None:
        word = str(word)

        if not self.case_sensitive:
            word = self.lowercase(word)

        t_word = self.splitter.split(word)
        self._insert_node(t_word)

    def insert_black(self, word: str) -> None:
        """
        黑名单.
        如遇到 `watch tv` 时, 不要识别出 `watch`.
        注意: 因为是最大匹配, 所以在 `watch` 在黑名单时, `watch tv` 是可以识别到的.
        """
        if word not in self._black_list:
            self.insert(word)
            self._black_list.append(word)

    def _insert_node(self, t_word: List[str]) -> None:
        now = self.trie
        for t in t_word[:-1]:
            if t not in now.children:
                now.add_children(t, TrieNode())
            now = now.children[t]
        t = t_word[-1]

        if t not in now.children:
            now.add_children(t, TrieNode(t_word))
        else:
            now.children[t].t_word = t_word

    def _tokenize(self, t_word: list, full_mode: bool = False):
        outlst, iswlst = list(), list()
        l = len(t_word)
        b_idx = 0
        l_idx = 0
        max_e_idx = 0
        while b_idx < l:
            now = self.trie
            found = False
            ptr = b_idx
            e_idx = None
            while True:
                t = t_word[ptr]
                if not self.case_sensitive:
                    t = self.lowercase(t)

                if t not in now.children and e_idx is not None:
                    found = True
                    break
                if t not in now.children and e_idx is None:
                    break
                if now.isword and full_mode:
                    if full_mode:
                        outlst.append(t_word[b_idx: ptr])
                        iswlst.append(True)

                now = now.children[t]
                ptr += 1
                if now.isword:
                    e_idx = ptr

                if ptr == l and e_idx is None:
                    break
                if ptr == l and e_idx is not None:
                    found = True
                    break

            if found is True:
                if l_idx != b_idx:
                    outlst.append(t_word[l_idx: b_idx])
                    iswlst.append(False)

                outlst.append(t_word[b_idx: e_idx])
                iswlst.append(True)
                max_e_idx = max(max_e_idx, e_idx)
                if full_mode:
                    b_idx += 1
                else:
                    b_idx = e_idx
                l_idx = b_idx
            else:
                b_idx += 1

        if max_e_idx < l:
            outlst.append(t_word[l_idx:l])
            iswlst.append(False)
        return outlst, iswlst

    def tokenize(self, text: Union[str, List[str]], full_mode=False) -> Tuple[List[str], List[bool]]:
        if isinstance(text, list):
            text_list = text
        else:
            text_list = [text]

        outlst, iswlst = list(), list()
        for text in text_list:
            t_word = self.splitter.split(text)
            outlst_tmp, iswlst_tmp = self._tokenize(t_word, full_mode)

            outlst.extend(outlst_tmp)
            iswlst.extend(iswlst_tmp)

        outlst = self.splitter.post_process(outlst)

        outlst = self.token_list_to_string_list(outlst)

        # 应用黑名单.
        for idx, out in enumerate(outlst):
            if out in self._black_list:
                iswlst[idx] = False

        outlst, iswlst = self._merge_tokens(outlst, iswlst)
        return outlst, iswlst


class TagTokenizer(FastTokenizer):
    def __init__(self, name=_DEFAULT_TOKENIZER_NAME, case_sensitive=False):
        super().__init__(name=name, case_sensitive=case_sensitive)
        self._word2flags_dict = defaultdict(list)

    def insert(self, word: str, tag: str = None) -> None:
        if tag is not None:
            self._word2flags_dict[word].append(tag)
        super().insert(word)

    def tokenize(self, text: Union[str, List[str]], full_mode: bool = False) -> Tuple[List[str], List[bool]]:
        outlst, iswlst = super().tokenize(text)

        iswlst2 = list()
        for out, isw in zip(outlst, iswlst):
            if isw is True:
                iswlst2.append(self._word2flags_dict.get(out, True))
            else:
                iswlst2.append(False)
        return outlst, iswlst2


class RegularTokenizer(Tokenizer):
    """
    不同于 FastTokenizer, 此处用正则表示代替词来进行匹配.

    优化:
    1. 基于正则表达式 index 的快速查找.
    2. re.compile. 在遇到无效正则表达式时, 会报错.
    """
    @staticmethod
    def demo1():
        regular = RegularTokenizer()
        regular.insert('我要退款')
        regular.insert('色彩显示')
        regular.insert('我要')
        regular.insert('退款')
        regular.insert('eid')
        regular.insert('手机')
        regular.insert('机不')
        regular.insert('\d+左右')

        text = '1500左右的手机不错我要退款'

        ret = regular.tokenize(text, full_mode=False)
        print(ret)
        return

    @staticmethod
    def _outlst_iswlst_append(token, isword, outlst, iswlst):
        if len(token) > 0:
            outlst.append(token)
            iswlst.append(isword)
        return outlst, iswlst

    def __init__(self, name=_DEFAULT_TOKENIZER_NAME, case_sensitive=False):
        self.regular_quick_find_tokenizer = RegularQuickFindTokenizer()
        self._black_list = list()
        super(RegularTokenizer, self).__init__(name=name, case_sensitive=case_sensitive)

    def insert(self, word: str) -> None:
        """
        :param word: 正则表达式.
        """
        self.regular_quick_find_tokenizer.insert(pattern=str(word))

    def insert_black(self, word: str) -> None:
        """添加黑名单"""
        if word not in self._black_list:
            self._black_list.append(word)

    def tokenize(self, text: str, full_mode: bool = False) -> Tuple[List[str], List[bool]]:
        text = str(text)
        if not self.case_sensitive:
            text_ = self.lowercase(text)
        else:
            text_ = text

        potential_pattern, no_index_pattern = self.regular_quick_find_tokenizer.get_potential_pattern(text=text_)
        # | 取并集, & 取交集.
        pattern_set = potential_pattern | no_index_pattern
        span_list = list()
        for pattern in pattern_set:
            try:
                if self.case_sensitive:
                    pattern = re.compile(pattern)
                else:
                    pattern = re.compile(pattern, re.I)
            except re.error as e:
                logger.error('{}, pattern: {}'.format(e, pattern))
                continue
            match_iter = re.finditer(pattern, text_)
            for match in match_iter:
                match_str = match.group(0).strip()
                if len(match_str) >= 2:
                    span_list.append(match.span())

        if full_mode:
            span_accept = span_list
        else:
            span_list = sorted(span_list, key=lambda x: x[1] - x[0], reverse=True)
            span_list = sorted(span_list, key=lambda x: x[0], reverse=False)

            span_accept = [(0, 0)]
            for span in span_list:
                if span[0] >= span_accept[-1][1]:
                    span_accept.append(span)

        outlst, iswlst = list(), list()
        last_idx = None
        for b, e in span_accept:
            if last_idx is None:
                outlst, iswlst = self._outlst_iswlst_append(text[:b], False, outlst, iswlst)
            else:
                outlst, iswlst = self._outlst_iswlst_append(text[last_idx:b], False, outlst, iswlst)
            outlst, iswlst = self._outlst_iswlst_append(text[b:e], True, outlst, iswlst)
            last_idx = e
        outlst, iswlst = self._outlst_iswlst_append(text[last_idx:], False, outlst, iswlst)

        # 应用黑名单.
        for idx, out in enumerate(outlst):
            if out in self._black_list:
                iswlst[idx] = False
        return self._merge_tokens(outlst, iswlst)


class RegularQuickFindTokenizer(FastTokenizer):
    """
    根据正则表达式的锚点, 快速查找可能在 text 上成立的正则表达式.

    1. insert 正则表达式,
    2. 获取索引, 并插入分词器,
    3. 使用分词器对句子分词, 匹配到的部分就有可能匹配其正则表达式.

    """
    @staticmethod
    def demo1():
        quick = RegularQuickFindTokenizer()
        quick.insert('.*[0-9]000.*到[0-9]999.*')
        quick.insert('^(?=.*(华为|苹果).*(手机|手表)).*(电脑|平板).*(?=.*小米(手机|手表)).*$')
        quick.insert('.*(输入密码)0米(\d{2.10}).*')
        quick.insert('.*(输入|密码)(\d{2.10}).*')
        quick.insert('^(?=.*(华为|苹果).*(电脑|平板|手表).*$')
        quick.insert('*0米.*(左|右).*')
        quick.insert('.*[0-9].*[0-9].*')
        quick.insert('\d+左右')

        text = '3000-3999 的华为手表, 有没有, 1500左右的也可以. '

        ret = quick.tokenize(text)
        print(ret)
        ret = quick.get_potential_pattern(text)
        print(ret)
        return

    def __init__(self, splitter: Optional[Splitter] = None, name=_DEFAULT_TOKENIZER_NAME, case_sensitive=False):
        splitter = splitter or ByCharSplitterV2()
        self._no_index_pattern: Set[str] = set()
        self._index_to_pattern: Dict[str, Set[str]] = defaultdict(set)
        super().__init__(splitter=splitter, name=name, case_sensitive=case_sensitive)

    def insert(self, pattern: str) -> None:
        indexes: List[str] = RegularIndexParse.get_indexes(pattern)
        if indexes is None:
            self._no_index_pattern.add(pattern)
        else:
            for index in indexes:
                self._index_to_pattern[index].add(pattern)
                super(RegularQuickFindTokenizer, self).insert(index)

    def get_potential_pattern(self, text: str) -> Tuple[Set[str], Set[str]]:
        """
        :return: 两个集合, 第一个是潜在正则表达式集合, 第二个是 insert 进来的无 index 正则,
        """
        pattern = set()
        # full_mode 默认为 True, 全量匹配所有可能的正则.
        outlst, iswlst = self.tokenize(text, full_mode=True)
        for out, isw in zip(outlst, iswlst):
            if isw is True:
                # 这里的方括号索引, 应该不会报错.
                pattern.update(self._index_to_pattern[out])
        return pattern, self._no_index_pattern


class RegularIndexParse(object):
    alp_num_ch = re.compile(Pattern.alp_num_ch)
    brackets = re.compile(Pattern.brackets)
    square_brackets = re.compile(Pattern.square_brackets)
    regex_dsw_find = re.compile(Pattern.regex_dsw_find)

    @staticmethod
    def demo1():
        pattern = '\d+左右'

        ret = RegularIndexParse.get_indexes(pattern)
        print(ret)
        return

    def __init__(self):
        pass

    @classmethod
    def _split_by_brackers(cls, text):
        # 按照括号对称分割字符串
        brackets = ['(', ')']
        result = []
        tmp = ''
        flag = 0
        for s in text:
            if s not in brackets:
                tmp += s
            elif s == '(':
                if tmp and flag == 0:
                    result.append(tmp)
                    tmp = ''
                tmp += s
                flag = flag + 1
            else:
                tmp += s
                flag = flag - 1
                if flag == 0:
                    result.append(tmp)
                    tmp = ''
        if tmp:
            result.append(tmp)
        return result

    @classmethod
    def _get_index_in_brackets(cls, text):
        # 文本中存在括号
        # 先查找括号外是否有索引
        # 如果没有，则查找括号内的索引组
        index = cls._get_index_out_of_brackets(text)
        if index:
            return [index.group()]

        tmps = cls.brackets.findall(text)
        index = []
        for tmp in tmps:
            tmp_index = cls.alp_num_ch.findall(tmp)
            if len(index) == 0:
                index = tmp_index
            elif len(tmp_index) < len(index):
                index = tmp_index
        return index

    @classmethod
    def _get_index_out_of_brackets(cls, text):
        # 去除正则表达式中, 在圆括号内的文字.
        tmp1 = cls.brackets.sub('', text)
        # 去除正则表达式中, 方括号部分
        tmp2 = cls.square_brackets.sub('', tmp1)
        # 去除如 \d+, \s+ 等.
        tmp3 = cls.regex_dsw_find.sub('', tmp2)
        # 取去除括号后的正则中的第一个文字作为 index.
        tmp4 = cls.alp_num_ch.search(tmp3)
        return tmp4

    @classmethod
    def get_indexes(cls, text: str) -> Union[List[str], None]:
        indexes = cls._get_index_out_of_brackets(text)
        if indexes:
            return [indexes.group()]
        pieces = cls._split_by_brackers(text)
        for p in pieces:
            if '(' in p:
                if '(' in p[1:-1]:
                    tmp_index = cls._get_index_in_brackets(p[1:-1])
                else:
                    tmp_index = cls.alp_num_ch.findall(p)

                if indexes is None:
                    indexes = tmp_index
                else:
                    if len(tmp_index) < len(indexes):
                        indexes = tmp_index
        return indexes


class IndivisibleTokenizer(FastTokenizer):
    def __init__(self,
                 indivisible_dict: Dict[str, Tuple[List[str], List[List[str]]]],
                 case_sensitive=False):
        """
        指定分割词 / 不可分割词.
        将词分按指定方式分割. 元组中第一项是分词的列表, 第二项是每个子词对应的词性(可以有多个词性).
        """
        super(IndivisibleTokenizer, self).__init__(case_sensitive=case_sensitive)
        self.word2tags = defaultdict(list)
        for word, t_words in indivisible_dict.items():
            self.insert(word, t_words)

    @classmethod
    def from_json_file(cls, filename, case_sensitive=False):
        with open(filename, 'r', encoding='utf-8') as f:
            indivisible_dict = json.load(f)
        return cls(indivisible_dict=indivisible_dict, case_sensitive=case_sensitive)

    def insert(self, word: str, tag: Tuple[List[str], List[List[str]]] = None) -> None:
        if tag is None:
            tag = list()
        self.word2tags[word] = tag
        super().insert(word)

    def tokenize(self, text: Union[str, List[str]], full_mode: bool = False) -> Tuple[List[str], List[bool]]:
        outlst, iswlst = super().tokenize(text)
        outlst2, iswlst2 = list(), list()
        for out, isw in zip(outlst, iswlst):
            if isw is True:
                word_list, tags_list = self.word2tags[out]
                outlst2.extend(word_list)
                iswlst2.extend(tags_list)
            else:
                outlst2.append(out)
                iswlst2.append(isw)
        return outlst2, iswlst2


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def demo1():
    text = '我想买一个老人用的, 1500左右, huawei watch gt 感觉还可以, 它性价比高吗, 有优惠活动吗?'
    fast = FastTokenizer()

    fast.insert_from_list(['huawei watch gt', 'huawei p30系列', 'huawei p30 pro'])
    # fast.insert('huawei p30系列')

    result = fast.tokenize(text)
    print(result)
    return


def demo2():
    text = '我想买一个老人用的, 1500左右, huawei watch gt 感觉还可以, 它性价比高吗, 有优惠活动吗?'
    fast = RegularTokenizer()
    fast.insert_from_list([r'\d+'])

    result = fast.tokenize(text)
    print(result)
    return


def demo3():
    text = '我想买一个老人用的, 1500左右, huawei watch gt 感觉还可以, 它性价比高吗, 有优惠活动吗?'
    RegularIndexParse.get_indexes('')
    ret = RegularIndexParse.get_indexes('.*[0-9]000.*到[0-9]999.*')
    print(ret)
    ret = RegularIndexParse.get_indexes('.*[0-9].*[0-9].*')
    print(ret)
    # ret = RegularIndexParse.get_indexes('.*[0-9]000.*到[0-9]999.*')
    # print(ret)
    # ret = RegularIndexParse.get_indexes('.*[0-9]000.*到[0-9]999.*')
    # print(ret)
    # ret = RegularIndexParse.get_indexes('.*[0-9]000.*到[0-9]999.*')
    # print(ret)

    # quick.insert('^(?=.*(华为|苹果).*(手机|手表)).*(电脑|平板).*(?=.*小米(手机|手表)).*$')
    # quick.insert('.*(输入密码)0米(\d{2.10}).*')
    # quick.insert('.*(输入|密码)(\d{2.10}).*')
    # quick.insert('^(?=.*(华为|苹果).*(电脑|平板|手表).*$')
    # quick.insert('*0米.*(左|右).*')
    # quick.insert('.*[0-9].*[0-9].*')

    return


def demo4():
    FastTokenizer.demo2()
    # FastTokenizer.demo3()
    # RegularQuickFindTokenizer.demo1()
    return


if __name__ == '__main__':
    # demo1()
    # demo2()
    # demo3()
    demo4()
