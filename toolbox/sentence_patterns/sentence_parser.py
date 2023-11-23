#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
句子解析, 拆解
"""
import json
from typing import List, Union

from toolbox.string.tokenization import Tokenizer
from toolbox.string.tokenization import TagTokenizer, IndivisibleTokenizer, JiebaPosTokenizer, NltkPosTokenizer


class SentenceParser(object):
    def __init__(self,
                 default_pos_tokenizer: Tokenizer,
                 indivisible_tokenizer: IndivisibleTokenizer = None,
                 tag_tokenizer: TagTokenizer = None,
                 ):
        self.indivisible_tokenizer = indivisible_tokenizer
        self.tag_tokenizer = tag_tokenizer
        self.default_pos_tokenizer = default_pos_tokenizer

    def _tokenize(self, outlst: List[str], iswlst: List[Union[bool, List[str]]], tokenizer: Tokenizer):
        outlst2, iswlst2 = list(), list()
        for text, flag in zip(outlst, iswlst):
            if flag is False:
                outlst_tmp, iswlst_tmp = tokenizer.tokenize(text)
                outlst2.extend(outlst_tmp)
                iswlst2.extend(iswlst_tmp)
            elif flag is True:
                raise NotImplementedError()
            else:
                outlst2.append(text)
                iswlst2.append(flag)
        return outlst2, iswlst2

    def tokenize_and_pos(self, text: str):
        outlst = [text]
        iswlst = [False]
        if self.indivisible_tokenizer:
            outlst, iswlst = self._tokenize(outlst, iswlst, tokenizer=self.indivisible_tokenizer)

        if self.tag_tokenizer:
            outlst, iswlst = self._tokenize(outlst, iswlst, tokenizer=self.tag_tokenizer)

        outlst, iswlst = self._tokenize(outlst, iswlst, tokenizer=self.default_pos_tokenizer)
        return outlst, iswlst

    def tagged_sentence_to_string(self, tokens, tags):
        candidates = list()
        for token, tag in zip(tokens, tags):
            tmp = list()
            if len(candidates) == 0:
                for t in tag:
                    add = '<{}/{}>'.format(token, t)
                    candidates.append(add)
            else:
                for c in candidates:
                    for t in tag:
                        add = '<{}/{}>'.format(token, t)
                        tmp.append(c + add)
                    candidates = tmp
        return candidates


def _get_default_cn_sentence_parser():
    # indivisible_tokenizer
    indivisible_tokenizer = IndivisibleTokenizer()
    with open('config/cn_indivisible_dict.json', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            line = json.loads(line)
            text = line['text']
            tag = line['tag']
            indivisible_tokenizer.insert(word=text, tag=tag)

    # tag_tokenizer
    tag_tokenizer = TagTokenizer()
    with open('config/cn_tag_tokenizer_dict.json', 'r', encoding='utf-8') as f:
        tag_tokenizer_dict = json.load(f)
        for word, tags in tag_tokenizer_dict.items():
            for tag in tags:
                tag_tokenizer.insert(word=word, tag=tag)

    # jieba_pos_tokenizer
    with open('config/cn_jieba_pos_map.json', 'r', encoding='utf-8') as f:
        pos_map = json.load(f)
    with open('config/cn_word2tags.json', 'r', encoding='utf-8') as f:
        word2tags = json.load(f)
    jieba_pos_tokenizer = JiebaPosTokenizer(word2tags=word2tags, pos_map=pos_map)

    parser = SentenceParser(
        indivisible_tokenizer=indivisible_tokenizer,
        tag_tokenizer=tag_tokenizer,
        default_pos_tokenizer=jieba_pos_tokenizer,
    )
    return parser


default_cn_sentence_parser = _get_default_cn_sentence_parser()


def _get_default_en_sentence_parser():
    # indivisible_tokenizer
    indivisible_tokenizer = IndivisibleTokenizer()
    with open('config/en_indivisible_dict.json', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            line = json.loads(line)
            text = line['text']
            tag = line['tag']
            indivisible_tokenizer.insert(word=text, tag=tag)

    # tag_tokenizer
    tag_tokenizer = TagTokenizer()
    with open('config/en_tag_tokenizer_dict.json', 'r', encoding='utf-8') as f:
        tag_tokenizer_dict = json.load(f)
        for word, tags in tag_tokenizer_dict.items():
            for tag in tags:
                tag_tokenizer.insert(word=word, tag=tag)

    # jieba_pos_tokenizer
    with open('config/en_nltk_pos_map.json', 'r', encoding='utf-8') as f:
        pos_map = json.load(f)
    with open('config/en_word2tags.json', 'r', encoding='utf-8') as f:
        word2tags = json.load(f)
    nltk_pos_map = NltkPosTokenizer(word2tags=word2tags, pos_map=pos_map)

    parser = SentenceParser(
        indivisible_tokenizer=indivisible_tokenizer,
        tag_tokenizer=tag_tokenizer,
        default_pos_tokenizer=nltk_pos_map,
    )
    return parser


default_en_sentence_parser = _get_default_en_sentence_parser()


def demo1():

    text_list = [
        '有这个可不可以免费',
        '这个质量好不好',
        '不去行不行',
        '你怕不怕光呢',
        '你们害不害怕',
        '这个手机你喜不喜欢',
        '难道没有别的办法了吗',
        '也许这样就不会有事了吧',
        '好想你呀'

    ]
    for text in text_list:
        tokens, tags = default_cn_sentence_parser.tokenize_and_pos(text)
        result = default_cn_sentence_parser.tagged_sentence_to_string(
            tokens=tokens,
            tags=tags
        )
        print(result)
    return


def demo2():

    text_list = [
        'how \' do " you do',
        'it\'s me',
        'my name is honey',
        'i don\'t think so.',

    ]
    for text in text_list:
        ret = default_en_sentence_parser.tokenize_and_pos(text)
        print(ret)
    return


if __name__ == '__main__':
    demo1()
    # demo2()
