#!/usr/bin/python3
# -*- coding: utf-8 -*-
import re
from typing import List

from toolbox.sentence_patterns.sentence_parser import SentenceParser


class SentencePatternParser(object):
    def __init__(self, sentence_patterns: List[dict], sentence_parser: SentenceParser):
        self.sentence_parser = sentence_parser
        self.sentence_patterns = self.compile_sentence_patterns_regex(sentence_patterns)

    @classmethod
    def compile_sentence_patterns_regex(cls, sentence_patterns):
        """把从 json 文件中读取到的正则表达式都 compile 一下, 在使用时不再需要 compile. """
        result = list()
        for sp in sentence_patterns:
            en_title = sp['en_title']
            zh_title = sp['zh_title']
            sentence_pattern = sp['sentence_pattern']
            description = sp['description']
            utterances = sp.get('utterances', list())
            black_utterances = sp.get('black_utterances', list())

            tmp = list()
            for regex in sp['regex']:
                if regex.get('deleted', True):
                    continue
                positive = regex.get('positive', None)
                if positive is not None:
                    try:
                        positive = re.compile(positive, flags=re.IGNORECASE)
                    except re.error as e:
                        positive = None

                negative = regex.get('negative', None)
                if negative is not None:
                    try:
                        negative = re.compile(negative, flags=re.IGNORECASE)
                    except re.error as e:
                        negative = None
                tmp.append({
                    'positive': positive,
                    'negative': negative,
                    'title': regex.get('title', '')
                })
            result.append({
                'en_title': en_title,
                'zh_title': zh_title,
                'sentence_pattern': sentence_pattern,
                'description': description,
                'utterances': utterances,
                'black_utterances': black_utterances,
                'regex': tmp,
            })
        return result

    def parse(self, text: str) -> List[dict]:
        tokens, tags = self.sentence_parser.tokenize_and_pos(text)
        candidates = self.sentence_parser.tagged_sentence_to_string(
            tokens=tokens,
            tags=tags
        )
        result = list()
        for candidate in candidates:
            pattern = self._match_patterns(candidate)
            result.extend(pattern)
        return result

    def _match_patterns(self, text) -> List[dict]:
        result = list()
        for sp in self.sentence_patterns:
            en_title = sp['en_title']
            zh_title = sp['zh_title']
            sentence_pattern = sp['sentence_pattern']
            description = sp['description']
            utterances = sp['utterances']
            black_utterances = sp['black_utterances']

            if text in black_utterances:
                continue

            if text in utterances:
                result.append({
                    'type': 'utterances',
                    # 'en_title': en_title,
                    'zh_title': zh_title,
                    'sentence_pattern': sentence_pattern,
                    # 'description': description,
                    'text': text,
                    'utterances': text,
                })
                continue

            for regex in sp['regex']:
                negative = regex['negative']
                if negative is not None:
                    if negative.match(text) is not None:
                        continue
                positive = regex['positive']
                if positive is not None:
                    match = positive.match(text)
                    if match is not None:
                        result.append({
                            'type': 'pattern',
                            # 'en_title': en_title,
                            'zh_title': zh_title,
                            'sentence_pattern': sentence_pattern,
                            # 'description': description,
                            'text': text,
                            'pattern': positive,
                            'pattern_title': regex['title']
                        })
                        # break
        return result


def demo1():
    import json
    from toolbox.sentence_patterns.sentence_parser import default_cn_sentence_parser
    with open('config/cn_sentence_patterns.json', 'r', encoding='utf-8') as f:
        sentence_patterns = json.load(f)

    sentence_pattern_parser = SentencePatternParser(
        sentence_patterns=sentence_patterns,
        sentence_parser=default_cn_sentence_parser
    )

    text_list = [
        '有这个可不可以免费',
        '这个质量好不好',
        '不去行不行',
        '你怕不怕光呢',
        '你们害不害怕',
        '这个手机你喜不喜欢',
        '难道没有别的办法了吗',
        '也许这样就不会有事了吧',
        '好想你呀',
        '我打你'
    ]

    for text in text_list:
        result = sentence_pattern_parser.parse(text)
        print(result)
    return


def demo2():
    import json
    from toolbox.sentence_patterns.sentence_parser import default_cn_sentence_parser
    with open('config/cn_sentence_patterns.json', 'r', encoding='utf-8') as f:
        sentence_patterns = json.load(f)

    sentence_pattern_parser = SentencePatternParser(
        sentence_patterns=sentence_patterns,
        sentence_parser=default_cn_sentence_parser
    )

    text_list = [
        # '有这个可不可以免费',
        # '这个质量好不好',
        # '不去行不行',
        # '你怕不怕光呢',
        # '你们害不害怕',
        # '这个手机你喜不喜欢',
        # '难道没有别的办法了吗',
        # '也许这样就不会有事了吧',
        # '好想你呀',
        '我好想你',
        '好吧我是兵...'
        # '打你',
        # '杀人'
    ]

    # pattern = '^(?:<[^<]*/[^>]*>)*?(?P<verb><[^<]*/动词v>)(?P<noun><[^<]*/(名词n|人称代词r)>)(?P<tone><[^<]*/语气词y>)?$'
    pattern = '^(?:<[^<]*/[^>]*>)*?(?P<noun1><[^<]*/(名词n|人称代词r)>)(?P<adj><[^<]*/形容词a>)?(?P<verb><[^<]*/动词v>)(?P<noun2><[^<]*/(名词n|人称代词r)>)(?:<[^<]*/[^>]*>)*?$'
    for text in text_list:
        tokens, tags = sentence_pattern_parser.sentence_parser.tokenize_and_pos(text)
        candidates = sentence_pattern_parser.sentence_parser.tagged_sentence_to_string(
            tokens=tokens,
            tags=tags
        )
        for candidate in candidates:
            print(candidate)
            match = re.match(pattern, candidate, flags=re.IGNORECASE)
            print(match)
    return


def demo3():
    import json
    from toolbox.sentence_patterns.sentence_parser import default_en_sentence_parser
    with open('config/en_sentence_patterns.json', 'r', encoding='utf-8') as f:
        sentence_patterns = json.load(f)

    sentence_pattern_parser = SentencePatternParser(
        sentence_patterns=sentence_patterns,
        sentence_parser=default_en_sentence_parser
    )

    text_list = [
        # 'how do you think about it.',
        # 'it is interesting to learn english',
        # 'it is difficult for me to speak english',
        # 'it is kind of you to help me',
        # 'it appears in china',
        # 'it appeared in china',
        # 'it was natural',
        # 'it is expected to have it',
        # 'it happens that he is here',
        # 'it seems to me that',
        # 'it took me five months to write this book',
        # 'he is such a young boy.',
        # 'he is such a young boy that he can\'t do it',
        # 'hunt it',
        # 'how to do it',
        # 'where to live.',
        # 'when to come',
        # 'what to do',
        # 'whom to see',
        # 'which to choose',
        # 'why to do it',
        # 'how to begin is far more difficult than where to stop',
        # 'tell me what to do next',
        # 'what to do next',
        # 'i do not know',
        # 'i know',
        # 'i do know',
        # 'i exactly do not know',
        # 'the mother have the book',
        # 'the limit never increased',
        # 'yes',
        # 'no',
        # 'yes i do',
        # 'do the housework',
        # 'i saw the boy taken to school',
        # 'the quota has not increased',
        # 'he came home crying',
        'he sat in the corner reading a newspaper'
    ]

    for text in text_list:
        tokens, tags = sentence_pattern_parser.sentence_parser.tokenize_and_pos(text)
        candidates = sentence_pattern_parser.sentence_parser.tagged_sentence_to_string(
            tokens=tokens,
            tags=tags
        )
        print(candidates)
        result = sentence_pattern_parser.parse(text)
        print(result)
    return


if __name__ == '__main__':
    # demo1()
    # demo2()
    demo3()
