#!/usr/bin/python3
# -*- coding: utf-8 -*-
import re
from typing import Dict, List, Tuple

from nltk.corpus import wordnet


_DEFAULT_CONTRACTION_PATTERNS = [
    (r'let\'s', 'let us'),
    (r'won\'t', 'will not'),
    (r'can\'t', 'can not'),
    (r'wasn\'t', 'was not'),
    (r'cannot', 'can not'),
    (r'don\'t', 'do not'),
    (r'donnot', 'do not'),
    (r'i\'m', 'i am'),
    (r'isn\'t', 'is not'),
    (r'(\w+)\'ll', r'\g<1> will'),
    (r'(\w+)n\'t', r'\g<1> not'),
    (r'(\w+)\'ve', r'\g<1> have'),
    (r'(\w+)\'s', r'\g<1> is'),
    (r'(\w+)\'re', r'\g<1> are'),
    (r'(\w+)\'d', r'\g<1> would'),

    (r'won’t', 'will not'),
    (r'can’t', 'can not'),
    (r'wasn’t', 'was not'),
    (r'cannot', 'can not'),
    (r'don’t', 'do not'),
    (r'donnot', 'do not'),
    (r'i’m', 'i am'),
    (r'isn’t', 'is not'),
    (r'(\w+)’ll', r'\g<1> will'),
    (r'(\w+)n’t', r'\g<1> not'),
    (r'(\w+)’ve', r'\g<1> have'),
    (r'(\w+)’s', r'\g<1> is'),
    (r'(\w+)’re', r'\g<1> are'),
    (r'(\w+)’d', r'\g<1> would')
]


class RegexpReplacer(object):
    def __init__(self, patterns: List[Tuple[str, str]]):
        self.patterns = [
            (re.compile(regex, flags=re.IGNORECASE), repl) for (regex, repl) in patterns
        ]

    def replace(self, text: str) -> str:
        s = text
        for pattern, repl in self.patterns:
            s = re.sub(pattern, repl, s)
        return s


contraction_replacer = RegexpReplacer(
    patterns=_DEFAULT_CONTRACTION_PATTERNS
)


punctuation_replacer = RegexpReplacer(
    patterns=[
        ('[,.!?@():！。？：，、<>《》{}｛｝\[\]]', ' '),
    ]
)


class RepeatReplacer(object):
    def __init__(self, repeat_regexp: str = '(\\w*)(\\w)\\2(\\w*)', repl: str = '\\1\\2\\3', use_wordnet: bool = True):
        self.repeat_regexp = re.compile(repeat_regexp)
        self.use_wordnet = use_wordnet
        self.repl = repl

    def replace(self, word: str):
        if self.use_wordnet and wordnet.synsets(word):
            return word

        repl_word = self.repeat_regexp.sub(self.repl, word)
        if repl_word != word:
            return self.replace(repl_word)
        else:
            return repl_word


repeat_replacer = RepeatReplacer()


def demo1():

    text_list = [
        'can\'t is a contraction',
        'I\'m should\'ve done that thing I didn\'t do',
        'let\'s talk about it'
    ]
    for text in text_list:
        result = contraction_replacer.replace(text)
        print(result)
    return


def demo2():
    from nltk.tokenize import _treebank_word_tokenizer

    text_list = [
        '<Shakespeare> works',
        '[what should i do] is a question!',
    ]
    for text in text_list:
        text = punctuation_replacer.replace(text)
        tokens = _treebank_word_tokenizer.tokenize(text)
        print(tokens)
    return


def demo3():
    import time

    text_list = [
        'my loooove is life',
        'oooooh',
        '天天开心'
    ]
    for text in text_list:
        begin = time.time()
        result = repeat_replacer.replace(text)
        cost = time.time() - begin
        print('cost: {}'.format(cost))
        print(result)
    return


def demo4():
    import os
    import sys
    import time

    pwd = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(os.path.join(pwd, '../../../'))
    os.environ['NLTK_DATA'] = os.path.join(pwd, '../../../data/nltk_data')

    repeat_replacer = RepeatReplacer(
        repeat_regexp='(\\w*)(\\w)\\2\\2(\\w*)',
        repl='\\1\\2\\2\\3',
        use_wordnet=False,
    )
    text_list = [
        'my loooove is life',
        'loooove',
        'oooooh',
        '要要天天天天开心啊啊啊啊啊哈哈哈'
    ]
    for text in text_list:
        begin = time.time()

        result = repeat_replacer.replace(text)
        cost = time.time() - begin
        print('cost: {}'.format(cost))
        print(result)
    return


if __name__ == '__main__':
    # demo1()
    # demo2()
    # demo3()
    demo4()
