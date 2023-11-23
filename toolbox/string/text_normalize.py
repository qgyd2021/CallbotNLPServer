#!/usr/bin/python3
# -*- coding: utf-8 -*-
import re
from typing import List, Set, Union

from toolbox.common.registrable import Registrable


_DEFAULT_STOP_CHAR_SET = set('~@#$%^&*()_-+=/\\<>{}')

_DEFAULT_ZH2EN_PUNCTUATION = {
    '“': '"',
    '”': '"',
    '‘': '\'',
    '’': '\'',
    '！': '!',
    '？': '?',
    '，': ',',
    '、': ',',
    '。': '.',
    '：': ':',
    '；': ':',
    '（': '(',
    '）': ')',
}


class TextNormalizer(Registrable):

    def __init__(self, normalizer_list: List[Union[str, 'TextNormalizer']] = None):
        self.normalizer_list = normalizer_list or [
            'multiple_spaces_to_one',
            'clean_space_not_between_alnum',
            'clean_stop_char',
            'full2semi_angle',
            'zh2en_punctuation',

        ]

    def stacked_normalize(self, text: str, normalizer_list: List[Union[str, 'TextNormalizer']] = None):
        normalizer_list = normalizer_list or self.normalizer_list
        for normalizer in normalizer_list:
            if isinstance(normalizer, str):
                normalizer = self.by_name(normalizer)
                normalizer = normalizer()

            text = normalizer.normalize(text)
        return text

    def normalize(self, text: str):
        raise NotImplementedError('父类, 请使用 stacked_normalize')


@TextNormalizer.register('multiple_spaces_to_one')
class MultipleSpacesToOne(TextNormalizer):
    def __init__(self):
        super().__init__()

    def normalize(self, text: str):
        pattern = r'\s+'

        text = re.sub(pattern=pattern, repl=' ', string=text)
        return text


@TextNormalizer.register('clean_space_not_between_alnum')
class CleanSpaceNotBetweenAlnum(TextNormalizer):
    @classmethod
    def is_space(cls, ch):
        """空格类字符判断"""
        if ch in (" ", '\n', '\r', '\t'):
            return True
        return False

    @classmethod
    def is_alnum(cls, ch: str):
        """注意: string.isalnum() 函数, 会对汉字识别为 True. """
        if cls.is_cjk_character(ch):
            return False
        if ch.isalnum():
            return True
        return False

    @classmethod
    def is_cjk_character(cls, ch):
        """CJK类字符判断（包括中文字符也在此列）
        参考：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        """
        code = ord(ch)

        if 0x4E00 <= code <= 0x9FFF or \
            0x3400 <= code <= 0x4DBF or \
            0x20000 <= code <= 0x2A6DF or \
            0x2A700 <= code <= 0x2B73F or \
            0x2B740 <= code <= 0x2B81F or \
            0x2B820 <= code <= 0x2CEAF or \
            0xF900 <= code <= 0xFAFF or \
            0x2F800 <= code <= 0x2FA1F:
            return True
        return False

    def __init__(self):
        super().__init__()

    def normalize(self, text: str):
        """
        只保留, 字母之间, 字母和数字之间的空隔.
        """
        l = len(text)
        result = ''
        for i, c in enumerate(text):
            if i == 0 or i == l-1:
                if self.is_space(c):
                    continue
            else:
                last = text[i-1]
                next = text[i+1]
                if self.is_space(c):
                    if not all([self.is_alnum(last), self.is_alnum(next)]):
                        continue
            result += c
        return result


@TextNormalizer.register('clean_stop_char')
class CleanStopChar(TextNormalizer):
    def __init__(self, stop_char_set: Set[str] = None):
        super().__init__()
        self.stop_char_set = stop_char_set or _DEFAULT_STOP_CHAR_SET

    def normalize(self, text: str, stop_char_set: Set[str] = None):
        stop_char_set = stop_char_set or self.stop_char_set
        result = ''
        for c in text:
            if c not in stop_char_set:
                result += c
        return result


@TextNormalizer.register('full2semi_angle')
class Full2SemiAnlge(TextNormalizer):
    def __init__(self):
        super().__init__()

    def normalize(self, text: str):
        """全角字符转半角字符"""
        result = ''
        for c in text:
            code = ord(c)
            if code == 12288:
                code = 32
            elif 65281 <= code <= 65374:
                code -= 65248
            result += chr(code)
        return result


@TextNormalizer.register('zh2en_punctuation')
class Zh2EnPunctuation(TextNormalizer):
    def __init__(self, zh2en_punctuation_dict=None):
        super().__init__()
        self.zh2en_punctuation_dict = zh2en_punctuation_dict or _DEFAULT_ZH2EN_PUNCTUATION

    def normalize(self, text: str, zh2en_punctuation_dict=None):
        zh2en_punctuation_dict = zh2en_punctuation_dict or self.zh2en_punctuation_dict
        result = ''
        for c in text:
            result += zh2en_punctuation_dict.get(c, c)
        return result


def demo1():
    text_normalizer = TextNormalizer()

    text = '[手机] 我想买一个huawei p 40   手机。'
    # text = '2020年的第一天发烧，姨妈来难受???'

    normalizer_list = [
        'multiple_spaces_to_one',
        'clean_space_not_between_alnum',
        'clean_stop_char',
        'full2semi_angle',
        'zh2en_punctuation',

    ]
    result = text_normalizer.stacked_normalize(text, normalizer_list=normalizer_list)
    print(result)
    return


def demo2():
    normalizer_list = [
        'multiple_spaces_to_one',
        'clean_space_not_between_alnum',
        'clean_stop_char',
        'full2semi_angle',
        'zh2en_punctuation',

    ]
    text_normalizer = TextNormalizer(normalizer_list)

    # text = '[手机] 我想买一个huawei p 40   手机。'
    # text = '2020年的第一天发烧，姨妈来难受???'
    text = '得意学堂小枇杷,#元旦快乐##枇杷手法小结#每个娃都是有故事的娃。'

    result = text_normalizer.stacked_normalize(text)
    print(result)
    return


if __name__ == '__main__':
    # LSTM翻译()
    demo2()
