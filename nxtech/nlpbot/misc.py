#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
句子相似度衡量, 涉及:
同义词替换,
停用词去除,
相似度算法
"""
from collections import defaultdict
import json
import os
from typing import Any, Dict, List, Tuple

from arabicstemmer import arabic_stemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.Stemmer.Filter import TextNormalizer
import zhconv

from nxtech.common.params import Params
from project_settings import project_path
from toolbox.nltk.replacer.regexp import RegexpReplacer, RepeatReplacer, contraction_replacer, repeat_replacer
from toolbox.re.string2number import String2Number as string2number_convert


class Preprocess(Params):
    def __init__(self):
        super().__init__()

    def process(self, text: str) -> str:
        raise NotImplementedError


@Preprocess.register('punctuation')
class PunctuationPreprocess(Preprocess):
    def __init__(self, patterns: List[Tuple[str, str]] = None):
        super().__init__()
        patterns = patterns or [
            ('[,.?!$&@():：，、。？！“”‘’\'\"<>《》{}｛｝\[\]]', ' '),
        ]
        self.replacer = RegexpReplacer(
            patterns=patterns
        )

    def process(self, text: str) -> str:
        return self.replacer.replace(text)


@Preprocess.register('en_punctuation')
class EnPunctuationPreprocess(Preprocess):
    """
    英文的 don't, don’t 中的标点符号不应去除.
    """
    def __init__(self, patterns: List[Tuple[str, str]] = None):
        super().__init__()

        patterns = patterns or [
            ('[,.?!$&@():：，、。？！“”‘\"<>《》{}｛｝\[\]]', ' '),
        ]
        self.replacer = RegexpReplacer(
            patterns=patterns
        )

    def process(self, text: str) -> str:
        return self.replacer.replace(text)


@Preprocess.register('contraction')
class ContractionPreprocess(Preprocess):
    def __init__(self):
        super().__init__()

    def process(self, text: str) -> str:
        return contraction_replacer.replace(text)


@Preprocess.register('en_repeat')
class ENRepeatPreprocess(Preprocess):
    """
    很费时间, 不要使用.

    句子处理:
    loooove -> love
    ooooh -> ooh
    天天开心 -> 天开心
    """
    def __init__(self):
        super().__init__()

    def process(self, text: str) -> str:
        return repeat_replacer.replace(text)


@Preprocess.register('cn_repeat')
class CNRepeatPreprocess(Preprocess):
    """
    不使用 wordnet, 速度较快.

    将连续的多个相同字符替换为只有两个.

    句子处理:
    my loooove is life -> my loove is life
    loooove -> loove
    oooooh -> ooh
    要要天天天天开心啊啊啊啊啊哈哈哈 -> 要要天天开心啊啊哈哈
    """
    def __init__(self):
        super().__init__()
        self.repeat_replacer = RepeatReplacer(
            repeat_regexp='(\\w*)(\\w)\\2\\2(\\w*)',
            repl='\\1\\2\\2\\3',
            use_wordnet=False,
        )

    def process(self, text: str) -> str:
        return repeat_replacer.replace(text)


@Preprocess.register('do_lowercase')
class DoLowercase(Preprocess):
    def __init__(self):
        super().__init__()

    def process(self, text: str) -> str:
        result = str(text).lower()
        return result


@Preprocess.register('vietnamese_lowercase')
class VietnameseLowercase(Preprocess):
    alphabet_lower = 'àáâãăạảấầẩẫậắằẳặđèéêẹẻẽếềểễệìíĩỉịòóôõơọỏốồổỗộớờởỡợùúũưụủứừửữựýỳỷỹ'
    alphabet_upper = 'ÀÁÂÃĂẠẢẤẦẨẪẬẮẰẲẶĐÈÉÊẸẺẼẾỀỂỄỆÌÍĨỈỊÒÓÔÕƠỌỎỐỒỔỖỘỚỜỞỠỢÙÚŨƯỤỦỨỪỬỮỰÝỲỶỸ'

    lowercase_map = {
        'à': 'a',
        'á': 'a',
        'â': 'a',
        'ã': 'a',
        'ă': 'a',
        'ạ': 'a',
        'ả': 'a',
        'ấ': 'a',
        'ầ': 'a',
        'ẩ': 'a',
        'ẫ': 'a',
        'ậ': 'a',
        'ắ': 'a',
        'ằ': 'a',
        'ẳ': 'a',
        'ặ': 'a',

        'đ': 'd',

        'è': 'e',
        'é': 'e',
        'ê': 'e',
        'ẹ': 'e',
        'ẻ': 'e',
        'ẽ': 'e',
        'ế': 'e',
        'ề': 'e',
        'ể': 'e',
        'ễ': 'e',
        'ệ': 'e',

        'ì': 'i',
        'í': 'i',
        'ĩ': 'i',
        'ỉ': 'i',
        'ị': 'i',

        'ò': 'o',
        'ó': 'o',
        'ô': 'o',
        'õ': 'o',
        'ơ': 'o',
        'ọ': 'o',
        'ỏ': 'o',
        'ố': 'o',
        'ồ': 'o',
        'ổ': 'o',
        'ỗ': 'o',
        'ộ': 'o',
        'ớ': 'o',
        'ờ': 'o',
        'ở': 'o',
        'ỡ': 'o',
        'ợ': 'o',

        'ù': 'u',
        'ú': 'u',
        'ũ': 'u',
        'ư': 'u',
        'ụ': 'u',
        'ủ': 'u',
        'ứ': 'u',
        'ừ': 'u',
        'ử': 'u',
        'ữ': 'u',
        'ự': 'u',

        'ý': 'y',
        'ỳ': 'y',
        'ỷ': 'y',
        'ỹ': 'y',
    }

    def __init__(self, alphabet_map: Dict[str, str] = None):
        super().__init__()

        if alphabet_map is not None:
            self.lowercase_map.update(alphabet_map)

    def process(self, text: str) -> str:
        text = str(text).lower()

        result = ''
        for c in text:
            c = self.lowercase_map.get(c, c)
            result += c
        return result


@Preprocess.register('strip')
class Strip(Preprocess):
    def __init__(self, chars: str = None):
        super().__init__()

        self.chars = chars

    def process(self, text: str) -> str:
        result = str(text).strip(self.chars)
        return result


@Preprocess.register('zh_conv')
class ZhConv(Preprocess):
    """简体繁体转换"""
    locale_enums = (
        'zh-hans', 'zh-hant', 'zh-cn', 'zh-sg',
        'zh-tw', 'zh-hk', 'zh-my', 'zh-mo'
    )

    def __init__(self, locale: str = 'zh-cn', update: Dict[str, str] = None):
        """
        :param locale: 需要转到哪种中文. 如 `zh-cn` 表示, 将文字转到中文简体.
        :param update:
        """
        super().__init__()
        if locale not in self.locale_enums:
            raise ValueError('locale: {} not in ({})'.format(locale, self.locale_enums))
        self.locale = locale
        self.update = update

    def process(self, text: str) -> str:
        result = zhconv.convert(
            text,
            locale=self.locale,
            update=self.update
        )
        return result


@Preprocess.register('string2number')
class String2Number(Preprocess):
    def __init__(self):
        super().__init__()

    def process(self, text: str) -> str:
        return string2number_convert.transform(text)


class Replacer(Params):
    def __init__(self):
        super().__init__()

    def replace(self, text_list: List[str]) -> List[List[str]]:
        raise NotImplementedError


@Replacer.register('synonym')
class SynonymReplacer(Replacer):
    def __init__(self,
                 synonyms: List[Dict[str, Any]] = None,
                 filename: str = None,
                 do_lowercase: bool = True):
        super().__init__()
        if synonyms is None and filename is None:
            raise AssertionError('one of synonyms and filename must be assigned.')
        self.synonyms_or_filename = synonyms or os.path.join(project_path, filename)
        self.do_lowercase = do_lowercase

        self._synonym_dict = self._init_synonym_dict()

    def _init_synonym_dict(self):
        """一个同义词可对应多个标准词"""
        result = defaultdict(list)
        if isinstance(self.synonyms_or_filename, str):
            with open(self.synonyms_or_filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line)
                    entity: str = line['entity']
                    synonyms: List[str] = line['synonyms']
                    if self.do_lowercase:
                        entity = str(entity).lower()
                        synonyms = [str(synonym).lower() for synonym in synonyms]
                    result[entity].append(entity)
                    for synonym in synonyms:
                        result[synonym].append(entity)
        else:
            for line in self.synonyms_or_filename:
                entity: str = line['entity']
                synonyms: List[str] = line['synonyms']
                if self.do_lowercase:
                    entity = str(entity).lower()
                    synonyms = [str(synonym).lower() for synonym in synonyms]

                result[entity].append(entity)
                for synonym in synonyms:
                    result[synonym].append(entity)
        return result

    def replace(self, text_list: List[str]) -> List[List[str]]:
        candidate_list: List[List[str]] = list()
        for text in text_list:
            if text in self._synonym_dict:
                entity_list = self._synonym_dict[text]
                if len(candidate_list) == 0:
                    for entity in entity_list:
                        candidate_list.append([entity])
                else:
                    candidate_list2 = list()
                    for candidate in candidate_list:
                        for entity in entity_list:
                            candidate_list2.append(candidate + [entity])
                    candidate_list = candidate_list2
            else:
                if len(candidate_list) == 0:
                    candidate_list.append([text])
                else:
                    for candidate in candidate_list:
                        candidate.append(text)
        return candidate_list


@Replacer.register('wordnet_lemma')
class WordNetLemmaReplacer(Replacer):
    def __init__(self):
        super().__init__()
        self.lemmatizer = WordNetLemmatizer()

    def replace(self, text_list: List[str]) -> List[List[str]]:
        result = list()
        for text in text_list:
            text = self.lemmatizer.lemmatize(text, pos='n')
            text = self.lemmatizer.lemmatize(text, pos='v')
            result.append(text)
        return [result]


@Replacer.register('sastrawi_stemmer')
class SastrawiStemmerReplacer(Replacer):
    """indonesian"""
    def __init__(self):
        super().__init__()

        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()

    def replace(self, text_list: List[str]) -> List[List[str]]:
        result = list()
        for text in text_list:
            text = TextNormalizer.normalize_text(text)

            if self.stemmer.cache.has(text):
                stemmed_text = self.stemmer.cache.get(text)
            else:
                stemmed_text = self.stemmer.delegatedStemmer.stem(text)
                self.stemmer.cache.set(text, stemmed_text)
            result.append(stemmed_text)
        return [result]


@Replacer.register('arabic_stemmer')
class ArabicStemmerReplacer(Replacer):
    """arabic"""
    def __init__(self):
        super().__init__()

        self.stemmer = arabic_stemmer.ArabicStemmer()

    def replace(self, text_list: List[str]) -> List[List[str]]:
        result = list()
        for text in text_list:
            text = self.stemmer.stemWord(text)
            result.append(text)
        return [result]


class Filter(Params):
    def __init__(self):
        super().__init__()

    def filter(self, text_list: List[str]) -> List[str]:
        raise NotImplementedError


@Filter.register('stopwords')
class StopWordsFilter(Filter):
    def __init__(self,
                 stopwords: List[str] = None,
                 filename: str = None,
                 ):
        super().__init__()

        if stopwords is None and filename is None:
            raise AssertionError('one of stopwords and filename must be assigned.')
        self.stopwords_or_filename = stopwords or os.path.join(project_path, filename)
        self.stopwords = self._init_stopwords()

    def _init_stopwords(self):
        result = set()
        if isinstance(self.stopwords_or_filename, str):
            with open(self.stopwords_or_filename, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    result.add(word)
        else:
            result.update(set(self.stopwords_or_filename))
        return result

    def filter(self, text_list: List[str]) -> List[str]:
        result = [text for text in text_list if text not in self.stopwords]
        return result


@Filter.register('nltk')
class NltkStopWordsFilter(Filter):
    def __init__(self, language: str):
        super().__init__()

        self.language = language
        self.stopwords = stopwords.words('english')

    def filter(self, text_list: List[str]) -> List[str]:
        result = [text for text in text_list if text not in self.stopwords]
        return result


def demo1():
    replacer = Replacer.from_json(
        params={
            'type': 'synonym',
            'synonyms': [
                {
                    'entity': '分期',
                    'synonyms': ['分气', '芬期']
                },
                # {
                #     'entity': '芬汽',
                #     'synonyms': ['分气', '芬期']
                # }
            ]
        }
    )

    text_list = ['可', '不', '可以', '芬期', '还', '.']
    result = replacer.replace(text_list)
    print(result)
    return


def demo2():
    word_filter = Filter.from_json(
        params={
            'type': 'stopwords',
            'stopwords': ['可', '不', '.']
        }
    )

    text_list = ['可', '不', '可以', '分气', '还', '.']
    result = word_filter.filter(text_list)
    print(result)
    return


def demo3():
    sentence = 'Perekonomian Indonesia sedang dalam pertumbuhan yang membanggakan'
    # sentence = 'Mereka meniru-nirukannya'

    # replacer = SastrawiStemmerReplacer()
    replacer = SastrawiStemmerReplacer.from_json()

    text_list = sentence.split()
    result = replacer.replace(text_list)
    print(result)
    return


if __name__ == '__main__':
    demo1()
    # demo2()
    # demo3()
