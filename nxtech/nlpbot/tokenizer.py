# -*- encoding=UTF-8 -*-
import json
from typing import Any, Dict, List, Tuple

import jieba
import laonlp
import MeCab
from nltk.tokenize import _treebank_word_tokenizer, WordPunctTokenizer
from pythainlp.tokenize import word_tokenize

from toolbox.string.tokenization import FastTokenizer
from toolbox import pyidaungsu
from nxtech.common.params import Params
from nxtech.nlpbot.misc import Replacer
from project_settings import project_path


class Tokenizer(Params):
    """分词器"""
    def __init__(self):
        super().__init__()

    def tokenize(self, text: str) -> Tuple[List[str], List[bool]]:
        raise NotImplementedError


@Tokenizer.register('list')
class ListTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def tokenize(self, text: str):
        outlst = list(text)
        iswlst = [False] * len(outlst)
        return outlst, iswlst


@Tokenizer.register('indivisible')
class IndivisibleTokenizer(Tokenizer):
    """
    可用于
    1. 同义词替换.
    2. 将特定词分割为另外两个不相关的词.
    """
    def __init__(self, word_map_list: List[Dict[str, Any]], filename: str = None):
        super().__init__()

        if word_map_list is None and filename is None:
            raise AssertionError('one of word_map_list or filename must be assigned.')
        self.word_map_list_or_filename = word_map_list or filename
        self._word2word_list: Dict[str, List[str]] = dict()

        self._tokenizer = self._init_tokenizer()

    def _init_tokenizer(self):
        tokenizer = FastTokenizer()

        if isinstance(self.word_map_list_or_filename, str):
            with open(self.word_map_list_or_filename, 'r', encoding='utf-8') as f:
                for line in f:
                    word_map = json.loads(line)
                    word: str = word_map['word']
                    word_list: List[str] = word_map['word_list']
                    tokenizer.insert(word)
                    self._word2word_list[word] = word_list
        else:
            for word_map in self.word_map_list_or_filename:
                word: str = word_map['word']
                word_list: List[str] = word_map['word_list']
                tokenizer.insert(word)
                self._word2word_list[word] = word_list
        return tokenizer

    def tokenize(self, text: str):
        tmp_outlst, tmp_iswlst = self._tokenizer.tokenize(text)

        outlst, iswlst = list(), list()
        for out, isw in zip(tmp_outlst, tmp_iswlst):
            if isw is True:
                word_list = self._word2word_list[out]
                outlst.extend(word_list)
                iswlst.extend([True] * len(word_list))
            else:
                outlst.append(out)
                iswlst.append(False)
        return outlst, iswlst


@Tokenizer.register('forward_max_match')
class ForwardMaxMatchTokenizer(Tokenizer):
    def __init__(self, word_list: List[str] = None, filename: str = None):
        super().__init__()

        if word_list is None and filename is None:
            raise AssertionError('one of word_list or filename must be assigned.')
        self.word_list_or_filename = word_list or filename
        self._tokenizer = self._init_tokenizer()

    def _init_tokenizer(self):
        tokenizer = FastTokenizer()

        if isinstance(self.word_list_or_filename, str):
            with open(self.word_list_or_filename, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    tokenizer.insert(word)
        else:
            for word in self.word_list_or_filename:
                tokenizer.insert(word)
        return tokenizer

    def tokenize(self, text: str):
        outlst, iswlst = self._tokenizer.tokenize(text)
        return outlst, iswlst


@Tokenizer.register('character')
class CharacterTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def tokenize(self, text: str) -> Tuple[List[str], List[bool]]:
        outlst = list(text)
        iswlst = [False] * len(outlst)
        return outlst, iswlst


@Tokenizer.register('jieba')
class JiebaTokenizer(Tokenizer):
    """
    jieba 分词会将英文单词切开, 空隔会切开.
    """
    def __init__(self):
        super().__init__()

    def tokenize(self, text: str) -> Tuple[List[str], List[bool]]:
        outlst = jieba.lcut(text)
        iswlst = [False] * len(outlst)
        return outlst, iswlst


@Tokenizer.register('khmer')
class KhmerNltkTokenizer(Tokenizer):
    """
    有 BUG 弃用.
    分词:
    高棉语,
    """
    @staticmethod
    def demo1():
        text = 'ខ្ញុំមិនមានលុយទេ ។'
        result = KhmerNltkTokenizer().tokenize(text)
        print(result)
        return

    def __init__(self):
        super().__init__()
        from toolbox import khmernltk
        self.word_tokenize = khmernltk.word_tokenize

    def tokenize(self, text: str) -> Tuple[List[str], List[bool]]:
        outlst = self.word_tokenize(text)
        iswlst = [False] * len(outlst)
        return outlst, iswlst


@Tokenizer.register('lao')
class LaoTokenizer(Tokenizer):
    """
    分词:
    老挝语
    """
    def __init__(self):
        super().__init__()

    def tokenize(self, text: str) -> Tuple[List[str], List[bool]]:
        outlst = laonlp.word_tokenize(text)
        iswlst = [False] * len(outlst)
        return outlst, iswlst


@Tokenizer.register('mecab')
class MeCabTokenizer(Tokenizer):
    """
    比较有名的日语分词器.

    安装:
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple mecab-python3==1.0.4
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple unidic==1.1.0

    # 安装 unidic 后, 下载配置文件.
    python3 -m unidic download

    # 下载文件的网速可能比较慢, 可用讯雷下载:
    https://cotonoha-dic.s3-ap-northeast-1.amazonaws.com/unidic-3.1.0.zip

    使用时, 如下, 以指定配置文件.
    MeCab.Tagger("-Owakati -r /dev/null -d /data/tianxing/PycharmProjects/CallbotNLPServer/data/unidic-3.1.0/unidic")
    """
    def __init__(self, rawargs: str = None):
        super().__init__()

        self.rawargs = rawargs or "-Owakati -r /dev/null -d {}".format((project_path / "data/unidic-3.1.0/unidic").as_posix())
        self.wakati = MeCab.Tagger(self.rawargs)

    def tokenize(self, text: str) -> Tuple[List[str], List[bool]]:
        text = self.wakati.parse(text)
        outlst = text.split()
        iswlst = [False] * len(outlst)
        return outlst, iswlst

    def __getstate__(self):
        """
        pymysql.connect 对象不能被 pickle 序列化.

        __getstate__ 和 __setstate__ 方法是为了解决这个问题.
        https://docs.python.org/zh-cn/3.7/library/pickle.html#object.__getstate__

        默认情况, pickle 将类实例的 __dict__ 打包 (称作 state).
        在恢复对象时, 先创建一个对象实例, 再将 __dict__ 打包的 state 中的 k, v 赋值为实例属性.

        当定义了 __getstate__ 方法后, pickle 则按此方法打包实例.
        当定义了 __setstate__ 方法后, pickle 在恢复对象时, 调用此方法.
        """
        result = {
            **self.__dict__
        }
        result.pop('wakati')
        return result

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

        wakati = MeCab.Tagger(self.rawargs)

        setattr(self, 'wakati', wakati)
        return self


@Tokenizer.register('nltk')
class NltkTokenizer(Tokenizer):
    """
    分词:
    英语, 印尼语, 马来语, 越南语,
    """
    def __init__(self,
                 replacer: Replacer = None,
                 ):
        super().__init__()

        self.replacer = replacer

    def tokenize(self, text: str) -> Tuple[List[str], List[bool]]:
        outlst = _treebank_word_tokenizer.tokenize(text)
        iswlst = [False] * len(outlst)
        return outlst, iswlst


@Tokenizer.register('nltk_punct')
class NltkWordPunctTokenizer(Tokenizer):
    """
    分词:
    西班牙语,
    """
    def __init__(self):
        super().__init__()

        self.tokenizer = WordPunctTokenizer()

    def tokenize(self, text: str) -> Tuple[List[str], List[bool]]:
        outlst = self.tokenizer.tokenize(text)
        iswlst = [False] * len(outlst)
        return outlst, iswlst


@Tokenizer.register('pyidaungsu')
class PyidaungsuTokenizer(Tokenizer):
    """
    分词:
    缅甸语

    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple Cython==0.29.28
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple fasttext==0.8.1
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pyidaungsu==0.1.4
    """
    def __init__(self):
        super().__init__()

    def tokenize(self, text: str) -> Tuple[List[str], List[bool]]:
        outlst = pyidaungsu.tokenize(text)
        iswlst = [False] * len(outlst)
        return outlst, iswlst


@Tokenizer.register('pythainlp')
class PyThainlpTokenizer(Tokenizer):
    """
    分词:
    泰语
    """
    def __init__(self):
        super().__init__()

    def tokenize(self, text: str) -> Tuple[List[str], List[bool]]:
        outlst = word_tokenize(text)
        iswlst = [False] * len(outlst)
        return outlst, iswlst


@Tokenizer.register('tokenizer_pipe')
class TokenizerPipe(Tokenizer):
    def __init__(self, tokenizer_list: List[Tokenizer]):
        """
        :param tokenizer_list: 用多个分词器按顺序对句子分词.
        由于 jieba 分词后 iswlst 中都是 False, 一般情况应将其放在列表的最后一个.
        """
        super().__init__()

        self.tokenizer_list = tokenizer_list

    def _tokenize(self, outlst: List[str], iswlst: List[bool], tokenizer: Tokenizer):
        ret_outlst, ret_iswlst = list(), list()
        for out, isw in zip(outlst, iswlst):
            if isw is False:
                tmp_outlst, tmp_iswlst = tokenizer.tokenize(out)
                ret_outlst.extend(tmp_outlst)
                ret_iswlst.extend(tmp_iswlst)
            else:
                ret_outlst.append(out)
                ret_iswlst.append(isw)
        return ret_outlst, ret_iswlst

    def tokenize(self, text: str):
        outlst = [text]
        iswlst = [False]

        for tokenizer in self.tokenizer_list:
            outlst, iswlst = self._tokenize(outlst, iswlst, tokenizer)

        return outlst, iswlst


def demo1():
    tokenizer = Tokenizer.from_json(
        params={
            'type': 'indivisible',
            'word_map_list': [
                {
                    'word': '今天天气',
                    'word_list': ['今日', '天气'],
                }
            ]
        }
    )
    print(tokenizer)
    text = '今天天气真好'
    result = tokenizer.tokenize(text)
    print(result)
    return


def demo2():
    tokenizer = Tokenizer.from_json(
        params={
            'type': 'jieba',
        }
    )
    print(tokenizer)
    text = '今天天气真好'
    result = tokenizer.tokenize(text)
    print(result)
    return


def demo3():
    tokenizer = Tokenizer.from_json(
        params={
            'type': 'forward_max_match',
            'word_list': ['今天', '天气']
        }
    )
    print(tokenizer)
    text = '今天天气真好'
    result = tokenizer.tokenize(text)
    print(result)
    return


def demo4():
    tokenizer = Tokenizer.from_json(
        params={
            'type': 'tokenizer_pipe',
            'tokenizer_list': [
                {
                    'type': 'indivisible',
                    'word_map_list': [
                        {
                            'word': '今天天气',
                            'word_list': ['今日', '天气'],
                        }
                    ]
                },
                {
                    'type': 'forward_max_match',
                    'word_list': ['今天', '天气']
                },
                {
                    'type': 'jieba',
                }
            ]
        }
    )

    text = '今天天气是真的真的不错啊'
    result = tokenizer.tokenize(text)
    print(result)
    return


def demo5():
    tokenizer = Tokenizer.from_json(
        params={
            'type': 'mecab',
        }
    )

    text = '天気がいいから、散歩しましょう'
    result = tokenizer.tokenize(text)
    print(result)
    return


def demo6():
    import pickle

    tokenizer = Tokenizer.from_json(
        params={
            'type': 'mecab',
        }
    )

    pkl = pickle.dumps(tokenizer)
    print(pkl)
    return


def demo7():
    KhmerNltkTokenizer.demo1()
    return


if __name__ == '__main__':
    # demo1()
    # demo2()
    # demo3()
    # demo4()
    # demo5()
    # demo6()
    demo7()
