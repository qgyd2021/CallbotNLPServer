#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import time
from typing import List

# from pymysql.converters import escape_string
import requests


_escape_table = [chr(x) for x in range(128)]
_escape_table[0] = "\\0"
_escape_table[ord("\\")] = "\\\\\\\\"
_escape_table[ord("\n")] = "\\n"
_escape_table[ord("\r")] = "\\r"
_escape_table[ord("\032")] = "\\Z"
_escape_table[ord('"')] = '\\\\\\"'
_escape_table[ord("'")] = "\\\\\\'"


def escape_string(value, mapping=None):
    """escapes *value* without adding quote.

    Value should be unicode
    """
    return value.translate(_escape_table)


class GoogleTranslate(object):
    """
    英语: en
    汉语: zh
    法语: fr
    日语: ja
    德语: de
    印尼语: id
    """
    LANGUAGES = {
        'af': 'afrikaans',
        'sq': 'albanian',
        'am': 'amharic',
        'ar': 'arabic',
        'hy': 'armenian',
        'az': 'azerbaijani',
        'eu': 'basque',
        'be': 'belarusian',
        'bn': 'bengali',
        'bs': 'bosnian',
        'bg': 'bulgarian',
        'ca': 'catalan',
        'ceb': 'cebuano',
        'ny': 'chichewa',
        'zh-cn': 'chinese (simplified)',
        'zh-tw': 'chinese (traditional)',
        'co': 'corsican',
        'hr': 'croatian',
        'cs': 'czech',
        'da': 'danish',
        'nl': 'dutch',
        'en': 'english',
        'eo': 'esperanto',
        'et': 'estonian',
        'tl': 'filipino',
        'fi': 'finnish',
        'fr': 'french',
        'fy': 'frisian',
        'gl': 'galician',
        'ka': 'georgian',
        'de': 'german',
        'el': 'greek',
        'gu': 'gujarati',
        'ht': 'haitian creole',
        'ha': 'hausa',
        'haw': 'hawaiian',
        'iw': 'hebrew',
        'he': 'hebrew',
        'hi': 'hindi',
        'hmn': 'hmong',
        'hu': 'hungarian',
        'is': 'icelandic',
        'ig': 'igbo',
        'id': 'indonesian',
        'ga': 'irish',
        'it': 'italian',
        'ja': 'japanese',
        'jw': 'javanese',
        'kn': 'kannada',
        'kk': 'kazakh',
        'km': 'khmer',
        'ko': 'korean',
        'ku': 'kurdish (kurmanji)',
        'ky': 'kyrgyz',
        'lo': 'lao',
        'la': 'latin',
        'lv': 'latvian',
        'lt': 'lithuanian',
        'lb': 'luxembourgish',
        'mk': 'macedonian',
        'mg': 'malagasy',
        'ms': 'malay',
        'ml': 'malayalam',
        'mt': 'maltese',
        'mi': 'maori',
        'mr': 'marathi',
        'mn': 'mongolian',
        'my': 'myanmar (burmese)',
        'ne': 'nepali',
        'no': 'norwegian',
        'or': 'odia',
        'ps': 'pashto',
        'fa': 'persian',
        'pl': 'polish',
        'pt': 'portuguese',
        'pa': 'punjabi',
        'ro': 'romanian',
        'ru': 'russian',
        'sm': 'samoan',
        'gd': 'scots gaelic',
        'sr': 'serbian',
        'st': 'sesotho',
        'sn': 'shona',
        'sd': 'sindhi',
        'si': 'sinhala',
        'sk': 'slovak',
        'sl': 'slovenian',
        'so': 'somali',
        'es': 'spanish',
        'su': 'sundanese',
        'sw': 'swahili',
        'sv': 'swedish',
        'tg': 'tajik',
        'ta': 'tamil',
        'te': 'telugu',
        'th': 'thai',
        'tr': 'turkish',
        'uk': 'ukrainian',
        'ur': 'urdu',
        'ug': 'uyghur',
        'uz': 'uzbek',
        'vi': 'vietnamese',
        'cy': 'welsh',
        'xh': 'xhosa',
        'yi': 'yiddish',
        'yo': 'yoruba',
        'zu': 'zulu',
    }

    url = 'https://translate.google.cn/_/TranslateWebserverUi/data/batchexecute?rpcids=MkEWBc&f.sid=-4896104385845517761&bl=boq_translate-webserver_20211103.08_p0&hl=zh-CN&soc-app=1&soc-platform=1&soc-device=1&_reqid=16934490&rt=c'

    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "content-type": "application/x-www-form-urlencoded;charset=UTF-8",
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36'
    }

    @classmethod
    def requests(cls, text: str, dest: str, src: str = 'auto', retry: int = 100, interval: int = 2):
        f_req = r"""[[["MkEWBc","[[\"{}\",\"{}\",\"{}\",true],[null]]",null,"generic"]]]""".format(
            text, src, dest
        )
        # print(f_req)
        data = {
            "f.req": f_req
        }

        count = 0
        error = None
        while True:
            if count >= retry:
                raise error
            try:
                resp = requests.post(cls.url, headers=cls.headers, data=data, timeout=2)
                return resp
            except Exception as e:
                error = e
                print(error)
                time.sleep(interval)
                interval += 1
                count += 1
                continue

    @classmethod
    def batch_translate(cls, text_list: List[str], dest: str, src: str = 'auto') -> List[str]:
        text_list = [text.strip() for text in text_list]
        text_list = [escape_string(text) for text in text_list]

        l = len(text_list)
        text = '\\\\n'.join(text_list)
        result = GoogleTranslate.translate(
            text=text,
            dest=dest,
            src=src,
            do_escape=False
        )
        if len(result) == 1 and l != 1:
            result = result[0].split('\n')
        return result

    @classmethod
    def _translate(cls):
        return

    @classmethod
    def translate(cls, text: str, dest: str, src: str = 'auto', do_escape=True) -> List[str]:
        if do_escape:
            text = escape_string(text)
        if not isinstance(text, str):
            raise AssertionError('text must be a string. ')
        text = text.strip()
        if len(text) == 0:
            raise AssertionError('the length if text must greater than 0')
        dest = dest.lower()
        src = src.lower()
        if src != 'auto' and src not in cls.LANGUAGES:
            raise AssertionError('invalid source language')
        if dest not in cls.LANGUAGES:
            raise AssertionError('invalid destination language')

        for _ in range(10):
            resp = cls.requests(
                text=text,
                dest=dest,
                src=src,
            )
            try:
                result = cls._decode_for_auto(resp.text)
                break
            except LookupError as e:
                continue
            except Exception as e:
                print('-{}-'.format(text))
                print(resp.text)
                raise e
        else:
            print('LookupError: ', text)
            result = text
        return result

    @staticmethod
    def _decode_for_auto(text: str):
        lines = text.split('\n')
        js = list()
        for line in lines:
            try:
                line = json.loads(line)
                js.append(line)
            except json.decoder.JSONDecodeError as e:
                continue

        if js[1][0][0] == 'wrb.fr':
            if js[1][0][2] is None:
                raise LookupError()
            js = json.loads(js[1][0][2])
            js = js[1][0][0]

            if js[5] is None:
                js = [js]
            else:
                js = js[5]
        else:
            js = js[1][0][0][5]
        result = list()

        for line in js:
            if line[0] == '\n':
                continue
            result.append(line[0].lower())
        return result


def demo1():
    import time
    begin = time.time()
    text_list = [
        '我付不起',
        '暂时无法还钱',
        '为什么这么多',
        '怎么这么多'
    ]
    for text in text_list:
        result = GoogleTranslate.translate(
            text=text,
            dest='id'
        )
        print(result)

    cost = time.time() - begin
    print(cost)
    return


def demo2():
    import time
    begin = time.time()

    text_list = ['Thank you']
    result = GoogleTranslate.batch_translate(
        text_list=text_list,
        dest='my'
    )
    print(result)

    cost = time.time() - begin
    print(cost)
    return


def demo3():
    import time
    begin = time.time()

    text_list = [

    ]

    print(len(''.join(text_list)))
    result = GoogleTranslate.batch_translate(
        text_list=text_list,
        dest='en',
        src='th'
    )
    print(result)

    cost = time.time() - begin
    print(cost)
    return


def demo4():
    text = '\\做为鸡要有道德你不可以酱紫'
    print(text)
    text = escape_string(text)
    print(text)
    return


if __name__ == '__main__':
    # demo1()
    demo2()
    # demo3()
    # demo4()
