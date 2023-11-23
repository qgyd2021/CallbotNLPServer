#!/usr/bin/python3
# -*- coding: utf-8 -*-


class String2Number(object):
    """
    把字符串中表示数字的汉字转成数字.
    """

    _string2number = {
        '零': 0,
        '一': 1, '壹': 1,
        '二': 2, '贰': 2,
        '三': 3, '叁': 3,
        '四': 4, '肆': 4,
        '五': 5, '伍': 5,
        '六': 6, '陆': 6,
        '七': 7, '柒': 7,
        '八': 8, '捌': 8,
        '九': 9, '玖': 9,
        '十': 10, '拾': 10
    }

    @classmethod
    def transform(cls, text: str) -> str:
        result = list()
        for char in text:
            to_char = cls._string2number.get(char, char)
            to_char = str(to_char)
            result.append(to_char)

        result = ''.join(result)
        return result


def demo1():

    text_list = [
        '玖壹一八七',
        '5五1'
    ]
    for text in text_list:
        result = String2Number.transform(text)
        print(result)
    return


if __name__ == '__main__':
    demo1()
