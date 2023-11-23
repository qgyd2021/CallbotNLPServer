#!/usr/bin/python3
# -*- coding: utf-8 -*-
import re
import unicodedata
import six


class Character(object):
    f_unknown = 'unknown'
    f_is_alnum = 'is_alnum'
    f_is_alpha = 'is_alpha'
    f_is_num = 'is_num'
    f_is_space = 'is_space'
    f_is_hyphens = 'is_hyphens'
    f_is_punctuation = 'is_punctuation'
    f_is_cjk_character = 'is_cjk_character'
    f_is_jap_character = 'is_jap_character'
    f_is_russian_character = 'is_russian_character'

    @classmethod
    def is_alnum(cls, ch: str):
        """注意: string.isalnum() 函数, 会对汉字识别为 True. """
        if cls.is_cjk_character(ch):
            return False
        if ch.isalnum():
            return True
        return False

    @classmethod
    def is_alpha(cls, ch: str):
        if cls.is_cjk_character(ch):
            return False
        if ch.isalpha():
            return True
        return False

    @staticmethod
    def is_control(ch):
        """控制类字符判断"""
        if ch in ('\t', '\n', '\r'):
            return False
        return unicodedata.category(ch) in ("Cc", "Cf")

    @classmethod
    def is_num(cls, ch: str):
        if cls.is_cjk_character(ch):
            return False
        if ch.isdigit():
            return True
        return False

    @classmethod
    def is_space(cls, ch):
        """空格类字符判断"""
        if ch in (" ", '\n', '\r', '\t'):
            return True
        if unicodedata.category(ch) == 'Zs':
            return True
        return False

    @classmethod
    def is_hyphens(cls, ch):
        """
        是否为连字符, `-` 匹配减号.
        + : 43
        - : 45
        """
        code = ord(ch)
        if code in (43, 45):
            return True
        return False

    @classmethod
    def is_punctuation(cls, ch):
        """标点符号类字符判断（全/半角均在此内）"""
        code = ord(ch)
        if 33 <= code <= 47 or \
            58 <= code <= 64 or \
            91 <= code <= 96 or \
            123 <= code <= 126 or \
            unicodedata.category(ch).startswith("P"):
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

    @classmethod
    def is_jap_character(cls, ch):
        code = ord(ch)
        if 0x3040 <= code <= 0x309F or \
               0x30A0 <= code <= 0x30FF or \
               0x31F0 <= code <= 0x31FF:
            return True
        return False

    @classmethod
    def is_russian_character(cls, ch):
        code = ord(ch)
        if 1040 <= code <= 1104:
            return True
        return False

    @staticmethod
    def convert_to_unicode(text):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if six.PY3:
            if isinstance(text, str):
                return text
            elif isinstance(text, bytes):
                return text.decode("utf-8", "ignore")
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        elif six.PY2:
            if isinstance(text, str):
                return text.decode("utf-8", "ignore")
            elif isinstance(text, unicode):
                return text
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        else:
            raise ValueError("Not running on Python2 or Python 3?")


class LowerCase(object):
    confuse_map = {
        # 俄语
        'Й': 'И',
        'й': 'и',
        'ѐ': 'е',
        'ё': 'е',
        'ѓ': 'г',
        'ї': 'і',

        # 西语
        'á': 'a',
        'é': 'e',
        'í': 'i',
        'ó': 'o',
        'ú': 'u',
        'ü': 'u',
        'ñ': 'n',
    }

    @classmethod
    def lowercase(cls, string):
        """转小写不应改变字符串的长度"""
        string = str(string).lower()
        result = ''
        for c in string:
            code = ord(c)

            # 俄语转小写.
            if 1040 <= code <= 1072:
                c = chr(ord(c) + 32)

            # 混淆字转换
            flag = cls.confuse_map.get(c)
            if flag is not None:
                c = flag

            result += c

        if len(string) != len(result):
            raise AssertionError('this method should not change the char num. '
                                 'string: {}, result: {}'.format(string, result))
        return result


class Pattern(object):
    """
    \d              匹配任意数字符, 等价于 [0-9].
    \s              匹配任意空白字符, 等价于 [\t\n\r\f] 包括空隔.
    re{n,m}         匹配 n 到 m 次由前面的正则表达式定义的片段, 贪婪方式.
    """
    alp_num_ch = '[A-Z0-9a-z\u4e00-\u9fa5]+'          # 提取中文数字字母
    alp_num_or_others = '[^A-Z0-9a-z]|[A-Z0-9a-z]+'   # 用于在 ' '.join() 中分融数字字母与其它字符.
    brackets = '\(.*?\)'                              # 识别括号
    hw_ry_xy = '华为|荣耀|小艺'
    p_pattern = '[a-z]\d{1,2}\s+p\d{1,2}'
    pro_pattern = '([a-z]+\s*\d{1,2})\s+(p\d{1,2})'
    any_blanks = '\s+'
    square_brackets = '\[.*?\]'                       # 识别方括号
    regex_dsw_find = '\\\\[dDsSwW][\+\*]?'            # 从如 `\d+左右` 中去除 `\d+`. 用于正则索引的获取.


class ValidPeriod(object):
    """有效期"""
    l_compare_lt = 'l_compare_lt'
    l_compare_gt = 'l_compare_gt'
    l_time = 'l_time'
    l_time_unit = 'l_time_unit'

    # 每个子正则表达式 (形如: `<?label>pattern`) 都包含一个标签.
    l_compare_lt_prefix_regex = f'?<{l_compare_lt}>不超过|没到|不到|少于'
    l_compare_gt_prefix_regex = f'?<{l_compare_gt}>超|超过|超过了|大于|不止'

    l_compare_lt_suffix_regex = f'?<{l_compare_lt}>没到|不到|内|以内|之内'
    l_compare_gt_suffix_regex = f'?<{l_compare_gt}>以上|不止'

    l_time_regex = f'?<{l_time}>[两|壹|零|一|二|三|四|五|六|七|八|九|十|百|千\d]+'
    # l_time_unit_regex = f'?<{l_time_unit}>年|个月|周|天|星期'
    l_time_unit_regex = f'?<{l_time_unit}>(?:个)?年|个月|周|天|日|星期|个星期'

    # 正则表达式: 识别 -> 不到十天, 一个星期, 超七天后, 七日内, 第七天, 超过了七天, 不止5天 等. 类似的模式.
    pattern1 = f'(?:({l_compare_lt_prefix_regex})|({l_compare_gt_prefix_regex}))?\s*({l_time_regex})\s*({l_time_unit_regex})\s*(?:({l_compare_lt_suffix_regex})|({l_compare_gt_suffix_regex}))?'

    # 正则表达式: 识别 -> 上个月5号, 这个月14日 等. 日期模式
    pass

    @staticmethod
    def demo1():
        """
        例句:
        一个星期, 超七天后, 七日内, 第七天, 超过了七天, 不止5天

        # 以下句子都是从标注数据中找出的有效期, 将来也许需要处理这些.
        刚买2天, 昨天取的, 昨天到货, 签收后的第二天, 签收后七天内, 前两天, 货还没发, 用了几天
        :return:
        """

        string = "5天不止"

        ret = ValidPeriod.valid_period_parse(string)
        print(ret)
        return

    @staticmethod
    def time_convert(time_string: str):
        base_num_dict = {
            '十': 10,
            '百': 100,
            '千': 1000,
        }
        d = {
            '壹': 1,
            '两': 2,
            '零': 0,
            '一': 1,
            '二': 2,
            '三': 3,
            '四': 4,
            '五': 5,
            '六': 6,
            '七': 7,
            '八': 8,
            '九': 9,
        }
        result = 0
        tmp = ''
        for c in time_string:
            if c.isdecimal():
                tmp += c
                continue

            base_num = base_num_dict.get(c, None)
            if base_num is not None:
                if len(tmp) == 0:
                    result += base_num
                elif len(tmp) == 1:
                    result += base_num * int(tmp)
                    print(result)
                    tmp = ''
                else:
                    pass
            else:
                tmp += str(d.get(c, ''))
        else:
            result += int(tmp)
        return result

    @staticmethod
    def time_unit_convert(time_unit_string: str):
        d = {
            '天': 1,
            '日': 1,
            '周': 7,
            '星期': 7,
            '个星期': 7,
            '个月': 30,
            '年': 365,
        }
        result = d.get(time_unit_string, 1)
        return result

    @staticmethod
    def get_pattern_label(pattern: str):
        """
        子正则表达式都包含了一个标签,
        :param pattern:
        :return:
        """
        pattern_inner = re.compile(r'\?<(.*?)>')
        label_name_list = re.findall(pattern=pattern_inner, string=pattern)
        return label_name_list

    @staticmethod
    def clean_pattern_label(pattern: str):
        pattern_inner = re.compile(r'\?<.*?>')
        result = re.sub(pattern=pattern_inner, repl='', string=pattern)
        return result

    @classmethod
    def valid_period_parse(cls, string: str) -> (int, dict) or (None, dict):
        """cls.pattern1"""
        label_name_list, label_string_list = cls._search_label_list(string, cls.pattern1)
        days, detail = cls._estimate_days(label_name_list, label_string_list)
        return days, detail

    @classmethod
    def _estimate_days(cls, label_name_list, label_string_list) -> (int, dict) or (None, dict):
        """当一个标签都没有识别到时, 两 list 为空. 返回结果为 0. """
        bias = 0
        scale = 1
        main_time = 0

        for label_name, label_string in zip(label_name_list, label_string_list):
            if label_name == cls.l_compare_lt:
                # bias = -1
                pass
            elif label_name == cls.l_compare_gt:
                bias = 1
            elif label_name == cls.l_time:
                main_time += cls.time_convert(label_string)
            elif label_name == cls.l_time_unit:
                scale = cls.time_unit_convert(label_string)
            else:
                pass
        days = main_time * scale + bias
        detail = {
            'main_time': main_time,
            'scale': scale,
            'bias': bias
        }

        if len(label_name_list) == 0 or len(label_string_list) == 0:
            return None, detail

        return days, detail

    @classmethod
    def _search_label_list(cls, string: str, pattern: str) -> (list, list):
        label_name_list = cls.get_pattern_label(pattern)
        pattern = cls.clean_pattern_label(pattern)
        match = re.search(pattern=pattern, string=string)
        if match is None:
            return list(), list()
        label_string_list = match.groups()

        new_label_name_list, new_label_string_list = list(), list()
        for label_name, label_string in zip(label_name_list, label_string_list):
            if label_string is None:
                continue
            new_label_name_list.append(label_name)
            new_label_string_list.append(label_string)

        return new_label_name_list, new_label_string_list

    def __init__(self):
        pass
