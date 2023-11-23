#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import List


class RequestContext(object):
    def __init__(self,
                 product_id: str, scene_id: str,
                 node_id: str, env: str,
                 text: str, debug: bool = False):
        self.product_id = product_id
        self.scene_id = scene_id
        self.node_id = node_id
        self.env = env

        self._text = text

        self.result: List[dict] = list()

        self._debug = debug
        self._report: str = ''

    @property
    def text(self):
        return self._text

    @property
    def debug(self):
        return self._debug

    @property
    def report(self):
        return self._report

    def export_report(self, filename: str = None):
        if filename is None:
            filename = 'temp.md'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.report)
        return None

    def append_row(self, row: str):
        if not self.debug:
            return None
        self._report += row
        self._report += '\n'

        return None

    def append_table(self, table: List[dict]):
        """
        |  表头   | 表头  |
        |  ----  | ----  |
        | 单元格  | 单元格 |
        | 单元格  | 单元格 |

        """
        if not self.debug:
            return None

        headers = list()
        for row in table:
            for k, v in row.items():
                if k not in headers:
                    headers.append(k)

        row = ' | '.join(headers)
        self.append_row('| {} |'.format(row))
        row = ' | '.join(['----'] * len(headers))
        self.append_row('| {} |'.format(row))

        for data in table:
            row = '| '
            for header in headers:
                value = data.get(header)
                if value is None:
                    row += ' '
                else:
                    row += str(value)
                row += ' | '
            row += ' |'
            self.append_row(row)

        return None


def demo1():
    context = RequestContext(text='测试', debug=True)
    context.append_row(row="""## 意图识别匹配详情""")

    table = [
        {'name': 'jack', 'age': 27},
        {'name': 'honey', 'age': 28},
        {'name': 'pony', 'age': 29, 'company': 'tencent'},

    ]
    context.append_table(table)

    context.export_report()
    return


if __name__ == '__main__':
    demo1()
