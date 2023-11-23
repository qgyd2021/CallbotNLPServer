#!/usr/bin/python3
# -*- coding: utf-8 -*-
import logging
import os
import sys
import time
from typing import List

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))

from cacheout import Cache
import pandas as pd

from nxtech.table_lib.abstract_table import Table
from nxtech.database.mysql_connect import MySqlConnect

logger = logging.getLogger('nxtech')

know_word_table_cache = Cache(maxsize=256, ttl=1 * 60, timer=time.time)


class TKnowWordInfo(Table):
    def __init__(self,
                 basic_id: str,
                 mysql_connect: MySqlConnect = None,
                 product_id: str = 'callbot',
                 ):
        if not self._initialized:
            if basic_id is None:
                logger.warning('Usually `scene_id` is required to avoid whole table inquire.')
            self.basic_id = basic_id
            self.mysql_connect = mysql_connect
            self.product_id = product_id

            self._initialized = True

    @know_word_table_cache.memoize()
    def _init_data(self):
        if self.basic_id is None:
            sql = """SELECT * FROM t_know_word_info WHERE product_id='{product_id}';"""
        else:
            sql = """SELECT * FROM t_know_word_info WHERE product_id='{product_id}' AND basic_id='{basic_id}';""".format(
                product_id=self.product_id, basic_id=self.basic_id)

        # primary: product_id, basic_id, catalog_id, word_id
        data = self.mysql_connect.download_by_sql(
            sql=sql,
            headers=[
                'product_id',
                'basic_id',
                'catalog_id',
                'word_id',
                'word_name',
                'know_type',
                'remark',
                'intent_res_group_id',
                'action_res_group_id',
                'regular_res_group_id',
                'update_ts',
                'create_ts',
                'version',
                'last_update_ts',
                'key_words',
                'bot_words',
            ],
        )
        return data

    @property
    def data(self):
        return self._init_data()

    def get_rows_by_word_id(self, product_id: str, basic_id: str, word_id: str) -> List[dict]:
        df = self.data
        if len(df) == 0:
            return list()

        df = df[df['product_id'] == product_id]
        if len(df) == 0:
            return list()

        df = df[df['basic_id'] == basic_id]
        if len(df) == 0:
            return list()

        df = df[df['word_id'] == word_id]
        if len(df) == 0:
            return list()

        result = df.to_dict(orient='records')
        return result


if __name__ == '__main__':
    pass
