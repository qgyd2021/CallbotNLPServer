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


class TKnowWordNodeInfo(Table):
    """

    """
    def __init__(self,
                 scene_id: str,
                 mysql_connect: MySqlConnect = None,
                 product_id: str = 'callbot',
                 ):
        if not self._initialized:
            if scene_id is None:
                logger.warning('Usually `scene_id` is required to avoid whole table inquire.')
            self.scene_id = scene_id
            self.mysql_connect = mysql_connect
            self.product_id = product_id

            self._initialized = True

    @know_word_table_cache.memoize()
    def _init_data(self):
        if self.scene_id is None:
            sql = """SELECT * FROM t_know_word_node_info WHERE product_id='{product_id}';"""
        else:
            sql = """SELECT * FROM t_know_word_node_info WHERE product_id='{product_id}' AND scene_id='{scene_id}';""".format(
                product_id=self.product_id, scene_id=self.scene_id)

        # primary: scene_id, node_id, basic_id, word_id
        data = self.mysql_connect.download_by_sql(
            sql=sql,
            headers=[
                'product_id',
                'scene_id',
                'node_id',
                'basic_id',
                'word_id',
                'remark',
                'create_ts',
                'version',
                'last_update_ts',
            ],
        )
        return data

    @property
    def data(self):
        return self._init_data()

    def to_csv(self):
        self.data.to_csv('t_know_word_node_info.csv', index=False, encoding='utf_8_sig')
        return

    def get_rows_by_node_id(self, product_id: str, scene_id: str, node_id: str) -> List[dict]:
        df = self.data
        if len(df) == 0:
            return list()

        df = df[df['product_id'] == product_id]
        if len(df) == 0:
            return list()

        df = df[df['scene_id'] == scene_id]
        if len(df) == 0:
            return list()

        df = df[df['node_id'] == node_id]
        if len(df) == 0:
            return list()

        result = df.to_dict(orient='records')
        return result


def demo1():
    """python3 t_know_word_node_info.py --scene_id nt0byr0wso1q --area hongkong"""
    import argparse

    import os
    from project_settings import project_path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scene_id',
        default='6fu2fazrahmf',
        type=str,
    )
    parser.add_argument(
        '--area',
        default='guangzhou',
        type=str,
    )
    args = parser.parse_args()

    table = TKnowWordNodeInfo(
        scene_id=args.scene_id,
        mysql_connect=MySqlConnect(
            host='10.20.251.13' if args.area == 'guangzhou' else '10.52.66.41',
            port=3306,
            user='callbot',
            password='NxcloudAI2021!',
            database='callbot_ppe',
            charset='utf8',
        )
    )
    table.to_csv()
    return


if __name__ == '__main__':
    demo1()
