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

resource_table_cache = Cache(maxsize=256, ttl=1 * 60, timer=time.time)


class TDialogResourceInfo(Table):
    """
    t_dialog_edge_info,
    t_dialog_node_info,
    t_dialog_resource_info

    用户回答时,
    我们根据: product_id, scene_id, node_id 从 t_dialog_node_info 中查出

    TDialogResourceInfo
    res_type
    1 //意图词
    2 //机器话术
    3 //白名单正则
    4 //黑名单正则
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

    @resource_table_cache.memoize()
    def _init_data(self):
        if self.scene_id is None:
            sql = """SELECT * FROM t_dialog_resource_info;"""
        else:
            sql = """SELECT * FROM t_dialog_resource_info WHERE product_id='{product_id}' AND scene_id='{scene_id}';""".format(
                product_id=self.product_id, scene_id=self.scene_id)

        data = self.mysql_connect.download_by_sql(
            sql=sql,
            headers=[
                'product_id',
                'scene_id',
                'group_id',
                'res_id',
                'res_type',
                'res_desc',
                'word_id',
                'word',
                'res_url',
                'create_ts',
                'version',
                'last_update_ts',
                'res_url_id',
                'prever_word',
                'res_idx',
                'func_type',

            ],
        )
        return data

    @property
    def data(self):
        return self._init_data()

    def to_csv(self):
        self.data.to_csv('t_dialog_resource_info.csv', index=False, encoding='utf_8_sig')
        return

    def _init_data_from_file(self, filename):
        """上测试时加载本地 csv 文件初始化"""
        data = pd.read_csv(filename)
        return data

    def get_rows_by_group_id(self, product_id: str, scene_id: str, group_id: str) -> List[dict]:
        df = self.data

        if len(df) == 0:
            return list()

        df = df[df['product_id'] == product_id]
        if len(df) == 0:
            return list()

        df = df[df['scene_id'] == scene_id]
        if len(df) == 0:
            return list()

        df = df[df['group_id'] == group_id]
        if len(df) == 0:
            return list()

        result = df.to_dict(orient='records')
        return result


def demo1():
    """python3 t_dialog_resource_info.py --scene_id 6gxqrfz7cxop --area guangzhou"""
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

    resource = TDialogResourceInfo(
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
    resource.to_csv()
    return


if __name__ == '__main__':
    demo1()
