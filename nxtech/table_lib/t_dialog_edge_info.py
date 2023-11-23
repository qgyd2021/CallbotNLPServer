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


edge_table_cache = Cache(maxsize=256, ttl=1 * 60, timer=time.time)


class TDialogEdgeInfo(Table):
    """
    t_dialog_edge_info,
    t_dialog_node_info,
    t_dialog_resource_info

    用户回答时,
    我们根据: product_id, scene_id, node_id 从 t_dialog_node_info 中查出
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

    @edge_table_cache.memoize()
    def _init_data(self):
        if self.scene_id is None:
            sql = """SELECT * FROM t_dialog_edge_info;"""
        else:
            sql = """SELECT * FROM t_dialog_edge_info WHERE product_id='{product_id}' AND scene_id='{scene_id}';""".format(
                product_id=self.product_id, scene_id=self.scene_id)

        data = self.mysql_connect.download_by_sql(
            sql=sql,
            headers=[
                'product_id',
                'scene_id',
                'edge_id',
                'src_node_id',
                'dst_node_id',
                'edge_desc',
                'create_ts',
                'version',
                'last_update_ts',
                'edge_type',
            ],
        )
        return data

    @property
    def data(self):
        return self._init_data()

    def to_csv(self):
        self.data.to_csv('t_dialog_edge_info.csv', index=False, encoding='utf_8_sig')
        return

    def get_rows_by_node_id(self, product_id: str, scene_id: str, node_id: str) -> List[dict]:
        df = self.data

        if len(df) == 0:
            return list()

        df = df[df['product_id'] == product_id]
        if len(df) == 0:
            raise AssertionError('product_id: {} invalid'.format(product_id))

        df = df[df['scene_id'] == scene_id]
        if len(df) == 0:
            raise AssertionError('scene_id: {} invalid'.format(scene_id))

        df = df[df['src_node_id'] == node_id]
        if len(df) == 0:
            return list()

        result = df.to_dict(orient='records')
        return result


def demo1():
    """python3 t_dialog_edge_info.py --scene_id ad6e2oq406 --area hk"""
    import argparse

    import os
    from project_settings import project_path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scene_id',
        # default='6fu2fazrahmf',
        # default='ad6e2oq406',
        # default='rib2j7lrg4',
        default='7d3i1xk7eg81',
        type=str,
    )
    parser.add_argument(
        '--area',
        # default='gz',
        default='hk',
        type=str,
    )
    parser.add_argument(
        '--database',
        default='callbot_ppe',
        type=str,
    )
    args = parser.parse_args()

    if args.area == 'gz':
        host = '10.20.251.13'
        password = 'wm%msjngbtmheh3TdqYbmgg3s@nxprd230417'
    elif args.area == 'hk':
        host = '10.52.66.41'
        password = 'SdruuKtzmjexpq%dj6mu9qryk@nxprd230413'
    elif args.area == 'mx':
        host = '172.16.1.149'
        password = 'Vstr2ajjlYeduvf7bu%@nxprd230417'
    else:
        raise AssertionError

    t_dialog_edge_info = TDialogEdgeInfo(
        scene_id=args.scene_id,
        mysql_connect=MySqlConnect(
            host=host,
            port=3306,
            user='nx_prd',
            password=password,
            # user='callbot',
            # password='NxcloudAI2021!',
            database=args.database,
            charset='utf8',
        )

    )
    t_dialog_edge_info.to_csv()
    return


if __name__ == '__main__':
    demo1()
