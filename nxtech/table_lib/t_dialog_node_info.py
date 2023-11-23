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


node_table_cache = Cache(maxsize=256, ttl=1 * 60, timer=time.time)


class TDialogNodeInfo(Table):
    """
    t_dialog_edge_info,
    t_dialog_node_info,
    t_dialog_resource_info

    用户回答时,
    我们根据: product_id, scene_id, node_id 从 t_dialog_node_info 中查出

    TDialogNodeInfo:
    1 //主流程节点
    2 //非主流程节点之用户要求重复
    3 //非主流程节点之用户无回应（控制类）
    4 //非主流程节点之开放问题
    5 //非主流程节点之无意义回应
    6 //非主流程节点之打断节点（控制类）（此类型节点在一个场景中有且只有一个）
    7 //结束节点（控制类）（此类型节点在一个场景中有且只有一个）
    8 //超过轮次时候结束的节点（此类型节点在一个场景中有且只有一个）
    9 //系统异常时候，强制结束（此类型节点在一个场景中有且只有一个）
    10 //全局挽回节点，用于匹配中忽略节点时候使用（此类型节点在一个场景中有且只有一个）
    11 //全局噪音节点 用于忽略该次输入
    12 //跳转人工节点 用于跳转人工客服
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

    @node_table_cache.memoize()
    def _init_data(self):
        if self.scene_id is None:
            sql = """SELECT * FROM t_dialog_node_info;"""
        else:
            sql = """SELECT * FROM t_dialog_node_info WHERE product_id='{product_id}' AND scene_id='{scene_id}';""".format(
                product_id=self.product_id, scene_id=self.scene_id)

        data = self.mysql_connect.download_by_sql(
            sql=sql,
            headers=[
                'product_id',
                'scene_id',
                'node_id',
                'node_flag',
                'show_type',
                'node_priority',
                'node_desc',
                'others',
                'node_type',
                'node_time',
                'intent_res_group_id',
                'intent_desc',
                'action_res_group_id',
                'action_again_res_group_id',
                'lead_res_group_id',
                'create_ts',
                'version',
                'last_update_ts',
                'hold_before_lead',
                'hold_before_lead_on',
                'jump_to_node_id',
                'func_on',
                'max_repeat_time',
                'sms_id',
                'enable_qa',
                'pause_delay',
            ],
        )

        if len(data) == 0:
            logger.warning('no data !')
            logger.warning('sql: {}'.format(sql))
        return data

    @property
    def data(self):
        return self._init_data()

    def to_csv(self):
        self.data.to_csv('t_dialog_node_info.csv', index=False, encoding='utf_8_sig')
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

        df = df[df['node_id'] == node_id]
        if len(df) == 0:
            raise AssertionError('node_id: {} invalid'.format(node_id))

        result = df.to_dict(orient='records')
        return result


def demo1():
    """
    python3 t_dialog_node_info.py --scene_id cdg26b89j98y --area hk
    python3 t_dialog_node_info.py --scene_id gn6cojmsba9f --area gz
    python3 t_dialog_node_info.py --scene_id 6fu2fazrahmf --area gz

    """
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

    t_dialog_node_info = TDialogNodeInfo(
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

    t_dialog_node_info.to_csv()
    return


if __name__ == '__main__':
    demo1()
