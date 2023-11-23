#!/usr/bin/python3
# -*- coding: utf-8 -*-
import logging
import os
import sys
import time

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))

from cacheout import Cache
import pandas as pd

from nxtech.table_lib.abstract_table import Table
from nxtech.database.mysql_connect import MySqlConnect

logger = logging.getLogger('nxtech')


node_table_cache = Cache(maxsize=256, ttl=1 * 60, timer=time.time)


class TCallCentreTaskInfo(Table):
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
            sql = """SELECT * FROM t_callcentre_task_info;"""
        else:
            sql = """SELECT * FROM t_callcentre_task_info WHERE product_id='{product_id}' AND scene_id='{scene_id}';""".format(
                product_id=self.product_id, scene_id=self.scene_id)

        data = self.mysql_connect.download_by_sql(
            sql=sql,
            headers=[
                'product_id',
                'scene_id',
                'task_id',
                'create_by',
                'task_status',
                'task_priority',
                'task_name',
                'task_desc',
                'startup_type',
                'startup_by',
                'startup_at',
                'zone_second',
                'week_day',
                'up_hour',
                'down_hour',
                'line_id',
                'max_call',
                'redial_type',
                'robot_num',
                'create_ts',
                'version',
                'last_update_ts',
                'other',
                'import_type',
                'func_flag',
                'sms_info',
                'strategy_id',
                'callback_status',
                'callback_ts',
                'callback_time',
                'callback_url',
                'user_task_id',
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
        self.data.to_csv('t_callcentre_task_info.csv', index=False, encoding='utf_8_sig')
        return


def demo1():
    import argparse

    import os
    from project_settings import project_path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scene_id',
        default='9qtb38k09cpb',
        type=str,
    )
    parser.add_argument(
        '--area',
        # default='guangzhou',
        default='hongkong',
        type=str,
    )
    parser.add_argument(
        '--database',
        default='callbot_ppe',
        type=str,
    )

    args = parser.parse_args()

    t_callcentre_task_info = TCallCentreTaskInfo(
        scene_id=args.scene_id,
        mysql_connect=MySqlConnect(
            host='10.20.251.13' if args.area == 'guangzhou' else '10.52.66.41',
            port=3306,
            user='callbot',
            password='NxcloudAI2021!',
            database=args.database,
            charset='utf8',
        )
    )

    t_callcentre_task_info.to_csv()
    return


if __name__ == '__main__':
    demo1()
