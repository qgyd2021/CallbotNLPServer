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


class TCallCentreCallInfo(Table):

    def __init__(self,
                 task_id: str,
                 mysql_connect: MySqlConnect = None,
                 product_id: str = 'callbot',
                 ):
        if not self._initialized:
            if task_id is None:
                logger.warning('Usually `task_id` is required to avoid whole table inquire.')
            self.task_id = task_id
            self.mysql_connect = mysql_connect
            self.product_id = product_id

            self._initialized = True

    @node_table_cache.memoize()
    def _init_data(self):
        if self.task_id is None:
            sql = """SELECT * FROM t_callcentre_call_info;"""
        else:
            sql = """SELECT * FROM t_callcentre_call_info WHERE product_id='{product_id}' AND task_id='{task_id}';""".format(
                product_id=self.product_id, task_id=self.task_id)

        data = self.mysql_connect.download_by_sql(
            sql=sql,
            headers=[
                'product_id',
                'task_id',
                'scene_id',
                'user_name',
                'client',
                'user_phone',
                'session_id',
                'session_time',
                'call_id',
                'call_status',
                'call_start',
                'call_end',
                'call_elasped',
                'call_line',
                'calling_phone',
                'call_audio_url',
                'user_intent_id',
                'op_ts',
                'turn_time',
                'create_ts',
                'version',
                'last_update_ts',
                'manual_call_start',
                'manual_call_answer',
                'manual_call_end',
                'manual_call_status',
                'manual_elapsed',
                'call_idx',
                'agent_asr_status',
                'agent_name',
                'agent_group',
                'callback_status',
                'callback_ts',
                'callback_time',
                'callback_url',
                'other',
                'call_answer',
                'params',
                'strategy_id',
                'action_id',
                'src_session_id',
                'src_session_time',
                'action_ts',
                'user_real_phone',
                'sms_status',
                'sms_content',
                'sms_send_ts',
                'sms_reach_ts',
                'sms_click_ts',
                'sms_click_count',
                'sms_sys_message_id',
                'sms_message_id',
                'sms_id',
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
        self.data.to_csv('t_callcentre_call_info.csv', index=False, encoding='utf_8_sig')
        return


def demo1():
    import argparse

    import os
    from project_settings import project_path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task_id',
        default='snukckd64z',
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

    t_callcentre_call_info = TCallCentreCallInfo(
        task_id=args.task_id,
        mysql_connect=MySqlConnect(
            host='10.20.251.13' if args.area == 'guangzhou' else '10.52.66.41',
            port=3306,
            user='callbot',
            password='NxcloudAI2021!',
            database=args.database,
            charset='utf8',
        )
    )

    t_callcentre_call_info.to_csv()
    return


if __name__ == '__main__':
    demo1()
