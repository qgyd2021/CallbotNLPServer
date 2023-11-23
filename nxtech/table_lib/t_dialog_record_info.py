#!/usr/bin/python3
# -*- coding: utf-8 -*-
import logging
import time

from cacheout import Cache
import pandas as pd

from nxtech.table_lib.abstract_table import Table
from nxtech.database.mysql_connect import MySqlConnect

logger = logging.getLogger('nxtech')


record_table_cache = Cache(maxsize=256, ttl=1 * 60, timer=time.time)


class TDialogRecordInfo(Table):
    """
    会话记录.
    数据太多了, 按 product_id, scene_id 下载.
    """
    def __init__(self,
                 product_id: str,
                 scene_id: str,
                 start_time: str,
                 end_time: str,
                 mysql_connect: MySqlConnect = None,
                 ):
        """
        start_time = '2021-12-26'
        end_time = '2022-01-05'
        """
        if not self._initialized:
            if scene_id is None:
                logger.warning('Usually `scene_id` is required to avoid whole table inquire.')
            self.product_id = product_id
            self.scene_id = scene_id

            self.start_time = start_time
            self.end_time = end_time

            start_time_ts = time.strptime(start_time, '%Y-%m-%d')
            start_time_ts = time.mktime(start_time_ts)
            start_time_ts = int(start_time_ts)
            end_time_ts = time.strptime(end_time, '%Y-%m-%d')
            end_time_ts = time.mktime(end_time_ts)
            end_time_ts = int(end_time_ts)

            self.start_time_ts = start_time_ts
            self.end_time_ts = end_time_ts
            self.mysql_connect = mysql_connect
            self._initialized = True

    @record_table_cache.memoize()
    def _init_data(self):
        data = self.mysql_connect.download_by_sql(
            sql="""
SELECT * FROM t_dialog_record_info 
WHERE product_id='{}' AND scene_id='{}' AND create_ts>{} AND create_ts<{};
""".format(self.product_id, self.scene_id, self.start_time_ts, self.end_time_ts),
            headers=[
                'product_id',
                'user_id',
                'scene_id',
                'session_id',
                'record_id',
                'record_idx',
                'front',
                'record_type',
                'user_wording',
                'user_resource_url',
                'node_id',
                'node_type',
                'res_id',
                'res_wording',
                'res_url',
                'others',
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
        filename = 't_dialog_record_info_{}_{}_{}_{}.csv'.format(self.product_id, self.scene_id, self.start_time, self.end_time)
        self.data.to_csv(filename, index=False, encoding='utf_8_sig')
        return


def demo1():
    import os
    from project_settings import project_path

    t_dialog_record_info = TDialogRecordInfo(
        product_id='callbot',
        scene_id='uv86lt9z1q',
        mysql_connect=MySqlConnect(
            host='10.20.251.13',
            port=3306,
            user='callbot',
            password='NxcloudAI2021!',
            database='callbot_ppe',
            charset='utf8',
        ),
        start_time='2022-6-1',
        end_time='2022-6-16',
    )
    t_dialog_record_info.to_csv()
    return


if __name__ == '__main__':
    demo1()
