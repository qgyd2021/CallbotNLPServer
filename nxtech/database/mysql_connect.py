#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
from typing import List

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, "../../"))

import pymysql
import pandas as pd

from nxtech.common.params import Params
from toolbox.design_patterns.singleton import ParamsSingleton


class MySqlConnect(Params, ParamsSingleton):
    def __init__(self,
                 host: str = "10.20.251.13",
                 port: int = 3306,
                 user: str = "callbot",
                 password: str = "password",
                 database: str = "callbot_dev",
                 charset: str = "utf8"
                 ):
        if not self._initialized:
            self.host = host
            self.port = port
            self.user = user
            self.password = password
            self.database = database
            self.charset = charset
            print("host: {}".format(self.host))
            print("port: {}".format(self.port))
            print("user: {}".format(self.user))
            print("password: {}".format(self.password))

            self.connect = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset=self.charset,
                autocommit=False,
            )

            self._initialized = True

    def __getstate__(self):
        """
        pymysql.connect 对象不能被 pickle 序列化.

        __getstate__ 和 __setstate__ 方法是为了解决这个问题.
        https://docs.python.org/zh-cn/3.7/library/pickle.html#object.__getstate__

        默认情况, pickle 将类实例的 __dict__ 打包 (称作 state).
        在恢复对象时, 先创建一个对象实例, 再将 __dict__ 打包的 state 中的 k, v 赋值为实例属性.

        当定义了 __getstate__ 方法后, pickle 则按此方法打包实例.
        当定义了 __setstate__ 方法后, pickle 在恢复对象时, 调用此方法.
        """
        result = {
            **self.__dict__
        }
        result.pop("connect")
        return result

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

        connect = pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            charset=self.charset,
            autocommit=False,
        )
        setattr(self, "connect", connect)
        return self

    def execute(self, sql: str, commit: bool = False):
        self.connect.ping(reconnect=True)
        cursor = self.connect.cursor()
        cursor.execute(sql)
        if commit:
            self.connect.commit()
        all_rows = cursor.fetchall()
        all_rows = [row for row in all_rows]
        result = list()
        for row in all_rows:
            decoded_row = list()
            for field in row:
                if isinstance(field, bytes):
                    field = field.decode("utf-8")
                decoded_row.append(field)
            result.append(decoded_row)
        return result

    def download_by_sql(self, sql: str, headers: List[str]):
        all_rows = self.execute(sql)

        result = list()
        for row in all_rows:
            result.append(dict(zip(headers, row)))

        result = pd.DataFrame(result)
        return result


def demo1():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene_id",
        default="ad6e2oq406",
        type=str,
    )
    parser.add_argument(
        "--src_node_id",
        default="nlotjukusp",
        type=str,
    )
    parser.add_argument(
        "--area",
        default="hk",
        type=str,
    )
    parser.add_argument(
        "--database",
        default="callbot_ppe",
        type=str,
    )
    args = parser.parse_args()

    if args.area == "gz":
        host = "10.20.251.13"
        password = "wm%msjngbtmheh3TdqYbmgg3s@nxprd230417"
    elif args.area == "hk":
        host = "10.52.66.41"
        password = "SdruuKtzmjexpq%dj6mu9qryk@nxprd230413"
    elif args.area == "mx":
        host = "172.16.1.149"
        password = "Vstr2ajjlYeduvf7bu%@nxprd230417"
    else:
        raise AssertionError

    mysql_connect = MySqlConnect(
        host=host,
        port=3306,
        user="nx_prd",
        password=password,
        database=args.database,
        charset="utf8",
    )

    sql = """
DELETE FROM `callbot_ppe`.`t_dialog_edge_info`
WHERE `scene_id`="{scene_id}" AND `src_node_id`="{src_node_id}";
    """.format(scene_id=args.scene_id, src_node_id=args.src_node_id)
    mysql_connect.execute(sql)
    return


if __name__ == "__main__":
    demo1()
