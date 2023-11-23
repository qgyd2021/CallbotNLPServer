#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
from typing import List, Tuple

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, "../../"))

from elasticsearch import Elasticsearch

from toolbox.design_patterns.singleton import ParamsSingleton
from toolbox.elasticsearch.query import AsyncElasticSearch
from nxtech.common.params import Params


class ElasticSearchConnect(Params, ParamsSingleton):
    def __init__(self,
                 hosts: List[str],
                 http_auth: Tuple[str, str],
                 port: int = 9200
                 ):
        if not self._initialized:
            super().__init__()
            self.hosts = hosts
            self.http_auth = http_auth
            self.port = port

            self.es = Elasticsearch(
                hosts=self.hosts,
                http_auth=self.http_auth,
                port=self.port
            )
            self.async_es = AsyncElasticSearch(
                hosts=self.hosts,
                http_auth=self.http_auth,
                port=self.port
            )
            self._initialized = True

    def search(self, *args, **kwargs):
        return self.es.search(*args, **kwargs)

    async def async_search(self, *args, **kwargs):
        return await self.async_es.async_search(*args, **kwargs)

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
        result.pop('es')
        return result

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

        es = Elasticsearch(
            hosts=self.hosts,
            http_auth=self.http_auth,
            port=self.port
        )
        setattr(self, 'es', es)
        return self


def demo1():
    query = {
        'query': {
            'match_all': {}
        },
        'size': 2
    }

    es = ElasticSearchConnect.from_json(
        params={
            "hosts": ["10.20.251.8"],
            "http_auth": ["elastic", "ElasticAI2021!"],
            "port": 9200
        },
    )
    result = es.search(
        index='callbot_x9o2opcr7n_ppe',
        size=10,
        body=query,
    )
    print(result)
    return


def demo2():
    """查 es 中有哪些索引"""
    query = {
        'query': {
            'match_all': {}
        },
        'size': 2
    }

    es = ElasticSearchConnect.from_json(
        params={
            "hosts": ["10.20.251.8"],
            "http_auth": ["elastic", "ElasticAI2021!"],
            "port": 9200
        },
    )
    result = es.search(
        index='callbot_x9o2opcr7n_ppe',
        size=10,
        body=query,
    )
    print(result)
    return


def demo3():
    """查 es 中有哪些索引
    curl -XGET 'http://172.16.1.190:9200'
    curl -XGET 'http://172.16.1.190:9200' --user admin:NXaibot@2023!#
    """
    query = {
        'query': {
            'match_all': {}
        },
        'size': 2
    }

    es = ElasticSearchConnect.from_json(
        params={
            "hosts": ["172.16.1.190"],
            "http_auth": ["admin", "NXaibot@2023!#"],
            "port": 9200
        },
    )
    result = es.search(
        index='callbot_x9o2opcr7n_ppe',
        size=10,
        body=query,
    )
    print(result)
    return


if __name__ == '__main__':
    # demo1()
    # demo2()
    demo3()
