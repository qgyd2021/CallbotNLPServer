#!/usr/bin/python3
# -*- coding: utf-8 -*-
from aiohttp import BasicAuth
import asyncio
import json
import logging
from typing import List, Tuple

from elasticsearch import Elasticsearch

from toolbox.aiohttp import async_requests

logger = logging.getLogger("toolbox")


class AsyncElasticSearch(object):
    def __init__(self, hosts: List[str], port: int, http_auth: Tuple[str, str]):
        self.hosts = hosts
        self.port = port
        self.http_auth = http_auth
        self._http_auth = BasicAuth(
            login=self.http_auth[0],
            password=self.http_auth[1],
        )
        self.es = Elasticsearch(
            hosts=self.hosts,
            http_auth=self.http_auth,
            port=self.port
        )

        self.headers = {
            "Content-Type": "application/json"
        }

        self._host_index = 0

        self._search_url = 'http://{host}:{port}/{index}/_search?pretty'

    async def async_search(self, index, body, size=10):
        body['size'] = size
        host = self._next_host()
        url = self._search_url.format(host=host, port=self.port, index=index)
        data = json.dumps(body)
        # logger.debug('{}'.format(self.__class__.__name__))
        # logger.debug('url: {}'.format(url))
        # logger.debug('data: {}'.format(data))

        text, status_code = await async_requests.requests(
            'POST',
            url,
            headers=self.headers,
            data=data,
            auth=self._http_auth,
            timeout=2,
        )
        if status_code != 200:
            raise AssertionError('requests failed, status_code: {}, text: {}'.format(status_code, text))

        js = json.loads(text)
        return js

    def _next_host(self):
        if self._host_index >= len(self.hosts):
            self._host_index = 0

        host = self.hosts[self._host_index]
        self._host_index += 1
        return host

    def search(self, *args, **kwargs):
        return self.es.search(*args, **kwargs)

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
    es = AsyncElasticSearch(
        hosts=["10.20.251.8"],
        port=9200,
        http_auth=('elastic', 'ElasticAI2021!'),
    )

    index = 'nlpbot_callbot_ppe_default'

    # query = {
    #     'query': {
    #         'match_all': {}
    #     },
    #     'size': 2
    # }

    query = {
        "query": {
            "bool": {
                "must": [
                    # {
                    #     "match":
                    #         {
                    #             "text_preprocessed": "\u80fd \u4e0d\u80fd \u8bf4 \u6709\u7528 \u7684 \u3002"
                    #         }
                    # }
                ],
                "filter": [
                    {"term": {"product_id": "callbot"}},
                    {"term": {"scene_id": "x4r413sfaxf2"}},
                    # {"term": {"parent_node_id": "start_acwdg9pqjiozwqfgehtot"}}
                ]
            }
        },
        "size": 30
    }

    async def task():
        result = await es.async_search(index=index, size=10, body=query)
        print(result)
        return

    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        task()
    )

    return


def demo2():
    es = AsyncElasticSearch(
        hosts=["10.20.251.8"],
        port=9200,
        http_auth=('elastic', 'ElasticAI2021!'),
    )

    product_id = 'callbot'
    scene_id = 'k5q3l1hhtjmt'
    env = 'ppe'
    group_name = 'default'
    index = 'nlpbot_{}_{}_{}_{}'.format(product_id, scene_id, env, group_name)

    query = {
        'query': {
            'match_all': {}
        },
        'size': 10,
        'track_total_hits': True,

    }
    body = json.dumps(query)
    js = es.search(
        index=index, body=body, size=10
    )
    print(js)
    return


if __name__ == '__main__':
    # demo1()
    demo2()
