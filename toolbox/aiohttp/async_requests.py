#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import time
import asyncio
from aiohttp import ClientSession, BasicAuth, hdrs
from aiohttp.client import _RequestContextManager


async def get(url, headers=None, data=None,
              auth: BasicAuth = None, timeout=2
              ):
    async with ClientSession(auth=auth) as session:
        async with session.get(url, headers=headers, data=data, timeout=timeout) as response:
            status_code = response.status
            text = await response.read()
            text = text.decode(encoding='utf-8')
    return text, status_code


async def post(url, headers=None, data=None,
               auth: BasicAuth = None, timeout=2
               ):
    async with ClientSession(auth=auth) as session:
        async with session.post(url, headers=headers, data=data, timeout=timeout) as response:
            status_code = response.status
            text = await response.read()
            text = text.decode(encoding='utf-8')
    return text, status_code


async def requests(method, url, headers=None, data=None,
                   auth: BasicAuth = None, timeout=2
                   ):
    """
    :param method: hdrs.METH_POST. `GET`, `POST`
    :param url:
    :param headers:
    :param data:
    :param auth:
    :param timeout:
    :return:
    """
    async with ClientSession(auth=auth) as session:
        async with _RequestContextManager(
            session._request(method, url, data=data, headers=headers, timeout=timeout)
        ) as response:
            status_code = response.status
            text = await response.read()
            text = text.decode(encoding='utf-8')
    return text, status_code


def demo1():
    # 定义异步函数
    async def hello():
        await asyncio.sleep(1)
        print('Hello World:%s' % time.time())

    loop = asyncio.get_event_loop()
    tasks = [hello() for i in range(5)]
    loop.run_until_complete(asyncio.wait(tasks))

    return


def demo2():
    url = 'http://10.20.251.5:1080/HeartBeat'

    async def task():
        text, status_code = await get(url, timeout=2)
        if status_code == 200:
            js = json.loads(text)
            print(js)
        else:
            print(text)
        return

    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        task()
    )
    return


def demo3():
    index = 'nlpbot_callbot_ppe_default'

    url = 'http://10.20.251.8:9200/{index}/_search?pretty'.format(index=index)

    headers = {
        "Content-Type": "application/json"
    }
    query = {
        'query': {
            'match_all': {}
        },
        'size': 2
    }

    auth = BasicAuth(login='elastic', password='ElasticAI2021!')

    async def task():

        text, status_code = await requests(
            'POST',
            url,
            headers=headers,
            data=json.dumps(query),
            auth=auth,
            timeout=2,
        )
        if status_code == 200:
            js = json.loads(text)
            print(js)
        else:
            print(text)
        return

    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        task()
    )
    return


def demo4():
    url = 'http://10.52.66.97:13070/BasicIntent'

    headers = {
        "Content-Type": "application/json"
    }
    data = {'key': 'chinese', 'text': '你好, 你好吗? 嗯. '}

    async def task():
        text, status_code = await requests(
            'POST',
            url,
            headers=headers,
            data=json.dumps(data, ensure_ascii=False),
            timeout=2,
        )
        if status_code == 200:
            # js = json.loads(text)
            # print(js)
            print(text)
        else:
            print(text)
        return

    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        task()
    )
    return


if __name__ == '__main__':
    # demo1()
    # demo2()
    # demo3()
    demo4()
