#!/usr/bin/python3
# -*- coding: utf-8 -*-


def demo1():
    """设置 es 中可存在的索引总数量"""
    import json
    import requests
    from requests.auth import HTTPBasicAuth

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
    resp = requests.get(
        url,
        headers=headers,
        data=json.dumps(query),
        auth=HTTPBasicAuth('elastic', 'ElasticAI2021!'),
        timeout=2
    )
    print(resp)
    print(resp.text)
    return


if __name__ == '__main__':
    demo1()
