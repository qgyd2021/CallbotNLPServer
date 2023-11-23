#!/usr/bin/python3
# -*- coding: utf-8 -*-


def demo6():
    """设置 es 中可存在的索引总数量"""
    import json
    import requests
    from requests.auth import HTTPBasicAuth

    url = 'http://10.20.251.8:9200/_cluster/settings'

    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "transient": {
            "cluster": {
                "max_shards_per_node": 10000
            }
        }
    }
    resp = requests.put(
        url,
        headers=headers,
        data=json.dumps(data),
        auth=HTTPBasicAuth('elastic', 'ElasticAI2021!'),
        timeout=2
    )
    print(resp)
    print(resp.text)
    return



if __name__ == '__main__':
    pass
