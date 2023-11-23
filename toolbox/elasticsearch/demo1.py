#!/usr/bin/python3
# -*- coding: utf-8 -*-
from elasticsearch import Elasticsearch, helpers


mapping = {
    'properties': {
        'text': {
            'type': 'text',
            'analyzer': 'whitespace',
            'search_analyzer': 'whitespace'
        },
        'text_preprocessed': {
            'type': 'text',
            'analyzer': 'whitespace',
            'search_analyzer': 'whitespace'
        },
        'text_id': {
            'type': 'keyword'
        },
        'product_id': {
            'type': 'keyword'
        },
        'scene_id': {
            'type': 'keyword'
        },
        'node_id': {
            'type': 'keyword'
        },
        'resource_id': {
            'type': 'keyword'
        },
    }
}


def demo1():
    """查看 ES 版本"""
    import requests
    from requests.auth import HTTPBasicAuth

    url = 'http://10.20.251.8:9200'

    resp = requests.get(url, auth=HTTPBasicAuth('elastic', 'ElasticAI2021!'), timeout=2)

    print(resp)
    print(resp.text)
    return


def demo2():
    """
    创建索引
    """
    index = 'callbot_x9o2opcr7n_ppe'
    es = Elasticsearch(["10.20.251.8"], http_auth=('elastic', 'ElasticAI2021!'), port=9200)

    if es.indices.exists(index=index):
        es.indices.delete(index=index)

    # 设置索引最大数量据.
    body = {
        "settings": {
            "index": {
                "max_result_window": 10000000000
            }
        }
    }
    result = es.indices.create(index=index, body=body)
    print(result)

    result = es.indices.put_mapping(
        index=index,
        doc_type='_doc',
        body=mapping,
        params={"include_type_name": "true"}
    )
    print(result)

    row = {
        'product_id': 'callbot',
        'scene_id': 'x9o2opcr7n',
        'node_id': 'start_acwdg9pqjiozwqfgehtot',
        'resource_id': 'resource_id',
        'text': '暂时还不了',
        'text_preprocessed': '暂时 还 不 了'

    }
    result = es.index(index=index, body=row, id=0)
    print(result)

    result = es.get(index=index, doc_type='_doc', id=0)
    print(result)

    es.indices.refresh(index=index)

    query = {
        'query': {
            'match_all': {}
        },
        'size': 2
    }
    result = es.search(
        index=index,
        size=10,
        body=query,
    )
    print(result)
    return


def demo3():
    """查询"""
    index = 'callbot_x9o2opcr7n_ppe'
    es = Elasticsearch(["10.20.251.8"], http_auth=('elastic', 'ElasticAI2021!'), port=9200)

    query = {
        'query': {
            'match': {
                'text_preprocessed': '暂时 不行'
            },
        }
    }
    result = es.search(index=index, size=10, body=query)
    print(result)

    query = {
        'query': {
            'bool': {
                'must': [{
                    'match': {
                        'text_preprocessed': '暂时 不行'
                    }
                }],
                'filter': [
                    {'term': {'product_id': 'callbot'}},
                    {'term': {'scene_id': 'x9o2opcr7n'}},
                    {'term': {'node_id': 'start_acwdg9pqjiozwqfgehtot'}},
                ]
            },
        }
    }
    result = es.search(index=index, size=10, body=query)
    print(result)

    return


def demo4():
    """查询 ES 中所有索引的情况"""
    import re
    import requests
    from requests.auth import HTTPBasicAuth

    url = 'http://10.20.251.8:9200/_cat/indices?v&pretty'

    resp = requests.get(url, auth=HTTPBasicAuth('elastic', 'ElasticAI2021!'), timeout=2)

    # print(resp)
    # print(resp.text)
    text = resp.text
    lines = text.split('\n')
    print(len(lines))
    for line in lines:
        # print(line)
        line = re.findall('\S+', line)
        if len(line) < 3:
            continue
        print(line[2])
    return


def demo5():
    """删掉 ES 中指定的索引"""
    import requests
    from requests.auth import HTTPBasicAuth

    index_list = [
        't9h1nt52um2_45b028c4-f7c3-43af-821e-081395ccb3c2_ppe_0',
        't9h1nt52um2_45b028c4-f7c3-43af-821e-081395ccb3c2_ppe_1',
        '5w5f2qy86d_21ff9e3e-5e20-4f71-9c64-bcf0226374a3_ppe_1',
        '5w5f2qy86d_21ff9e3e-5e20-4f71-9c64-bcf0226374a3_ppe_0',
        '19txd42ws0_end_dzrt8ql7rb_ppe_1',
        'uv86lt9z1q_1b563316-837d-4ecf-a233-ab8d7a84ed51_ppe_0',
        'uv86lt9z1q_1b563316-837d-4ecf-a233-ab8d7a84ed51_ppe_1',
        'lbfk90nfs7_hna46aswy4jwy9ji_ppe_0',
        'lb5fsw12dv_vlff489yikbc8din_ppe_0',
        'lupmormypo_yuyep3sazp_ppe_0',
        'lb5fsw12dv_vlff489yikbc8din_ppe_1',
        'lupmormypo_yuyep3sazp_ppe_1',
        'qnbtccxrlzh3_end_nvg92z7xb2_ppe_0',
        't9h1nt52um2_y528hloux8_ppe_0',
        't9h1nt52um2_y528hloux8_ppe_1',
        'lbfk90nfs7_hna46aswy4jwy9ji_ppe_1',
        'qnbtccxrlzh3_end_nvg92z7xb2_ppe_1',

    ]

    es = Elasticsearch(["10.20.251.8"], http_auth=('elastic', 'ElasticAI2021!'), port=9200)

    for index in index_list:
        if es.indices.exists(index=index):
            es.indices.delete(index=index)
            print(index)
    return


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
    # demo1()
    # demo2()
    # demo3()
    # demo4()
    demo5()
    # demo6()
