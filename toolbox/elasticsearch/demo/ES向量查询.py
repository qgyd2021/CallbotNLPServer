#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
ES 向量相似度计算.
参考链接:
https://blog.csdn.net/qq_16164711/article/details/120140415

cosineSimilarity – 余弦函数
dotProduct – 向量点积
l1norm – 曼哈顿距离
l2norm - 欧几里得距离
"""
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
        'text_vector': {
            'type': 'dense_vector',
            'dims': 3
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
    index = 'nlpbot_test_index'
    es = Elasticsearch(["10.20.251.8"], http_auth=('elastic', 'ElasticAI2021!'), port=9200)

    if es.indices.exists(index=index):
        es.indices.delete(index=index)

    # 设置索引最大数量据.
    body = {
        "settings": {
            "index": {
                "max_result_window": 100000000
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
        'text_preprocessed': '暂时 还 不 了',
        'text_vector': [1, 2, 3]
    }
    # 单个写入数据
    # result = es.index(index=index, body=row, id=0)
    # print(result)
    # result = es.get(index=index, doc_type='_doc', id=0)
    # print(result)

    # 批量写入数据
    rows = [
        {
            '_op_type': 'index',
            '_index': index,
            # '_id': idx,
            # '_type': 'last',
            '_source': row
        }
    ]
    helpers.bulk(client=es, actions=rows)

    es.indices.refresh(index=index)

    return


def demo3():
    """查询"""
    index = 'nlpbot_test_index'
    es = Elasticsearch(["10.20.251.8"], http_auth=('elastic', 'ElasticAI2021!'), port=9200)

    query = {
        'query': {
            'script_score': {
                'query': {
                    'match_all': {},
                },
                'script': {
                    # 'source': "cosineSimilarity(params.query_vector, 'text_vector') + 1.0",
                    'source': "l2norm(params.query_vector, 'text_vector')",

                    'params': {
                        'query_vector': [
                            1,
                            2,
                            3
                        ]
                    }
                },
            },
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


def demo4():
    """
    查询

    好像, 按 script 算分之后, 就不能再结合其它字段算分, 有点鸡肋.
    query 过滤后的结果, 全部要参与向量计算, 运算量过大, 比 faiss 效率低太多.
    """
    index = 'nlpbot_test_index'
    es = Elasticsearch(["10.20.251.8"], http_auth=('elastic', 'ElasticAI2021!'), port=9200)

    query = {
        'query': {
            'script_score': {
                'query': {
                    'bool': {
                        'must': [{
                            'match': {
                                'text_preprocessed': '还 不 了'
                            }
                        }],
                        'filter': [
                            {'term': {'product_id': 'callbot'}},
                            {'term': {'scene_id': 'x9o2opcr7n'}},
                        ]
                    },
                },
                'script': {
                    # 'source': "cosineSimilarity(params.query_vector, 'text_vector') + 1.0",
                    'source': "l2norm(params.query_vector, 'text_vector')",

                    'params': {
                        'query_vector': [
                            1,
                            2,
                            3
                        ]
                    }
                },
            },
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


if __name__ == '__main__':
    # demo1()
    # demo2()
    # demo3()
    demo4()
