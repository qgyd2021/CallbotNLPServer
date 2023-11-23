#!/usr/bin/python3
# -*- coding: utf-8 -*-
from urllib.parse import urlparse
import json


def request_to_curl(method, url, headers=None, data=None):
    parse_result = urlparse(url)

    curl = """curl {scheme}://{netloc}/{path} -X POST -H "Content-Type:application/json" -d '{data}' """.format(
        scheme=parse_result.scheme,
        netloc=parse_result.netloc,
        path=parse_result.path,
        data=json.dumps(data),
    )
    return curl


def demo1():
    url = 'http://127.0.0.1:9072/chatbot/nlp'

    headers = {
        'Content-Type': 'application/json'
    }

    data = {
        "productID": "callbot",
        "userID": "curl_access_test",
        'sceneID': 'piaka1k48m',
        'currNodeID': 'd9c1e43a-aa6c-43e8-88b8-2706afedc53f',
        "userInput": "公司什么位置",

        "callID": "curl_access_test",
        "createTs": 1597480710,
        "sign": "curl_access_test",
    }

    curl_cmd = request_to_curl(
        method='POST',
        url=url,
    )
    print(curl_cmd)
    return


if __name__ == '__main__':
    demo1()
