#!/usr/bin/python3
# -*- coding: utf-8 -*-
import hashlib
import json


def hash_json(js):
    js_string = json.dumps(js)
    result = hashlib.md5(js_string.encode()).hexdigest()
    return result


def demo1():
    js = {
        '你好': 'hello'
    }
    result = hash_json(js)
    print(result)
    return


if __name__ == '__main__':
    demo1()
