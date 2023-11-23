#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json


def demo1():
    with open('cn_sentence_patterns.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    for s in data:
        print(s['zh_title'])
    return


if __name__ == '__main__':
    demo1()
