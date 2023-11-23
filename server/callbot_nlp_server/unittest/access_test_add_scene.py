#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import time

import requests


def get_args():
    """
    python3 access_test_add_scene_id.py --host_port 127.0.0.1:9072 --scene_id cdg26b89j98y --language cn --group cn

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=9072, type=int)
    parser.add_argument("--scene_id", default="cdg26b89j98y", type=str)
    parser.add_argument("--language", default="cn", type=str)
    parser.add_argument("--group", default="cn", type=str)

    args = parser.parse_args()
    return args


def main():
    import sys
    print(sys.platform)

    args = get_args()

    url = "http://{host}:{port}/chatbot/nlp/add_scene".format(
        host=args.host,
        port=args.port,
    )

    data = {
        "product_id": "callbot",

        "language": args.language,
        "scene_id": args.scene_id,
        "group": args.group,
        "env": "ppe",
    }

    headers = {
        "Content-Type": "application/json"
    }

    resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=None)
    print(resp.text)
    return


if __name__ == "__main__":
    main()
