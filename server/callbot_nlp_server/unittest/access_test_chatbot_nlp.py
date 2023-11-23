#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import time

import requests


def get_args():
    """
    python3 access_test_chatbot_nlp.py --port 9070 --scene_id aha6p0n4g9a8 --current_node_id 26032964-e631-4a0b-9e97-1ce03735d418

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=9072, type=int)
    parser.add_argument("--scene_id", default="mjjuzh32uqlu", type=str)
    parser.add_argument("--current_node_id", default="c826e3a9-c6c2-4dd1-a056-977a129caed5", type=str)
    parser.add_argument("--user_input", default="你们在什么位置", type=str)
    parser.add_argument("--debug", default=False, action="store_true")

    args = parser.parse_args()
    return args


def main():
    import sys
    print(sys.platform)

    args = get_args()

    url = "http://{host}:{port}/chatbot/nlp".format(
        host=args.host,
        port=args.port,
    )

    data = {
        "productID": "callbot",
        "userID": "access_test_user_id",
        "sceneID": args.scene_id,
        "currNodeID": args.current_node_id,
        "userInput": args.user_input,
        "callID": "access_test_call_id",
        "createTs": int(time.time()),
        "sign": "access_test_sign",
        "debug": args.debug,
    }

    headers = {
        "Content-Type": "application/json"
    }

    begin = time.time()
    resp = requests.post(url, headers=headers, data=json.dumps(data))
    cost = time.time() - begin
    print(cost)
    if resp.status_code != 200:
        print(resp.text)
    else:
        if sys.platform in ("win32",):
            js = resp.json()
            result = json.dumps(js, ensure_ascii=False, indent=4)
            print(result)
        else:
            print(resp.text)
    return


if __name__ == "__main__":
    main()
