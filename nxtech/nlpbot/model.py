import importlib
import json
import logging
import numpy as np
import os
import pickle
from typing import Dict, List

from cacheout import Cache, LIFOCache
import requests

from nxtech.common.params import Params
from toolbox.aiohttp import async_requests
from toolbox.design_patterns.singleton import ParamsSingleton

logger = logging.getLogger("nxtech")


class HttpAllenNlpPredictor(Params, ParamsSingleton):
    _initialized = False

    def __init__(self,
                 host_port: str,
                 ):
        if not self._initialized:
            super().__init__()
            self.host_port = host_port
            self._headers = {
                "Content-Type": "application/json"
            }

            self._predict_json_url = "{}/predict_json".format(host_port)
            self._predict_batch_json_url = "{}/predict_batch_json".format(host_port)

            self._cache = Cache(maxsize=20480)

            self._initialized = True

    def __repr__(self):
        return "<{}, {}>".format(self.__class__.__name__, self.__getstate__())

    def __getstate__(self):
        result = {
            **self.__dict__
        }
        result.pop("_cache")

        return result

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

        setattr(self, "_cache", Cache(maxsize=10240))
        return self

    def predict_json(self, inputs: dict):
        key = json.dumps(inputs)
        result = self._cache.get(key)
        if result is not None:
            return result

        data = {
            "inputs": inputs
        }
        resp = requests.post(self._predict_json_url, headers=self._headers, data=json.dumps(data))
        if resp.status_code != 200:
            raise AssertionError("requests failed, url: {}, status_code: {}, text: {}".format(self._predict_json_url, resp.status_code, resp.text))

        js = resp.json()
        if js["status_code"] != 60200:
            raise AssertionError("requests failed, url: {}, status_code: {}, text: {}".format(self._predict_json_url, js["status_code"], resp.text))

        result = js["result"]
        self._cache.set(key, result)
        return result

    async def async_predict_json(self, inputs: dict):
        key = json.dumps(inputs)
        result = self._cache.get(key)
        if result is not None:
            return result

        data = {
            "inputs": inputs
        }
        text, status_code = await async_requests.requests(
            method="POST", url=self._predict_json_url,
            headers=self._headers, data=json.dumps(data)
        )
        if status_code != 200:
            raise AssertionError("requests failed, status_code: {}, text: {}".format(status_code, text))
        js = json.loads(text)

        result = js["result"]
        self._cache.set(key, result)
        return result

    def predict_batch_json(self, inputs: List[dict]):
        data = {
            "inputs": inputs
        }
        resp = requests.post(self._predict_batch_json_url, headers=self._headers, data=json.dumps(data))
        if resp.status_code != 200:
            raise AssertionError("requests failed, url: {}, status_code: {}, text: {}".format(self._predict_json_url, resp.status_code, resp.text))

        js = resp.json()
        if js["status_code"] != 60200:
            raise AssertionError("requests failed, url: {}, status_code: {}, text: {}".format(self._predict_json_url, js["status_code"], resp.text))

        result = js["result"]
        return result

    async def async_predict_batch_json(self, inputs: List[dict]):
        data = {
            "inputs": inputs
        }
        text, status_code = await async_requests.requests(
            method="POST", url=self._predict_batch_json_url,
            headers=self._headers, data=json.dumps(data)
        )
        if status_code != 200:
            raise AssertionError("requests failed, status_code: {}, text: {}".format(status_code, text))
        js = json.loads(text)

        result = js["result"]
        return result


class TextClassifierPredictor(Params, ParamsSingleton):
    def __init__(self):
        super().__init__()

    def predict(self, text: str) -> dict:
        raise NotImplementedError

    async def async_predict(self, text: str) -> dict:
        raise NotImplementedError


@TextClassifierPredictor.register("basic_intent_classifier")
class BasicIntentClassifierPredictor(TextClassifierPredictor):
    """
    basic intent

    The most basic intent recognize label system.
    Labels are defined according to experience.
    Labels are divided into domain-related and domain-independent.
    Domain-related internals are further divided into various intents.
    Each language has about one hundred intents.

    language: chinese, english, japanese, vietnamese.

    """
    def __init__(self,
                 url: str,
                 language: str
                 ):
        if not self._initialized:
            super().__init__()
            self.url = url
            self.language = str(language).strip().lower()

            self._headers = {
                "Content-Type": "application/json"
            }

            self._url = url

            self._maxsize = 20480
            self._cache = Cache(maxsize=self._maxsize)

            self._initialized = True

    def __repr__(self):
        return "<{}, {}>".format(self.__class__.__name__, self.__getstate__())

    def __getstate__(self):
        result = {
            **self.__dict__
        }
        result.pop("_cache")

        return result

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

        setattr(self, "_cache", Cache(maxsize=self._maxsize))
        return self

    def predict(self, text: str) -> dict:
        json_dict = {"key": self.language, "text": text}
        key = json.dumps(json_dict)
        result = self._cache.get(key)
        if result is not None:
            return result

        resp = requests.post(
            self._url,
            headers=self._headers,
            # ensure_ascii=True 时, C++服务收到的是字符串 `\u4f60\u597d`.
            # 虽然 python 会自动转换, 但我的 C++ 服务并没有. 因此, 需要确保 C++服务收到的是正确的字符串.
            # data=json.dumps(json_dict, ensure_ascii=False).encode("utf-8")
            data=json.dumps(json_dict, ensure_ascii=False)
        )
        if resp.status_code != 200:
            raise AssertionError("requests failed, url: {}, status_code: {}, text: {}".format(
                self._url, resp.status_code, resp.text))

        js = resp.json()

        result = js
        self._cache.set(key, result)
        return result

    async def async_predict(self, text: str) -> dict:
        json_dict = {"key": self.language, "text": text}
        key = json.dumps(json_dict)
        result = self._cache.get(key)
        if result is not None:
            return result

        text, status_code = await async_requests.requests(
            method="POST", url=self._url,
            headers=self._headers,
            # ensure_ascii=True 时, C++服务收到的是字符串 `\u4f60\u597d`.
            # 虽然 python 会自动转换, 但我的 C++ 服务并没有. 因此, 需要确保 C++服务收到的是正确的字符串.
            # data=json.dumps(json_dict, ensure_ascii=False).encode("utf-8")
            # data=json.dumps(json_dict, ensure_ascii=False)
            data=json.dumps(json_dict)
        )
        if status_code != 200:
            raise AssertionError("requests failed, url: {}, status_code: {}, text: {}".format(
                self._url, status_code, text))
        js = json.loads(text)

        result = js
        self._cache.set(key, result)
        return result
