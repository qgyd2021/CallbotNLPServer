#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import tornado.web


class BaseHandler(tornado.web.RequestHandler):
    def set_default_headers(self) -> None:
        self.set_header("Content-Type", "application/json; charset=utf-8")

    def initialize(self, *args, **kwargs):
        pass

    async def handle_get(self):
        raise NotImplementedError

    async def handle_post(self):
        raise NotImplementedError

    async def get(self):
        response, status_code = await self.handle_get()
        response = json.dumps(response)

        self.write(response)
        self.set_status(status_code=status_code)

    async def post(self):
        response, status_code = await self.handle_post()
        response = json.dumps(response)

        self.write(response)
        self.set_status(status_code=status_code)


if __name__ == '__main__':
    pass
