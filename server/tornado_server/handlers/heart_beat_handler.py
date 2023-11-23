#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tornado.web

from server.tornado_server.route_wrap.common_async_route_wrap import common_async_route_wrap
from server.tornado_server.handlers.base_handler import BaseHandler


class HeartBeatHandler(BaseHandler):

    @common_async_route_wrap
    async def handle_get(self):
        return 'OK'

    @common_async_route_wrap
    async def handle_post(self):
        return 'OK'


if __name__ == '__main__':
    pass
