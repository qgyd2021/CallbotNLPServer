#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))

import tornado.httpserver
import tornado.ioloop
import tornado.netutil
import tornado.process
import tornado.web

from server.tornado_server import settings

from server.tornado_server.handlers.heart_beat_handler import HeartBeatHandler


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write('Hello, world')


application = tornado.web.Application([
    ('/', MainHandler),
    ('/HeartBeat', HeartBeatHandler)
])


parser = argparse.ArgumentParser()
parser.add_argument(
    '--port',
    default=settings.port,
    type=int,
)
args = parser.parse_args()


if sys.platform == 'win32':
    application.listen(args.port)
    print('server already, port: {}'.format(args.port))
    tornado.ioloop.IOLoop.instance().start()
else:
    # 多进程
    # https://blog.csdn.net/luoganttcc/article/details/119679922

    # method 1 (无效)
    # tornado.process.fork_processes(num_processes=0)
    #
    # sockets = tornado.netutil.bind_sockets(args.port)
    # server = tornado.httpserver.HTTPServer(application)
    # server.add_sockets(sockets)
    #
    # tornado.ioloop.IOLoop.instance().start()

    # method 2
    server = tornado.httpserver.HTTPServer(application)
    server.bind(args.port)
    server.start(0)
    print('server already, port: {}'.format(args.port))
    tornado.ioloop.IOLoop.instance().start()
