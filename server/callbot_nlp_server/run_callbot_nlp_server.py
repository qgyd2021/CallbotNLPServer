#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import datetime
import logging
import os
from pathlib import Path
import sys

pwd = os.path.abspath(os.path.dirname(__file__))
project_path = Path(os.path.join(pwd, "../../"))
sys.path.append(project_path.as_posix())

os.environ["NLTK_DATA"] = (project_path / "data/nltk_data").as_posix()

import tornado.httpserver
import tornado.ioloop
import tornado.netutil
import tornado.process
import tornado.web

from server.callbot_nlp_server import settings
from server.callbot_nlp_server.handlers.add_scene import AddSceneHandler
from server.callbot_nlp_server.handlers.blank import BlankHandler
from server.callbot_nlp_server.handlers.delete_scene import DeleteSceneHandler
from server.callbot_nlp_server.handlers.recommend import RecommendHandler
from server.callbot_nlp_server.service.nlpbot import nlpbot
from server.tornado_server.handlers.heart_beat_handler import HeartBeatHandler
from server.tornado_server import log

logger = logging.getLogger("server")


# 初始化服务
application = tornado.web.Application([
    ("/HeartBeat", HeartBeatHandler),
    ("/chatbot/nlp", RecommendHandler),
    ("/chatbot/nlp/blank", BlankHandler),
    ("/chatbot/nlp/add_scene", AddSceneHandler),
    ("/chatbot/nlp/delete_scene", DeleteSceneHandler),

], autoreload=False)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--port",
    default=settings.port,
    type=int,
)
args = parser.parse_args()


if sys.platform == "win32" or settings.num_processes == 1:
    application.listen(args.port)

    tornado.ioloop.PeriodicCallback(
        callback=nlpbot.update__check_and_init_node,
        callback_time=settings.server_update_interval * 1000,
        # jitter=settings.server_update_interval * 1000 * 0.1,
    ).start()

    tornado.ioloop.PeriodicCallback(
        callback=nlpbot.update__replace_node,
        callback_time=settings.server_update_interval * 1000,
        # jitter=settings.server_update_interval * 1000 * 0.1,
    ).start()

    log.setup(log_directory=settings.log_directory)
    logger.info("server is already, port: {}".format(args.port))

    tornado.ioloop.IOLoop.current().start()
else:
    # multi process
    server = tornado.httpserver.HTTPServer(application)
    server.bind(args.port)
    server.start(num_processes=settings.num_processes)

    tornado.ioloop.PeriodicCallback(
        callback=nlpbot.update__check_and_init_node,
        callback_time=settings.server_update_interval * 1000,
        # jitter=settings.server_update_interval * 1000 * 0.1,
    ).start()

    tornado.ioloop.PeriodicCallback(
        callback=nlpbot.update__replace_node,
        callback_time=settings.server_update_interval * 1000,
        # jitter=settings.server_update_interval * 1000 * 0.1,
    ).start()

    # 在 server.start 之后再设置日志, 使每个进程有一个独立的日志文件.
    log.tornado_multi_process_setup(log_directory=settings.log_directory)
    logger.info("server is already, port: {}".format(args.port))

    tornado.ioloop.IOLoop.instance().start()
