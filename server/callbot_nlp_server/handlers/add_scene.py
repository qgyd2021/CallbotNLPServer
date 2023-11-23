#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import logging

import jsonschema

from server.tornado_server.route_wrap.common_async_route_wrap import common_async_route_wrap
from server.callbot_nlp_server.schema.add_scene import add_scene_request_schema
from server.tornado_server.handlers.base_handler import BaseHandler
from server.callbot_nlp_server.service.nlpbot import nlpbot

from server.exception import ExpectedError

logger = logging.getLogger('server')


class AddSceneHandler(BaseHandler):
    @common_async_route_wrap
    async def handle_post(self):
        args = json.loads(self.request.body)
        logger.info('args: {}'.format(args))

        # 请求体校验
        try:
            jsonschema.validate(args, add_scene_request_schema)
        except (jsonschema.exceptions.ValidationError,
                jsonschema.exceptions.SchemaError,) as e:
            raise ExpectedError(
                status_code=60401,
                message='request body invalid. ',
                detail=str(e)
            )

        product_id = args['product_id']
        scene_id = args['scene_id']
        group = args['group']
        env = args['env']

        # 计算结果.
        result = nlpbot.add_scene(
            product_id=product_id,
            scene_id=scene_id,
            group=group,
            env=env,
        )
        return result


if __name__ == '__main__':
    pass
