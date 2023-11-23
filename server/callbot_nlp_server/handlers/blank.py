#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import logging

import jsonschema

from server.exception import ExpectedError
from server.callbot_nlp_server.route_wrap.recommend import recommend_route_wrap
from server.callbot_nlp_server.schema.recommend import recommend_request_schema
from server.tornado_server.handlers.base_handler import BaseHandler

logger = logging.getLogger('server')


class BlankHandler(BaseHandler):
    """任何场景 scene_id 访问时, 都返回空列表, 以触发答非所问. """
    @recommend_route_wrap
    async def handle_post(self):
        args = json.loads(self.request.body)

        # 请求体校验
        try:
            jsonschema.validate(args, recommend_request_schema)
        except (jsonschema.exceptions.ValidationError,
                jsonschema.exceptions.SchemaError,) as e:
            raise ExpectedError(
                status_code=60401,
                message='request body invalid. ',
                detail=str(e)
            )

        text = args['userInput']

        recommend = list()
        addition = {
            'userInput': text,
        }
        return recommend, addition


if __name__ == '__main__':
    pass
