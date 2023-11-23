#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import logging

import jsonschema

from nxtech.nlpbot.node import RequestContext
from server.exception import ExpectedError
from server.callbot_nlp_server.route_wrap.recommend import recommend_route_wrap
from server.callbot_nlp_server.schema.recommend import recommend_request_schema
from server.callbot_nlp_server.service.nlpbot import nlpbot
from server.tornado_server.handlers.base_handler import BaseHandler

logger = logging.getLogger('server')


class RecommendHandler(BaseHandler):
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

        # 计算结果.
        product_id = args['productID']
        scene_id = args['sceneID']
        node_id = args['currNodeID']
        text = args['userInput']
        env = args.get('env', 'ppe')
        debug = args.get('debug', False)

        context = RequestContext(
            product_id=product_id,
            scene_id=scene_id,
            node_id=node_id,
            env=env,
            text=text,
            debug=debug,
        )

        context: RequestContext = await nlpbot.recommend(
            context=context,
        )

        scores = context.result

        recommend = list()
        for score in scores:
            recommend.append({
                'resID': score['resource_id'],
                'nodeScore': score['score'],
                'nodeType': score['node_type'],
                'nodeID': score['node_id'],
                'nodeDesc': score['node_desc'],
                'wording': score['text'],
                'metric': score['metric'],
                'recaller': score['recaller'],
                'scorer': score['scorer']
            })
        addition = {
            'userInput': text,
        }

        if context.debug:
            addition['report'] = context.report
        return recommend, addition


if __name__ == '__main__':
    pass
