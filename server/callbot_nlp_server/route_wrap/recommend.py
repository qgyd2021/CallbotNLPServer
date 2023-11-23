# -*- encoding=UTF-8 -*-
import json
import logging
import time
import traceback

import jsonschema

from server.callbot_nlp_server.schema.recommend import recommend_response_schema
from server.exception import ExpectedError

logger = logging.getLogger("server")


def recommend_route_wrap(f):
    async def inner(*args, **kwargs):
        self = args[0]
        req_body = json.loads(self.request.body)

        begin = time.time()
        try:
            recommend, addition = await f(*args, **kwargs)
            message = "success"
            response = {
                "code": 0,
                "retCode": 0,
                "retMsg": message,
                "results": recommend,
                **addition
            }
            # 响应体校验
            try:
                jsonschema.validate(response, recommend_response_schema)
            except (jsonschema.exceptions.ValidationError,
                    jsonschema.exceptions.SchemaError,) as e:
                logger.error(e)
            status_code = 200
        except ExpectedError as e:
            message = e.message
            response = {
                "code": 60501,
                "retCode": 60501,
                "message": message,
                "results": None,
                "detail": e.detail,
                "traceback": e.traceback,
            }
            status_code = 501

        except Exception as e:
            tb = traceback.format_exc()
            message = str(e)
            response = {
                "code": 60500,
                "retCode": 60500,
                "message": message,
                "results": None,
                "traceback": tb,
            }
            status_code = 500
            logger.error(tb)
            logger.error(e)

        cost = time.time() - begin
        cost = round(cost, 4)
        response["time_cost"] = cost
        # url| elapsed| code| msg| req body| rsp body
        log_message = "{uri}|{elapsed}|{code}|{msg}|{req_body}|{res_body}".format(
            uri=self.request.uri,
            elapsed=cost,
            code=status_code,
            msg=message,
            req_body=json.dumps(req_body, ensure_ascii=False),
            res_body=json.dumps(response, ensure_ascii=False),
        )
        logger.info(log_message)
        # logger.info("response: {}".format(json.dumps(response, ensure_ascii=False)))

        return response, status_code
    return inner
