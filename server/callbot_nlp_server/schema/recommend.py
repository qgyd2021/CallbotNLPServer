# -*- encoding=UTF-8 -*-
"""
http://json-schema.org/understanding-json-schema/


{
    "productID":"callbot",
    "userID":"6281211551620",
    "sceneID":"qnbtccxrlz",
    "callID":"3a326680-f6f4-47ca-8f5b-68b2ed65c51a",
    "currNodeID":"start_68zso77tya",
    "userInput":"saya menganggur dan saya kehabisan uang.",
    "createTs":1619084002,
    "sign":"0c09967acbcb1ae6bf9af93004121f8d"
}


{
"userInput": "不还会怎样", "retCode": 0, "retMsg": "success",
"results":

    [
        {
        "resID": "V5HJd8uUjRC7Qvi41E0LL",
        "nodeScore": 0.6186197996139526,
        "nodeType": 4,
        "nodeID": "im49r9svra",
        "wording": "怎样做"
        },
        {"resID": "xDqptjjReWJrImXIDoZ_a", "nodeScore": 0.6176659464836121, "nodeType": 4, "nodeID": "im49r9svra", "wording": "怎样使用"}, {"resID": "PN8-BPtXrqvD_WhY3Gv2y", "nodeScore": 0.6156961917877197, "nodeType": 4, "nodeID": "im49r9svra", "wording": "怎样还"}, {"resID": "ccyWbNxcoRsabrU07nUWy", "nodeScore": 0.6156895756721497, "nodeType": 4, "nodeID": "im49r9svra", "wording": "怎样还啊"}, {"resID": "tIEtw-DKmyuJoO_rt6tIo", "nodeScore": 0.6156772971153259, "nodeType": 4, "nodeID": "im49r9svra", "wording": "怎样申报"}, {"resID": "Ph0JLH5DvxsRuWijtIyns", "nodeScore": 0.5975302457809448, "nodeType": 4, "nodeID": "06u9xxibso", "wording": "逾期会怎样"}, {"resID": "cLh3GemXaszK1L1tLEnKm", "nodeScore": 0.5534751415252686, "nodeType": 4, "nodeID": "06u9xxibso", "wording": "不还会有什么后果"}, {"resID": "p5rdjwJvV2BuwLmqbEPOJ", "nodeScore": 0.336089551448822, "nodeType": 1, "nodeID": "task_23y32q7ztv", "wording": "会还"}, {"resID": "w_Nz2Bq6mKvG8UPlmVcyu", "nodeScore": 0.3192789852619171, "nodeType": 4, "nodeID": "3147ve21c7", "wording": "怎样查"}, {"resID": "vrpWvTCPGiD5VnIsFwW_d", "nodeScore": 0.3190959692001343, "nodeType": 4, "nodeID": "3147ve21c7", "wording": "怎样联系"}

    ]

}
"""


recommend_request_schema = {
    'type': 'object',
    'required': ['productID', 'sceneID', 'currNodeID', 'userInput'],
    'properties': {
        'productID': {
            'type': 'string',
        },
        'userID': {
            'type': 'string',
        },
        'sceneID': {
            'type': 'string',
        },
        'callID': {
            'type': 'string',
        },
        'currNodeID': {
            'type': 'string',
        },
        'userInput': {
            'type': 'string',
        },
        'createTs': {
            'type': 'integer',
        },
        'sign': {
            'type': 'string',
        },
        'env': {
            'type': 'string'
        },
        'topK': {
            'type': 'integer',
        },
        'debug': {
            'type': 'boolean'
        }
    }
}


recommend_response_schema = {
    'type': 'object',
    'required': ['userInput', 'retCode', 'retMsg', 'results'],
    'properties': {
        'userInput': {
            'type': 'string',
        },
        'retCode': {
            'type': 'integer',
        },
        'retMsg': {
            'type': 'string',
        },
        'results': {
            'type': 'array',
            'items': {
                'type': 'object',
                'required': ['resID', 'nodeScore', 'nodeType', 'nodeID', 'wording'],
                'properties': {
                    'resID': {
                        'type': 'string',
                    },
                    'nodeScore': {
                        'type': 'number',
                    },
                    'nodeType': {
                        'type': 'integer',
                    },
                    'nodeID': {
                        'type': 'string',
                    },
                    'wording': {
                        'type': 'string',
                    },
                }
            },
        },
    }
}
