# -*- encoding=UTF-8 -*-

add_scene_request_schema = {
    'type': 'object',
    'required': ['product_id', 'scene_id', 'language', 'env'],
    'properties': {
        'product_id': {
            'type': 'string'
        },
        'scene_id': {
            'type': 'string'
        },
        'group': {
            'type': 'string'
        },
        'env': {
            'type': 'string'
        },
    }
}
