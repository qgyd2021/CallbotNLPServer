## 泰语



### 编辑距离
```json
{
    "recall": {
        "type": "elastic_search",
        "update_es": true,
        "tokenizer": {
            "type": "jieba"
        },
        "top_k": 30,
        "preprocess_list": [
            {
                "type": "do_lowercase"
            },
            {
                "type": "strip"
            },
            {
                "type": "zh_conv",
                "locale": "zh-cn"
            }
        ]
    },
    "scorer": {
        "type": "edit_distance",
        "max_edit_distance_list": [[[0, 3], [-1, -1]], [[4, 6], [-1, 1]], [[6, 9], [1, 2]], [[9, 15], [2, 3]], [[15, 20], [3, 5]], [[20, 200], [5, 7]]],
        "preprocess_list": [
            {
                "type": "do_lowercase"
            },
            {
                "type": "strip"
            },
            {
                "type": "zh_conv",
                "locale": "zh-cn"
            }
        ],
        "tokenizer": {
            "type": "list"
        },
        "filter_list": [

        ]
    }
}
```

### 同义词表匹配
```json
{
    "recall": {
        "type": "elastic_search",
        "update_es": true,
        "tokenizer": {
            "type": "pythainlp"
        },
        "top_k": 30,
        "preprocess_list": [
            {
                "type": "do_lowercase"
            },
            {
                "type": "strip"
            }
        ]
    },
    "scorer": {
        "type": "weighted_word_match",
        "language": "th",
        "synonyms_filename": "server/callbot_nlp_server/configs/weighted_word_xlsx/weighted_word_th.xlsx",
        "stages": [1.0, 0.85, 0.75, 0.65, 0.45, 0.35, 0.00],
        "scores": [1.0, 0.85, 0.75, 0.55, 0.35, 0.20, 0.00]
    }
}
```
