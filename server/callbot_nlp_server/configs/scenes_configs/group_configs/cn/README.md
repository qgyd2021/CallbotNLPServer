## 汉语


### 编辑距离句子相似度
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


### 加权同义词句子相似度

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
            },
            {
                "type": "cn_repeat"
            }
        ]
    },
    "scorer": {
        "type": "weighted_word_match",
        "synonyms_filename": "server/nlpbot_server/config/weighted_word_xlsx/weighted_word_cn.xlsx",
        "ltp_data_path": "third_party_data/pyltp/ltp_data_v3.4.0",
        "stages": [1.0, 0.90, 0.80, 0.70, 0.60, 0.40, 0.00],
        "scores": [1.0, 0.85, 0.75, 0.55, 0.35, 0.20, 0.00]
    }
}
```


### KNN BERT 句向量相似度
```json
{
    "recall": {
        "type": "deep_knn",
        "http_allennlp_predictor": {
            "host_port": "$str:deep_knn_cn_host_port"
        },
        "tokenizer": {
            "type": "jieba"
        },
        "top_k": 30,
        "dim": 256,
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
        "type": "scale",
        "stages": [1.0, 0.80, 0.70, 0.60, 0.50, 0.40, 0.35, 0.20, 0.0],
        "scores": [1.0, 0.95, 0.85, 0.75, 0.70, 0.60, 0.35, 0.20, 0.0]
    }
}
```
