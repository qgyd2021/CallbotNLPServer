## 意大利语


### 决策树
```text
{
    "recall": {
        "type": "elastic_search",
        "top_k": 30,
        "tokenizer": {
            "type": "nltk"
        },
        "preprocess_list": [
            {
                "type": "do_lowercase"
            },
            {
                "type": "strip"
            }
        ],
        "update_es": true
    },
    "scorer": {
        "type": "decision_tree",
        "candidates": {
            "type": "mysql",
            "resource_type_list": ["similar_question"]
        },
        "tokenizer": {
            "type": "nltk"
        },
        "preprocess_list": [
            {
                "type": "do_lowercase"
            },
            {
                "type": "strip"
            }
        ],
        "min_samples_split": 2,
        "min_samples_leaf": 1
    }
}
```
