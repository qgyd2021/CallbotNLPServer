## 越南语

```text
黑白名单正则表达式

意图标签和实体匹配

加权同义词表句子相似度匹配

```

### 决策树
```json
{
    "recall": {
        "type": "elastic_search",
        "top_k": 30,
        "tokenizer": {
            "type": "nltk"
        },
        "preprocess_list": [
            {
                "type": "vietnamese_lowercase"
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
                "type": "vietnamese_lowercase"
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
