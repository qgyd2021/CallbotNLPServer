## 印尼语

注意事项: 
```text
1. 应设置参数 `delimiter=" "`. 


```

### 编辑距离句子相似度
```json
{
    "recall": {
        "type": "elastic_search",
        "update_es": true,
        "tokenizer": {
            "type": "nltk"
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
        "type": "edit_distance",
        "max_edit_distance_list": [[[0, 3], [-1, -1]], [[4, 6], [-1, 1]], [[6, 9], [1, 2]], [[9, 15], [2, 3]], [[15, 20], [3, 5]], [[20, 200], [5, 7]]],
        "preprocess_list": [
            {
                "type": "do_lowercase"
            },
            {
                "type": "strip"
            }
        ],
        "tokenizer": {
            "type": "list"
        }
    }
}
```
