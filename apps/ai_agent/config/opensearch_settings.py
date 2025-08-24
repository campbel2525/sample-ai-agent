INDEX_CONFIG = {
    "settings": {
        # k-NN 有効化
        "index.knn": True,
        "index.knn.algo_param.ef_search": 100,
        # シャード設定
        "number_of_shards": 1,
        "number_of_replicas": 0,
        # Sudachi 設定
        "analysis": {
            "tokenizer": {
                "ja_sudachi": {
                    "type": "sudachi_tokenizer",
                    "resources_path": "/usr/share/opensearch/config/sudachi",
                    "additional_settings": '{"systemDict":"system.dic","userDict":[]}',
                    "split_mode": "C",
                    "discard_punctuation": True,
                }
            },
            "filter": {
                "ja_sudachi_split": {"type": "sudachi_split", "mode": "search"},
                "ja_sudachi_baseform": {"type": "sudachi_baseform"},
                "ja_sudachi_pos": {"type": "sudachi_part_of_speech"},
                "ja_sudachi_stop": {"type": "sudachi_ja_stop"},
            },
            "analyzer": {
                "ja_sudachi_analyzer": {
                    "type": "custom",
                    "tokenizer": "ja_sudachi",
                    "filter": [
                        "ja_sudachi_baseform",
                        "ja_sudachi_pos",
                        "ja_sudachi_stop",
                        "ja_sudachi_split",
                    ],
                }
            },
        },
    },
    "mappings": {
        "properties": {
            # Sudachi アナライザを適用するフィールド
            "content": {
                "type": "text",
                "analyzer": "ja_sudachi_analyzer",
            },
            # ベクトル検索用フィールド
            "vector": {
                "type": "knn_vector",
                "dimension": 3072,
                "method": {"name": "hnsw", "space_type": "l2", "engine": "faiss"},
            },
        }
    },
}
