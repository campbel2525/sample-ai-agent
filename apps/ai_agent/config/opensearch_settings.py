INDEX_CONFIG = {
    "settings": {
        "index": {"knn": True},
        "knn.algo_param.ef_search": 100,
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            "content": {"type": "text", "analyzer": "standard"},
            "vector": {
                "type": "knn_vector",
                "dimension": 3072,
                "method": {"name": "hnsw", "space_type": "l2", "engine": "faiss"},
            },
        }
    },
}
