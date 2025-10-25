import json

import requests
from opensearchpy import OpenSearch, helpers

from services.openai_service import get_embedding_client


def get_opensearch_client(
    host: str,
    port: int,
    user: str,
    password: str,
):
    """
    OpenSearchクライアントを生成して返す関数

    Args:
        host (str): OpenSearchのホスト名
        port (int): OpenSearchのポート番号
        user (str): ユーザー名
        password (str): パスワード

    Returns:
        OpenSearch: 設定済みのOpenSearchクライアント
    """
    opensearch_client = OpenSearch(
        hosts=[
            {
                "host": host,
                "port": port,
            },
        ],
        http_auth=(user, password),
        http_compress=True,  # 圧縮有効化
        use_ssl=False,  # SSLは無効（環境によって変更可）
        verify_certs=False,  # 証明書検証無効
        timeout=30,  # タイムアウト秒数
        retry_on_timeout=True,  # タイムアウト時のリトライ有効化
        max_retries=5,  # リトライ最大回数
    )
    return opensearch_client


def recreate_index(
    host: str,
    port: int,
    user: str,
    password: str,
    opensearch_base_url: str,
    opensearch_index_name: str,
    index_config: dict,
):
    """
    インデックスを再作成する関数

    指定したインデックスが存在する場合は削除し、
    新たにインデックスを作成した上でハイブリッド検索用パイプラインを設定する。

    Args:
        host (str): OpenSearchのホスト名
        port (int): OpenSearchのポート番号
        user (str): ユーザー名
        password (str): パスワード
        opensearch_base_url (str): OpenSearchのベースURL
        opensearch_index_name (str): 作成するインデックス名
        index_config (dict): インデックス設定
    """
    opensearch_client = get_opensearch_client(host, port, user, password)

    # 既存インデックスがある場合は削除
    if opensearch_client.indices.exists(index=opensearch_index_name):
        opensearch_client.indices.delete(index=opensearch_index_name)
        print(f"Existing index '{opensearch_index_name}' has been deleted.")

    # 新規インデックスを作成
    opensearch_client.indices.create(index=opensearch_index_name, body=index_config)

    # ハイブリッド検索用パイプラインの設定
    url = f"{opensearch_base_url}/_search/pipeline/hybrid-pipeline"
    data = {
        "description": "Post-processor for hybrid search",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "l2"},
                    "combination": {"technique": "arithmetic_mean"},
                }
            }
        ],
    }
    requests.put(url, verify=False, json=data)


def bulk_insert_documents(
    host: str,
    port: int,
    user: str,
    password: str,
    documents,
):
    """
    複数のドキュメントを一括でインデックスに投入する関数

    Args:
        host (str): OpenSearchのホスト名
        port (int): OpenSearchのポート番号
        user (str): ユーザー名
        password (str): パスワード
        documents (iterable): 登録するドキュメント群
    """
    opensearch_client = get_opensearch_client(host, port, user, password)
    helpers.bulk(opensearch_client, documents)


def hybrid_search(
    openai_api_key,
    openai_base_url,
    openai_embedding_model,
    openai_max_retries,
    opensearch_base_url: str,
    opensearch_index_name: str,
    question: str,
    k=50,
    size=20,
):
    """
    ハイブリッド検索を実行する関数

    グルテキスト検索とベクトル検索を組み合わせて
    高精度な検索結果を返す。

    Args:
        host (str): OpenSearchのホスト名
        port (int): OpenSearchのポート番号
        user (str): ユーザー名
        password (str): パスワード
        opensearch_base_url (str): OpenSearchのベースURL
        opensearch_index_name (str): 検索対象インデックス名
        question (str): 検索クエリ（自然文）
        k (int): ベクトル検索で取得する件数
        size (int): 最終的に返す件数

    Returns:
        dict: 検索結果
    """
    # 埋め込みベクトル生成
    embedding_client = get_embedding_client(
        openai_base_url,
        openai_api_key,
        openai_embedding_model,
        openai_max_retries,
    )
    query_embedding = embedding_client.embed_query(question)

    # ハイブリッド検索クエリの組み立て
    query = {
        "_source": {"exclude": ["vector"]},
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "bool": {
                            "must": [{"match": {"content": question}}],
                        }
                    },
                    {
                        "knn": {
                            "vector": {
                                "vector": query_embedding,
                                "k": k,
                            }
                        }
                    },
                ]
            }
        },
        "size": size,
    }

    # ハイブリッド検索リクエスト
    response = requests.post(
        f"{opensearch_base_url}/{opensearch_index_name}/_search?search_pipeline=hybrid-pipeline",  # noqa: E501
        verify=False,
        json=query,
    )
    if response.status_code != 200:
        raise Exception(response.text)

    result = json.loads(response.text)
    return result
