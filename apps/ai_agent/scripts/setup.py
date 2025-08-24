from langchain.text_splitter import RecursiveCharacterTextSplitter

from config.opensearch_settings import INDEX_CONFIG
from config.settings import Settings
from services.openai_service import bulk_get_embeddings
from services.opensearch_service import bulk_insert_documents, recreate_index


def execute_embeddings(text_data: str):
    """
    1. チャンクを行う
    2. チャンクをベクトル化する
    """
    # チャンク
    # 512 文字、128 文字の重複
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        separators=["\n\n", "\n", "。", "、", " "],
    )
    chunks = text_splitter.split_text(text_data)

    # ベクトル化
    settings = Settings()
    embeddings = bulk_get_embeddings(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
        texts=chunks,
        max_retries=3,
    )

    return chunks, embeddings


def insert_open_search(
    chunks,
    embeddings,
):
    """
    OpenSearch にベクトルを挿入する処理
    """
    settings = Settings()

    documents = []
    for i, (text_data, vector_data) in enumerate(zip(chunks, embeddings)):
        document = {
            "_index": settings.opensearch_default_index_name,
            "_id": str(i + 1),
            "content": text_data,
            "vector": vector_data,
        }
        documents.append(document)

    # OpenSearch にデータを挿入
    bulk_insert_documents(
        host=settings.opensearch_host,
        port=settings.opensearch_port,
        user=settings.opensearch_user,
        password=settings.opensearch_password,
        documents=documents,
    )


def run_setup():
    """
    セットアップの実行
    """

    # OpenSearchのインデックスを再作成
    settings = Settings()
    recreate_index(
        host=settings.opensearch_host,
        port=settings.opensearch_port,
        user=settings.opensearch_user,
        password=settings.opensearch_password,
        opensearch_base_url=settings.opensearch_base_url,
        opensearch_index_name=settings.opensearch_default_index_name,
        index_config=INDEX_CONFIG,
    )

    # テストデータの読み込み
    with open("data/insert_data/test_data.txt", "r") as f:
        text_data = f.read()

    # チャンク、ベクトル化
    chunks, embeddings = execute_embeddings(text_data)

    # OpenSearchにデータの挿入
    insert_open_search(chunks, embeddings)


if __name__ == "__main__":
    # from config.debug import *

    run_setup()
