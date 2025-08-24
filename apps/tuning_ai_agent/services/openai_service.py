from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr


def get_openai_client(
    base_url: str,
    api_key: str,
    model: str,
):
    """
    Chatモデル用の LangChain クライアント。
    /v1/chat/completions を叩く。
    """
    return ChatOpenAI(
        base_url=base_url,
        api_key=SecretStr(api_key),
        model=model,
        temperature=0,
    )


def get_embedding_client(
    base_url,
    api_key,
    model,
    max_retries: int = 5,
) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_retries=max_retries,
    )


def bulk_get_embeddings(
    base_url,
    api_key,
    model,
    texts,
    max_retries: int = 5,
):
    client = get_embedding_client(
        base_url=base_url,
        api_key=api_key,
        model=model,
        max_retries=max_retries,
    )

    return client.embed_documents(texts)
