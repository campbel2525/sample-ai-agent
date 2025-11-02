"""Application settings using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    # OpenAI
    openai_base_url: str = ""
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-2024-08-06"
    openai_embedding_model: str = "text-embedding-3-large"
    openai_max_retries: int = 3

    # OpenSearch
    opensearch_host: str = ""
    opensearch_port: int = 0
    opensearch_user: str = ""
    opensearch_password: str = ""
    opensearch_base_url: str = ""
    opensearch_default_index_name: str = ""

    # Fast API
    fastapi_port: int = 0
    fastapi_host: str = ""

    # Langfuse settings (optional)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = ""

    # 開発用途
    debugpy_port: int = 0  # デバッグ用ポート
