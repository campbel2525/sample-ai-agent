"""Langfuseトレーシング機能"""

from langfuse import Langfuse
from langfuse.openai import openai as langfuse_openai

from config.custom_logger import setup_logger

LANGFUSE_AVAILABLE = True

logger = setup_logger(__file__)


class LangfuseTracer:
    """Langfuseトレーシングを管理するクラス"""

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str,
    ) -> None:
        self.public_key = public_key
        self.secret_key = secret_key
        self.host = host
        self.langfuse = None

        try:
            self.langfuse = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host,
            )
            logger.info("✅ Langfuse client initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize Langfuse client: {e}")
            self.langfuse = None

    def is_available(self) -> bool:
        """Langfuseが利用可能かどうかを返す"""
        return self.langfuse is not None

    def flush(self) -> None:
        """Langfuseにデータを送信する"""
        if self.langfuse is None:  # ← 直接 None チェック
            return
        try:
            self.langfuse.flush()  # ここで mypy OK
        except Exception as e:
            logger.warning(f"Failed to flush Langfuse data: {e}")

    def get_openai_client(self, api_key: str, base_url: str):
        """Langfuse統合OpenAIクライアントを取得する"""
        if self.is_available() and langfuse_openai:
            try:
                return langfuse_openai.OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to create Langfuse-integrated OpenAI client: {e}"
                )

        # フォールバック: 標準のOpenAIクライアント
        from openai import OpenAI

        return OpenAI(api_key=api_key, base_url=base_url)

    def get_client(self):
        """内部で初期化した Langfuse クライアントを返す"""
        return self.langfuse
