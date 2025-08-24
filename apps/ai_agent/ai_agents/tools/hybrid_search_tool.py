from typing import Any, Dict, List, Type, Union

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from config.custom_logger import setup_logger
from services.opensearch_service import hybrid_search

logger = setup_logger(__file__)


# =========================
# ツールの引数スキーマ（実行時）
# =========================
class HybridSearchInput(BaseModel):
    """
    ハイブリッド検索の実行時パラメータ。
    """

    question: str = Field(..., description="検索クエリ（自然文）")
    k: int = Field(50, ge=1, le=1000, description="ベクトル検索で取得する候補件数")
    size: int = Field(20, ge=1, le=1000, description="最終的に返す件数")


# =========================
# ツール本体
# =========================
class HybridSearchTool(BaseTool):
    """
    OpenSearch に対して BM25 + ベクトルのハイブリッド検索を実行し、
    検索結果の _source のみを返すツール。
    """

    name: str = "hybrid_search_tool"
    description: str = (
        "OpenSearchでハイブリッド検索を実行し、_source の配列のみ返します。"
        "コンストラクタでOpenAIとOpenSearchの設定を受け取り、実行時はquestion/k/sizeを指定します。"
    )
    args_schema: Type[BaseModel] = HybridSearchInput

    # コンストラクタで保持する設定値
    openai_api_key: str
    openai_base_url: str
    openai_embedding_model: str
    openai_max_retries: int
    opensearch_base_url: str
    opensearch_index_name: str

    def __init__(
        self,
        *,
        openai_api_key: str,
        openai_base_url: str,
        openai_embedding_model: str,
        openai_max_retries: int,
        opensearch_base_url: str,
        opensearch_index_name: str,
    ):
        """
        ツールの初期化。OpenAI API設定とOpenSearch設定を受け取り保持します。
        """
        super().__init__(
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            openai_embedding_model=openai_embedding_model,
            openai_max_retries=openai_max_retries,
            opensearch_base_url=opensearch_base_url,
            opensearch_index_name=opensearch_index_name,
        )

    @staticmethod
    def _extract_sources(result: Dict[str, Any], size: int) -> List[Dict[str, Any]]:
        """
        OpenSearchのレスポンスから_sourceのみ抽出。
        """
        hits = (result.get("hits") or {}).get("hits") or []
        return [h["_source"] for h in hits[:size] if isinstance(h.get("_source"), dict)]

    def _run(
        self,
        question: str,
        k: int = 50,
        size: int = 20,
    ) -> Union[List[Dict[str, Any]], Dict[str, str]]:
        """
        実行時に検索クエリを受け取り、OpenSearchでハイブリッド検索を実行します。
        """
        logger.info(
            f"HybridSearchTool start | index={self.opensearch_index_name} "
            f"k={k} size={size} q_len={len(question)}"
        )
        try:
            raw = hybrid_search(
                openai_api_key=self.openai_api_key,
                openai_base_url=self.openai_base_url,
                openai_embedding_model=self.openai_embedding_model,
                openai_max_retries=self.openai_max_retries,
                opensearch_base_url=self.opensearch_base_url,
                opensearch_index_name=self.opensearch_index_name,
                question=question,
                k=k,
                size=size,
            )
            hits_count = len(((raw.get("hits") or {}).get("hits") or []))
            logger.info(f"HybridSearchTool success | raw_hits={hits_count}")
            return self._extract_sources(raw, size=size)
        except Exception as e:
            msg = f"ハイブリッド検索エラー: {str(e)}"
            logger.error(msg)
            return {"error": msg}
