from typing import Any, Dict, Optional, Sequence

from ragas import evaluate
from ragas.dataset_schema import (
    EvaluationDataset,
    EvaluationResult,
)
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, answer_similarity

from ai_agents.agents.general_purpose_ai_agent.models import AgentResult
from config.settings import Settings
from services.openai_service import get_embedding_client, get_openai_client

settings = Settings()


def run_ragas(
    query: str,
    agent_result: AgentResult,
    ragas_dataset_data: Optional[Dict[str, str]] = None,
    ragas_metrics_data: Optional[Sequence[Any]] = None,
) -> EvaluationResult:
    """
    Langfuse を介して AI エージェントを実行し、任意で RAGAS による回答評価を行います。

    パラメータ
    ----------
    query : str
        ユーザー入力
    agent_result : AgentResult
        AIエージェントの実行結果。
    ragas_dataset_data : Optional[Dict[str, str]]
        RAGAS 用のデータセット情報。
    ragas_metrics_data : Optional[Sequence[Any]]
        使用する RAGAS の評価指標。未指定時は
        [answer_relevancy, answer_similarity] を用いる。

    戻り値
    ------
    EvaluationResult
    """

    # RAGAS評価の実行
    ragas_dataset = []
    if ragas_dataset_data is not None:
        ragas_dataset_data["user_input"] = query
        ragas_dataset_data["response"] = agent_result.answer
        ragas_dataset.append(ragas_dataset_data)

    ragas_metrics = []
    if ragas_metrics_data is not None:
        if "answer_relevancy" in ragas_metrics_data:
            ragas_metrics.append(answer_relevancy)
        if "answer_similarity" in ragas_metrics_data:
            ragas_metrics.append(answer_similarity)

    openai_client = get_openai_client(
        settings.openai_base_url,
        settings.openai_api_key,
        settings.openai_model,
    )

    embedding_client = get_embedding_client(
        settings.openai_base_url,
        settings.openai_api_key,
        settings.openai_embedding_model,
        max_retries=3,
    )

    ragas_scores: EvaluationResult = evaluate(
        dataset=EvaluationDataset.from_list(ragas_dataset),
        metrics=ragas_metrics,
        llm=LangchainLLMWrapper(openai_client),
        embeddings=LangchainEmbeddingsWrapper(embedding_client),
        raise_exceptions=True,
        show_progress=True,
    )

    return ragas_scores
