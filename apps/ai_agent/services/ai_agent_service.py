from typing import Any, Optional, Sequence, Tuple

from openai.types.chat import ChatCompletionMessageParam
from ragas import evaluate
from ragas.dataset_schema import (
    EvaluationDataset,
    EvaluationResult,
)
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, answer_similarity

from ai_agents.agents.general_purpose_ai_agent.agent import Agent
from ai_agents.agents.general_purpose_ai_agent.models import (
    AgentResult,
    LLMPhaseConfigs,
)
from services.langfuse.wrapper import run_agent_with_langfuse
from config.settings import Settings
from services.openai_service import get_embedding_client, get_openai_client


def run_ai_agent(
    question: str,
    chat_history: list[ChatCompletionMessageParam],
    llm_phase_configs: LLMPhaseConfigs,
    ai_agent_tools: Any,
    ai_agent_prompts: Any,
    ai_agent_max_challenge_count: int = 3,
    langfuse_session_id: Optional[str] = None,
    langfuse_trace_name: str = "ai_agent_execution",
) -> AgentResult:
    """
    Langfuse を介して AI エージェントを実行し、任意で RAGAS による回答評価を行います。

    パラメータ
    ----------
    tools : Any
        エージェントに渡すツール群。
    ai_agent_prompts : Any
        エージェントのプロンプト定義。
    question : str
        ユーザーからの質問。
    is_run_ragas : bool, default True
        True の場合はエージェント実行後に RAGAS 評価を実施。
        False の場合はエージェント実行結果のみ返す。
    # ragas_retrieved_contexts : Optional[Sequence[str]]
    #     検索で取得したコンテキスト群。
    ragas_reference : Optional[str]
        正解参照テキスト。
    ragas_reference_contexts : Optional[Sequence[str]]
        参照側のコンテキスト群。
    ragas_metrics : Optional[Sequence[Any]]
        使用する RAGAS の評価指標。未指定時は
        [answer_relevancy, answer_similarity] を用いる。

    戻り値
    ------
    Union[AgentResult, Tuple[AgentResult, EvaluationResult]]
        is_run_ragas=False の場合は AgentResult を返し、
        True の場合は (AgentResult, EvaluationResult) を返す。
    """
    settings = Settings()

    # エージェント定義
    agent = Agent(
        openai_base_url=settings.openai_base_url,
        openai_api_key=settings.openai_api_key,
        llm_phase_configs=llm_phase_configs,
        tools=ai_agent_tools,
        prompts=ai_agent_prompts,
        max_challenge_count=ai_agent_max_challenge_count,
    )

    # Langfuse経由でAIエージェントを実行
    agent_result: AgentResult = run_agent_with_langfuse(
        agent=agent,
        question=question,
        chat_history=chat_history,
        langfuse_public_key=settings.langfuse_public_key,
        langfuse_secret_key=settings.langfuse_secret_key,
        langfuse_host=settings.langfuse_host,
        langfuse_session_id=langfuse_session_id,
        langfuse_trace_name=langfuse_trace_name,
    )

    return agent_result


def run_ai_agent_with_rags(
    question: str,
    chat_history: list[ChatCompletionMessageParam],
    llm_phase_configs: LLMPhaseConfigs,
    ai_agent_tools: Any,
    ai_agent_prompts: Any,
    ai_agent_max_challenge_count: int = 3,
    langfuse_session_id: Optional[str] = None,
    langfuse_trace_name: str = "ai_agent_execution",
    # ragas_retrieved_contexts: Optional[Sequence[str]] = None,
    ragas_reference: Optional[str] = None,
    # ragas_reference_contexts: Optional[Sequence[str]] = None,
    ragas_metrics: Optional[Sequence[Any]] = None,
) -> Tuple[AgentResult, EvaluationResult]:
    """
    Langfuse を介して AI エージェントを実行し、任意で RAGAS による回答評価を行います。

    パラメータ
    ----------
    tools : Any
        エージェントに渡すツール群。
    ai_agent_prompts : Any
        エージェントのプロンプト定義。
    question : str
        ユーザーからの質問。
    is_run_ragas : bool, default True
        True の場合はエージェント実行後に RAGAS 評価を実施。
        False の場合はエージェント実行結果のみ返す。
    # ragas_retrieved_contexts : Optional[Sequence[str]]
    #     検索で取得したコンテキスト群。
    ragas_reference : Optional[str]
        正解参照テキスト。
    ragas_reference_contexts : Optional[Sequence[str]]
        参照側のコンテキスト群。
    ragas_metrics : Optional[Sequence[Any]]
        使用する RAGAS の評価指標。未指定時は
        [answer_relevancy, answer_similarity] を用いる。

    戻り値
    ------
    Union[AgentResult, Tuple[AgentResult, EvaluationResult]]
        is_run_ragas=False の場合は AgentResult を返し、
        True の場合は (AgentResult, EvaluationResult) を返す。
    """
    settings = Settings()

    # エージェント定義
    agent = Agent(
        openai_base_url=settings.openai_base_url,
        openai_api_key=settings.openai_api_key,
        llm_phase_configs=llm_phase_configs,
        tools=ai_agent_tools,
        prompts=ai_agent_prompts,
        max_challenge_count=ai_agent_max_challenge_count,
    )

    # Langfuse経由でAIエージェントを実行
    agent_result: AgentResult = run_agent_with_langfuse(
        agent=agent,
        question=question,
        chat_history=chat_history,
        langfuse_public_key=settings.langfuse_public_key,
        langfuse_secret_key=settings.langfuse_secret_key,
        langfuse_host=settings.langfuse_host,
        langfuse_session_id=langfuse_session_id,
        langfuse_trace_name=langfuse_trace_name,
    )

    # RAGAS評価の実行
    ragas_data = [
        {
            "user_input": question,  # ユーザー入力
            # "retrieved_contexts": ragas_retrieved_contexts,
            "response": agent_result.answer,
            "reference": ragas_reference,  # 正しい回答
            # "reference_contexts": ragas_reference_contexts,
        }
    ]

    if ragas_metrics is None:
        ragas_metrics = [
            answer_relevancy,
            answer_similarity,
        ]

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
        dataset=EvaluationDataset.from_list(ragas_data),
        metrics=ragas_metrics,
        llm=LangchainLLMWrapper(openai_client),
        embeddings=LangchainEmbeddingsWrapper(embedding_client),
        raise_exceptions=True,
        show_progress=True,
    )

    return agent_result, ragas_scores
