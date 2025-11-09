from typing import Any, Optional

from openai.types.chat import ChatCompletionMessageParam

from ai_agents.agents.general_purpose_ai_agent.agent import Agent
from ai_agents.agents.general_purpose_ai_agent.models import (
    AgentResult,
    AgentSetting,
)
from config.settings import Settings
from services.langfuse_service import run_agent_with_langfuse

settings = Settings()


def run_ai_agent(
    query: str,
    chat_history: list[ChatCompletionMessageParam],
    ai_agent_setting: AgentSetting,
    ai_agent_tools: Any,
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
    ai_agent_setting : AgentSetting
        フェーズ別のプロンプトとモデル設定。
    query : str
        ユーザー入力
    is_execute_ragas : bool, default True
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
        is_execute_ragas=False の場合は AgentResult を返し、
        True の場合は (AgentResult, EvaluationResult) を返す。
    """
    # エージェント定義
    agent = Agent(
        openai_base_url=settings.openai_base_url,
        openai_api_key=settings.openai_api_key,
        settings=ai_agent_setting,
        tools=ai_agent_tools,
        max_challenge_count=ai_agent_max_challenge_count,
    )

    # Langfuse経由でAIエージェントを実行
    agent_result: AgentResult = run_agent_with_langfuse(
        agent=agent,
        query=query,
        chat_history=chat_history,
        langfuse_public_key=settings.langfuse_public_key,
        langfuse_secret_key=settings.langfuse_secret_key,
        langfuse_host=settings.langfuse_host,
        langfuse_session_id=langfuse_session_id,
        langfuse_trace_name=langfuse_trace_name,
    )

    return agent_result
