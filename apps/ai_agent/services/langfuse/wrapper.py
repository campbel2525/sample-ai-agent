from typing import Optional

from openai.types.chat import ChatCompletionMessageParam

from ai_agents.agents.general_purpose_ai_agent.agent import Agent
from ai_agents.agents.general_purpose_ai_agent.models import AgentResult
from config.custom_logger import setup_logger
from services.langfuse.tracer import LangfuseTracer

logger = setup_logger(__file__)


def run_agent_with_langfuse(
    agent: Agent,
    query: str,
    chat_history: list[ChatCompletionMessageParam],
    langfuse_public_key: str,
    langfuse_secret_key: str,
    langfuse_host: str,
    langfuse_session_id: Optional[str] = None,
    langfuse_user_id: Optional[int] = None,
    langfuse_trace_name: str = "ai_agent_execution",
) -> AgentResult:
    """
    エージェントをLangfuseトレーシング付きで実行する
    Args:
        agent (Agent): 純粋なエージェントインスタンス
        query (str): 入力の質問
        langfuse_public_key (str): LangfuseのPublic Key
        langfuse_secret_key (str): LangfuseのSecret Key
        langfuse_host (str): LangfuseのHost URL
        session_id (Optional[str]): LangfuseのセッションID（会話やスレッドを束ねたいときに指定）
        user_id (Optional[str]): LangfuseのユーザーID（任意・集計や検索用）
        trace_name (str): トレース名（デフォルト: "ai_agent_execution"）

    Returns:
        AgentResult: エージェントの実行結果
    """
    tracer = LangfuseTracer(
        public_key=langfuse_public_key,
        secret_key=langfuse_secret_key,
        host=langfuse_host,
    )
    if not tracer.is_available():
        raise Exception(
            "Langfuse tracer is not available with the provided credentials."
        )

    langfuse_client = tracer.get_openai_client(
        api_key=agent.openai_api_key,
        base_url=agent.openai_base_url,
    )
    original_client = agent.client
    try:
        agent.client = langfuse_client
        logger.info("✅ Temporarily using Langfuse-integrated OpenAI client")

        lf = tracer.get_client()
        with lf.start_as_current_span(name=langfuse_trace_name) as span:
            # AgentSetting の概要（存在すれば）をメタデータに付与
            settings_meta = None
            s = getattr(agent, "settings", None)
            if s is not None:
                settings_meta = {
                    "planner_model": s.planner.model_name,
                    "subtask_select_tool_model": s.subtask_select_tool.model_name,
                    "subtask_reflection_model": s.subtask_reflection.model_name,
                    "subtask_retry_answer_model": s.subtask_retry_answer.model_name,
                    "final_answer_model": s.final_answer.model_name,
                }

            span.update_trace(
                name=langfuse_trace_name,  # ★ 引数を使用
                input={"query": query, "chat_history": chat_history},
                metadata={
                    "agent_type": "general_purpose_ai_agent",
                    "max_challenge_count": agent.max_challenge_count,
                    "tools": [tool.name for tool in agent.tools],
                    "has_chat_history": bool(chat_history),
                    "chat_history_length": len(chat_history) if chat_history else 0,
                    "agent_settings": settings_meta,
                },
                session_id=langfuse_session_id,
                user_id=langfuse_user_id,
            )

            logger.info(
                f"🚀 Starting agent execution with Langfuse tracing ({langfuse_trace_name})..."  # noqa: E501
            )
            agent_result = agent.run_agent(query, chat_history)

            plan = getattr(agent_result, "plan", None)
            output = {
                "answer": agent_result.answer,
                "plan": plan.subtasks if plan is not None else None,
                "subtask_count": len(getattr(agent_result, "subtasks", [])),
            }
            metadata = {
                "execution_status": "success",
                "total_subtasks": len(getattr(agent_result, "subtasks", [])),
            }
            span.update_trace(
                output=output,
                metadata=metadata,
            )

            logger.info("✅ Agent execution completed successfully")
            return agent_result
    finally:
        agent.client = original_client
        tracer.flush()
