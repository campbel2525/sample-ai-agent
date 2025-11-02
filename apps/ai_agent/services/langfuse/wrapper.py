from typing import Optional

from openai.types.chat import ChatCompletionMessageParam

from ai_agents.agents.general_purpose_ai_agent.agent import Agent
from ai_agents.agents.general_purpose_ai_agent.models import AgentResult
from services.langfuse.tracer import LangfuseTracer
from config.custom_logger import setup_logger

logger = setup_logger(__file__)


def run_agent_with_langfuse(
    agent: Agent,
    query: str,
    langfuse_public_key: str,
    langfuse_secret_key: str,
    langfuse_host: str,
    chat_history: list[ChatCompletionMessageParam] = [],
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

    original_client = agent.client
    if tracer.is_available():
        try:
            langfuse_client = tracer.get_openai_client(
                api_key=agent.openai_api_key,
                base_url=agent.openai_base_url,
            )
            agent.client = langfuse_client
            logger.info("✅ Temporarily using Langfuse-integrated OpenAI client")
        except Exception as e:
            logger.warning(f"Failed to integrate Langfuse with agent: {e}")

    lf = tracer.get_client()

    if lf is not None:
        with lf.start_as_current_span(name=langfuse_trace_name) as span:
            try:
                span.update_trace(
                    name=langfuse_trace_name,  # ★ 引数を使用
                    input={"query": query, "chat_history": chat_history},
                    metadata={
                        "agent_type": "general_purpose_ai_agent",
                        "max_challenge_count": agent.max_challenge_count,
                        "tools": [tool.name for tool in agent.tools],
                        "has_chat_history": bool(chat_history),
                        "chat_history_length": len(chat_history) if chat_history else 0,
                    },
                    session_id=langfuse_session_id,
                    user_id=langfuse_user_id,
                )
            except Exception as e:
                logger.warning(f"Failed to update root trace metadata: {e}")

            try:
                logger.info(
                    f"🚀 Starting agent execution with Langfuse tracing ({langfuse_trace_name})..."  # noqa: E501
                )
                agent_result = agent.run_agent(query, chat_history)

                try:
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
                except Exception as e:
                    logger.warning(f"Failed to update trace output: {e}")

                logger.info("✅ Agent execution completed successfully")
                return agent_result

            except Exception as e:
                try:
                    span.update_trace(
                        output={"error": str(e)},
                        metadata={"execution_status": "error"},
                    )
                except Exception:
                    pass
                logger.error(f"Agent execution failed: {e}")
                raise

            finally:
                agent.client = original_client
                tracer.flush()

    try:
        logger.info("🚀 Starting agent execution without Langfuse tracing...")
        agent_result = agent.run_agent(query, chat_history)
        logger.info("✅ Agent execution completed successfully")
        return agent_result
    finally:
        agent.client = original_client
