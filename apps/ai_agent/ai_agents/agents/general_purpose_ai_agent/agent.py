import json
import operator
from typing import (
    Annotated,
    Any,
    Iterable,
    Literal,
    Optional,
    Sequence,
    Type,
    TypedDict,
    cast,
)

from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
)
from pydantic import BaseModel

from .custom_logger import setup_logger
from .models import (
    AgentResult,
    AgentSettings,
    PromptSystemUser,
    PromptUserOnly,
    Plan,
    ReflectionResult,
    Subtask,
    ToolResult,
)

logger = setup_logger(__file__)


class AgentSubGraphState(TypedDict):
    query: str
    plan: list[str]
    subtask: str
    is_completed: bool
    messages: list[ChatCompletionMessageParam]
    challenge_count: int
    tool_results: Annotated[Sequence[Sequence[ToolResult]], operator.add]
    reflection_results: Annotated[Sequence[ReflectionResult], operator.add]
    subtask_answer: str


class AgentState(TypedDict):
    query: str
    chat_history: list[ChatCompletionMessageParam]
    plan: list[str]
    current_step: int
    subtask_results: Annotated[Sequence[Subtask], operator.add]
    last_answer: str


class Agent:

    def __init__(
        self,
        openai_base_url: str,
        openai_api_key: str,
        settings: AgentSettings | None = None,
        tools: list[BaseTool] = [],
        max_challenge_count: int = 3,
    ) -> None:
        self.openai_base_url = openai_base_url
        self.openai_api_key = openai_api_key
        self.settings = settings or AgentSettings()
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}

        # OpenAIクライアントを初期化
        self.client = OpenAI(
            base_url=self.openai_base_url,
            api_key=self.openai_api_key,
        )

        self.max_challenge_count = max_challenge_count

    def create_plan(self, state: AgentState) -> dict:
        """計画を作成する

        Args:
            state (AgentState): 入力の状態

        Returns:
            AgentState: 更新された状態
        """

        logger.info("🚀 Starting plan generation process...")

        # チャット履歴 + プロンプトからメッセージを生成
        messages: list[ChatCompletionMessageParam] = list(state.get("chat_history", []))
        planner_prompt = cast(PromptSystemUser, self.settings.planner.prompt)
        messages.append({"role": "system", "content": planner_prompt.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": planner_prompt.user_prompt.format(query=state["query"]),
            }
        )

        logger.debug(f"Final prompt messages: {messages}")

        # OpenAIにリクエストを送信
        try:
            logger.info("Sending request to OpenAI...")
            response = self._chat_parse(
                model=self.settings.planner.model_name,
                messages=messages,
                response_format=Plan,
                **self.settings.planner.model_params,
            )
            logger.info("✅ Successfully received response from OpenAI.")
        except Exception as e:
            logger.error(f"Error during OpenAI request: {e}")
            raise

        # レスポンスからStructured outputを利用しPlanクラスを取得
        plan = response.choices[0].message.parsed

        logger.info("Plan generation complete!")

        # 生成した計画を返し、状態を更新する
        return {"plan": plan.subtasks}

    def select_tools(self, state: AgentSubGraphState) -> dict:
        """ツールを選択する

        Args:
            state (AgentSubGraphState): 入力の状態

        Returns:
            dict: 更新された状態
        """

        logger.info("🚀 Starting tool selection process...")

        # OpenAI対応のtool定義に書き換える
        logger.debug("Converting tools for OpenAI format...")
        openai_tools = [convert_to_openai_tool(tool) for tool in self.tools]

        messages: list[ChatCompletionMessageParam]

        # リトライされたかどうかでプロンプトを切り替える
        if state["challenge_count"] == 0:
            logger.debug("Creating user prompt for tool selection...")
            st_prompt = cast(PromptSystemUser, self.settings.subtask_select_tool.prompt)
            user_prompt = st_prompt.user_prompt.format(
                query=state["query"], plan=state["plan"], subtask=state["subtask"]
            )
            messages = [
                {"role": "system", "content": st_prompt.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            try:
                logger.info("Sending request to OpenAI...")
                response = self._chat_create(
                    model=self.settings.subtask_select_tool.model_name,
                    messages=messages,
                    tools=openai_tools,
                    **self.settings.subtask_select_tool.model_params,
                )
                logger.info(response.choices[0].message.tool_calls)
                logger.info("✅ Successfully received response from OpenAI.")
            except Exception as e:
                logger.error(f"Error during OpenAI request: {e}")
                raise

        else:
            logger.debug("Creating user prompt for tool retry...")

            # NOTE: トークン数節約のため過去の検索結果は除く
            # roleがtoolまたはtool_callsを持つものは除く
            messages = [
                message
                for message in state["messages"]
                if message["role"] != "tool" and "tool_calls" not in message
            ]

            retry_prompt = cast(
                PromptUserOnly, self.settings.subtask_retry_answer.prompt
            )
            messages.append({"role": "user", "content": retry_prompt.user_prompt})

            try:
                logger.info("Sending request to OpenAI...")
                response = self._chat_create(
                    model=self.settings.subtask_retry_answer.model_name,
                    messages=messages,
                    tools=openai_tools,
                    **self.settings.subtask_retry_answer.model_params,
                )
                logger.info(response.choices[0].message.tool_calls)
                logger.info("✅ Successfully received response from OpenAI.")
            except Exception as e:
                logger.error(f"Error during OpenAI request: {e}")
                raise

        tool_calls = response.choices[0].message.tool_calls
        ai_message: ChatCompletionAssistantMessageParam = {
            "role": "assistant",
        }

        if tool_calls:
            ai_message["tool_calls"] = [tc.model_dump() for tc in tool_calls]
        else:
            ai_message["content"] = response.choices[0].message.content or ""

        logger.info("Tool selection complete!")
        messages.append(ai_message)

        return {"messages": messages}

    def execute_tools(self, state: AgentSubGraphState) -> dict:
        logger.info("🚀 Starting tool execution process...")
        messages = state["messages"]

        tool_calls = cast(Optional[list[Any]], messages[-1].get("tool_calls"))

        # ★ツールが無い＝スキップ（空結果で後段の型を満たす）
        if tool_calls is None or len(tool_calls) == 0:
            logger.warning("No tool calls found. Skipping tool execution.")
            return {"messages": messages, "tool_results": [[]]}

        # 以降は既存の実行ループ
        tool_results = []
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args_str = tool_call["function"]["arguments"]
            tool_args = json.loads(tool_args_str)

            tool = self.tool_map[tool_name]
            tool_result = tool.invoke(tool_args)

            tool_results.append(
                ToolResult(
                    tool_name=tool_name,
                    args=tool_args,
                    results=tool_result,
                )
            )

            messages.append(
                {
                    "role": "tool",
                    "content": str(tool_result),
                    "tool_call_id": tool_call["id"],
                }
            )

        logger.info("Tool execution complete!")
        return {"messages": messages, "tool_results": [tool_results]}

    def create_subtask_answer(self, state: AgentSubGraphState) -> dict:
        """サブタスク回答を作成する

        Args:
            state (AgentSubGraphState): 入力の状態

        Returns:
            dict: 更新された状態
        """

        logger.info("🚀 Starting subtask answer creation process...")
        messages = state["messages"]

        try:
            logger.info("Sending request to OpenAI...")
            # 回答生成は subtask_answer のモデルを使用（retry と同一設定を流用）
            response = self._chat_create(
                model=self.settings.subtask_retry_answer.model_name,
                messages=messages,
                **self.settings.subtask_retry_answer.model_params,
            )
            logger.info("✅ Successfully received response from OpenAI.")
        except Exception as e:
            logger.error(f"Error during OpenAI request: {e}")
            raise

        subtask_answer = response.choices[0].message.content

        ai_message = cast(
            ChatCompletionMessageParam,
            {
                "role": "assistant",
                "content": subtask_answer,
            },
        )
        messages.append(ai_message)

        logger.info("Subtask answer creation complete!")

        return {
            "messages": messages,
            "subtask_answer": subtask_answer,
        }

    def reflect_subtask(self, state: AgentSubGraphState) -> dict:
        """サブタスク回答を内省する

        Args:
            state (AgentSubGraphState): 入力の状態

        Raises:
            ValueError: reflection resultがNoneの場合

        Returns:
            dict: 更新された状態
        """

        logger.info("🚀 Starting reflection process...")
        messages = state["messages"]

        refl_prompt = cast(PromptUserOnly, self.settings.subtask_reflection.prompt)
        messages.append({"role": "user", "content": refl_prompt.user_prompt})

        try:
            logger.info("Sending request to OpenAI...")
            response = self._chat_parse(
                model=self.settings.subtask_reflection.model_name,
                messages=messages,
                response_format=ReflectionResult,
                **self.settings.subtask_reflection.model_params,
            )
            logger.info("✅ Successfully received response from OpenAI.")
        except Exception as e:
            logger.error(f"Error during OpenAI request: {e}")
            raise

        reflection_result = response.choices[0].message.parsed
        if reflection_result is None:
            raise ValueError("Reflection result is None")

        messages.append(
            {
                "role": "assistant",
                "content": reflection_result.model_dump_json(),
            }
        )

        update_state = {
            "messages": messages,
            "reflection_results": [reflection_result],
            "challenge_count": state["challenge_count"] + 1,
            "is_completed": reflection_result.is_completed,
        }

        if (
            update_state["challenge_count"] >= self.max_challenge_count
            and not reflection_result.is_completed
        ):
            update_state["subtask_answer"] = (
                f"{state['subtask']}の回答が見つかりませんでした。"
            )

        logger.info("Reflection complete!")
        return update_state

    def create_answer(self, state: AgentState) -> dict:
        """最終回答を作成する

        Args:
            state (AgentState): 入力の状態

        Returns:
            dict: 更新された状態
        """

        logger.info("🚀 Starting final answer creation process...")
        # サブタスク結果のうちタスク内容と回答のみを取得
        subtask_results = [
            (result.task_name, result.subtask_answer)
            for result in state["subtask_results"]
        ]
        fa_prompt = cast(PromptSystemUser, self.settings.final_answer.prompt)
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": fa_prompt.system_prompt},
            {
                "role": "user",
                "content": fa_prompt.user_prompt.format(
                    query=state["query"],
                    plan=state["plan"],
                    subtask_results=str(subtask_results),
                ),
            },
        ]

        try:
            logger.info("Sending request to OpenAI...")
            response = self._chat_create(
                model=self.settings.final_answer.model_name,
                messages=messages,
                **self.settings.final_answer.model_params,
            )
            logger.info("✅ Successfully received response from OpenAI.")
        except Exception as e:
            logger.error(f"Error during OpenAI request: {e}")
            raise

        logger.info("Final answer creation complete!")

        return {"last_answer": response.choices[0].message.content}

    def _execute_subgraph(self, state: AgentState):
        subgraph = self._create_subgraph()

        result = subgraph.invoke(
            {
                "query": state["query"],
                "plan": state["plan"],
                "subtask": state["plan"][state["current_step"]],
                "current_step": state["current_step"],
                "is_completed": False,
                "challenge_count": 0,
            }
        )

        subtask_result = Subtask(
            task_name=result["subtask"],
            tool_results=result["tool_results"],
            reflection_results=result["reflection_results"],
            is_completed=result["is_completed"],
            subtask_answer=result["subtask_answer"],
            challenge_count=result["challenge_count"],
        )

        return {"subtask_results": [subtask_result]}

    def _should_continue_exec_subtasks(self, state: AgentState) -> list:
        return [
            Send(
                "execute_subtasks",
                {
                    "query": state["query"],
                    "plan": state["plan"],
                    "current_step": idx,
                },
            )
            for idx, _ in enumerate(state["plan"])
        ]

    def _should_continue_exec_subtask_flow(
        self, state: AgentSubGraphState
    ) -> Literal["end", "continue"]:
        if (
            state["is_completed"]
            or state["challenge_count"] >= self.max_challenge_count
        ):
            return "end"
        else:
            return "continue"

    def _create_subgraph(self) -> Pregel:
        """サブグラフを作成する

        Returns:
            Pregel: サブグラフ
        """
        workflow = StateGraph(AgentSubGraphState)

        # ツール選択ノードを追加
        workflow.add_node("select_tools", self.select_tools)

        # ツール実行ノードを追加
        workflow.add_node("execute_tools", self.execute_tools)

        # サブタスク回答作成ノードを追加
        workflow.add_node("create_subtask_answer", self.create_subtask_answer)

        # サブタスク内省ノードを追加
        workflow.add_node("reflect_subtask", self.reflect_subtask)

        # ツール選択からスタート
        workflow.add_edge(START, "select_tools")

        # ノード間のエッジを追加
        workflow.add_edge("select_tools", "execute_tools")
        workflow.add_edge("execute_tools", "create_subtask_answer")
        workflow.add_edge("create_subtask_answer", "reflect_subtask")

        # サブタスク内省ノードの結果から繰り返しのためのエッジを追加
        workflow.add_conditional_edges(
            "reflect_subtask",
            self._should_continue_exec_subtask_flow,
            {"continue": "select_tools", "end": END},
        )

        app = workflow.compile()

        return app

    def create_graph(self) -> Pregel:
        """エージェントのメイングラフを作成する

        Returns:
            Pregel: エージェントのメイングラフ
        """
        workflow = StateGraph(AgentState)

        # Add the plan node
        workflow.add_node("create_plan", self.create_plan)

        # Add the execution step
        workflow.add_node("execute_subtasks", self._execute_subgraph)

        workflow.add_node("create_answer", self.create_answer)

        workflow.add_edge(START, "create_plan")

        # From plan we go to agent
        workflow.add_conditional_edges(
            "create_plan",
            self._should_continue_exec_subtasks,
        )

        # From agent, we replan
        workflow.add_edge("execute_subtasks", "create_answer")

        workflow.set_finish_point("create_answer")

        app = workflow.compile()

        return app

    def run_agent(
        self, query: str, chat_history: list[ChatCompletionMessageParam] = []
    ) -> AgentResult:
        """エージェントを実行する

        Args:
            query (str): 入力の質問
            chat_history (list[dict], optional): チャット履歴

        Returns:
            AgentResult: エージェントの実行結果
        """

        app = self.create_graph()
        result = app.invoke(
            {
                "query": query,
                "chat_history": chat_history,
                "current_step": 0,
            }
        )

        agent_result = AgentResult(
            query=query,
            plan=Plan(subtasks=result["plan"]),
            subtasks=result["subtask_results"],
            answer=result["last_answer"],
        )

        return agent_result

    def _chat_parse(
        self,
        *,
        model: str,
        messages: Iterable[ChatCompletionMessageParam],
        response_format: Type[BaseModel],
        **rest: Any,
    ):
        return self.client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_format,
            **rest,
        )

    def _chat_create(
        self,
        *,
        model: str,
        messages: Iterable[ChatCompletionMessageParam],
        **rest: Any,
    ):
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            **rest,
        )
