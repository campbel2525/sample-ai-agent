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
    AgentSetting,
    Plan,
    ReflectionResult,
    Subtask,
    ToolResult,
)

logger = setup_logger(__file__)


class AgentSubGraphState(TypedDict):
    """サブグラフ（単一サブタスク実行）で用いる状態。

    各サブタスクについて、ツール選択→実行→回答生成→内省の
    一連の処理で受け渡すデータを保持します。
    """

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
    """メイングラフ（全体実行）で用いる状態。

    計画作成、各サブタスクの集約、最終回答作成のための
    入力・中間結果・最終結果を保持します。
    """

    query: str
    chat_history: list[ChatCompletionMessageParam]
    plan: list[str]
    current_step: int
    subtask_results: Annotated[Sequence[Subtask], operator.add]
    answer: str


class Agent:
    """汎用RAGエージェント。

    - 計画作成（質問分解）
    - サブタスク実行（ツール選択/実行→回答→内省の繰り返し）
    - 最終回答作成（全サブタスク結果の統合）

    をLangGraphで構成して実行します。
    """

    def __init__(
        self,
        openai_base_url: str,
        openai_api_key: str,
        settings: AgentSetting | None = None,
        tools: list[BaseTool] = [],
        max_challenge_count: int = 3,
        # チャット履歴の最大使用件数（Noneで全件）
        chat_history_max_turns: Optional[int] = None,
    ) -> None:
        """エージェントを初期化する。

        Args:
            openai_base_url (str): OpenAI互換エンドポイントのベースURL。
            openai_api_key (str): OpenAI APIキー。
            settings (AgentSetting | None): 各フェーズのモデル/プロンプト設定。未指定時は既定値。
            tools (list[BaseTool]): 利用可能なツール一覧（LangChain Tool）。
            max_challenge_count (int): 内省に基づくリトライの最大回数。
        """
        self.openai_base_url = openai_base_url
        self.openai_api_key = openai_api_key
        self.settings = settings or AgentSetting()
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}

        # OpenAIクライアントを初期化
        self.client = OpenAI(
            base_url=self.openai_base_url,
            api_key=self.openai_api_key,
        )

        self.max_challenge_count = max_challenge_count
        self.chat_history_max_turns = chat_history_max_turns

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
            answer=result["answer"],
        )

        return agent_result

    def create_graph(self) -> Pregel:
        """エージェントのメイングラフを作成する

        Returns:
            Pregel: エージェントのメイングラフ
        """
        workflow = StateGraph(AgentState)

        # Add the plan node
        workflow.add_node("create_plan", self._create_plan)

        # Add the execution step
        workflow.add_node("execute_subtasks", self._execute_subgraph)

        workflow.add_node("create_answer", self._create_answer)

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

    def _create_plan(self, state: AgentState) -> dict:
        """1. 計画作成｜質問分解とサブタスクリスト作成

        Args:
            state (AgentState): 入力の状態

        Returns:
            AgentState: 更新された状態
        """

        logger.info("🚀 Starting plan generation process...")

        # メッセージ作成
        planner_prompt = self.settings.planner.prompt
        conversation_context = self._format_chat_history(state.get("chat_history", []))
        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": planner_prompt.system_prompt.replace(
                    "{conversation_context}", conversation_context
                ),
            },
            {
                "role": "user",
                "content": planner_prompt.user_prompt.replace(
                    "{query}", str(state["query"])
                ),
            },
        ]

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

    def _select_tools(self, state: AgentSubGraphState) -> dict:
        """2.1 ツール選択｜LLMが適切なツールを判断・選択

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
            subtask_prompt = self.settings.subtask_select_tool.prompt
            messages = [
                {
                    "role": "system",
                    "content": subtask_prompt.system_prompt,
                },
                {
                    "role": "user",
                    "content": subtask_prompt.user_prompt.replace(
                        "{query}", str(state["query"])
                    )
                    .replace("{plan}", str(state["plan"]))
                    .replace("{subtask}", str(state["subtask"])),
                },
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

            retry_prompt = self.settings.subtask_retry_answer.prompt
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

    def _execute_tools(self, state: AgentSubGraphState) -> dict:
        """2.2 ツール実行｜選択したツールを実行。

        select_tools の結果（直前メッセージ）に含まれる `tool_calls` を順に実行し、
        各ツールの戻り値を `ToolResult` として蓄積します。ツール呼び出しが無い場合は
        実行をスキップし、空の結果を返します。

        Args:
            state (AgentSubGraphState): サブタスク実行中の状態（messages を含む）。

        Returns:
            dict: 以下を含む更新済み状態の差分。
                - `messages`: ツール実行結果（toolロール）を追加したメッセージ列
                - `tool_results`: 実行したツール結果（List[List[ToolResult]]] 形式）
        """

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

    def _create_subtask_answer(self, state: AgentSubGraphState) -> dict:
        """2.3 回答生成｜ツール実行結果から回答を作成

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

    def _reflect_subtask(self, state: AgentSubGraphState) -> dict:
        """2.4 自己修正｜回答の適切性評価と原因分析→再試行指示

        Args:
            state (AgentSubGraphState): 入力の状態

        Raises:
            ValueError: reflection resultがNoneの場合

        Returns:
            dict: 更新された状態
        """

        logger.info("🚀 Starting reflection process...")
        messages = state["messages"]

        refl_prompt = self.settings.subtask_reflection.prompt
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

    def _create_answer(self, state: AgentState) -> dict:
        """3. 最終回答作成｜全サブタスク回答を統合

        Args:
            state (AgentState): 入力の状態

        Returns:
            dict: 更新された状態
        """

        logger.info("🚀 Starting final answer creation process...")
        # サブタスク結果のうちタスク内容と回答のみを取得
        subtask_results_seq = state.get("subtask_results", [])
        subtask_results = [
            (result.task_name, result.subtask_answer) for result in subtask_results_seq
        ]
        final_answer_prompt = self.settings.final_answer.prompt
        conversation_context = self._format_chat_history(state.get("chat_history", []))
        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": final_answer_prompt.system_prompt.replace(
                    "{conversation_context}",
                    conversation_context,
                ).replace("{subtask_results}", str(subtask_results)),
            },
            {
                "role": "user",
                "content": final_answer_prompt.user_prompt.replace(
                    "{query}", str(state["query"])
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

        return {"answer": response.choices[0].message.content}

    def _execute_subgraph(self, state: AgentState):
        """単一サブタスクのサブグラフを実行する。

        与えられた `current_step` のサブタスクに対して、
        ツール選択→ツール実行→回答生成→内省（必要に応じてループ）
        を実行し、`Subtask` 結果を1件返します。

        Args:
            state (AgentState): メイングラフの状態（query/plan/current_step など）。

        Returns:
            dict: `subtask_results`（List[Subtask]）を含む差分。
        """
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
        """全サブタスクに並列送信するための分岐を生成する。

        与えられた計画 `plan` の各インデックスに対して、
        `execute_subtasks` へ送る `Send` を生成します。

        Args:
            state (AgentState): メイングラフの状態（plan を含む）。

        Returns:
            list: `Send` オブジェクトのリスト。
        """
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
        """サブタスク内のループ継続/終了を判定する。

        内省結果の `is_completed` が真、または挑戦回数が
        `max_challenge_count` に到達した場合は終了、それ以外は継続。

        Args:
            state (AgentSubGraphState): サブタスク実行中の状態。

        Returns:
            Literal["end", "continue"]: 継続フラグ。
        """
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
        workflow.add_node("select_tools", self._select_tools)

        # ツール実行ノードを追加
        workflow.add_node("execute_tools", self._execute_tools)

        # サブタスク回答作成ノードを追加
        workflow.add_node("create_subtask_answer", self._create_subtask_answer)

        # サブタスク内省ノードを追加
        workflow.add_node("reflect_subtask", self._reflect_subtask)

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

    def _chat_parse(
        self,
        *,
        model: str,
        messages: Iterable[ChatCompletionMessageParam],
        response_format: Type[BaseModel],
        **rest: Any,
    ):
        """構造化出力（parse）でChat Completionsを呼び出すヘルパ。

        Args:
            model (str): モデル名。
            messages (Iterable[ChatCompletionMessageParam]): メッセージ列。
            response_format (Type[BaseModel]): Pydanticモデル型（構造化出力）。
            **rest: 追加パラメータ（temperature 等）。

        Returns:
            Any: OpenAIクライアントのレスポンス。
        """
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
        """通常のChat Completionsを呼び出すヘルパ。

        Args:
            model (str): モデル名。
            messages (Iterable[ChatCompletionMessageParam]): メッセージ列。
            **rest: 追加パラメータ（tools 等）。

        Returns:
            Any: OpenAIクライアントのレスポンス。
        """
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            **rest,
        )

    def _format_chat_history(
        self, chat_history: list[ChatCompletionMessageParam]
    ) -> str:
        """ユーザー/アシスタントの履歴のみを文字列に整形する。

        - roleがuser/assistant以外（system/toolなど）は除外
        - 表示ラベルは日本語化（ユーザー/チャットボット）
        - chat_history_max_turnsが指定されていれば末尾からその件数を採用
        """
        if not chat_history:
            return ""

        filtered = [m for m in chat_history if m.get("role") in ("user", "assistant")]

        # 末尾N件に制限（Noneなら全件）
        if self.chat_history_max_turns is not None and self.chat_history_max_turns > 0:
            filtered = filtered[-self.chat_history_max_turns :]  # NOQA: E203

        role_map = {"user": "ユーザー", "assistant": "チャットボット"}
        lines: list[str] = []
        for m in filtered:
            role = role_map.get(m.get("role", ""), "")
            content = str(m.get("content", "")).strip()
            if role and content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)
