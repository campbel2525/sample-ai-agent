import time
import uuid
from typing import Any, List, Optional

import debugpy
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openai.types.chat import ChatCompletionMessageParam
from pydantic import AliasChoices, BaseModel, Field, model_validator
from ragas.dataset_schema import EvaluationResult

from ai_agents.agents.general_purpose_ai_agent.models import (
    AgentResult,
    AgentSetting,
)
from ai_agents.agents.general_purpose_ai_agent.settings import (
    FINAL_ANSWER_SYSTEM_PROMPT,
    FINAL_ANSWER_USER_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT,
    SUBTASK_REFLECTION_USER_PROMPT,
    SUBTASK_RETRY_ANSWER_USER_PROMPT,
    SUBTASK_TOOL_SELECTION_SYSTEM_PROMPT,
    SUBTASK_TOOL_SELECTION_USER_PROMPT,
)
from ai_agents.tools.hybrid_search_tool import HybridSearchTool
from config.settings import Settings
from services.ai_agent_service import run_ai_agent, run_ai_agent_with_rags

settings = Settings()

debugpy.listen((settings.fastapi_host, settings.debugpy_port))

app = FastAPI(
    title="AI Agents API", description="AI Agents API with FastAPI", version="1.0.0"
)


class AIAgentRequest(BaseModel):
    """
    AI Agentの実行リクエストモデル
    """

    # 基本パラメータ
    query: str = Field(
        ...,
        description="ユーザーの質問",
        examples=["Pythonでファイルを読み込む方法を教えてください"],
    )
    chat_history: list[ChatCompletionMessageParam] = Field(
        default_factory=list,
        description="チャット履歴",
        json_schema_extra={
            "example": [
                {
                    "role": "user",
                    "content": "Pythonについて教えて",
                    "timestamp": "2025-01-17T10:00:00Z",
                },
                {
                    "role": "assistant",
                    "content": "Pythonは汎用プログラミング言語です。",
                    "timestamp": "2025-01-17T10:00:30Z",
                },
            ]
        },
    )

    # AgentSetting: 入力をまとめて受け取るためのコンテナ
    # （個別フィールドも後方互換のため残します）

    ai_agent_setting: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Agentの各種設定（モデル名・パラメータ・プロンプト）をまとめて渡すフィールド。"
            "AgentSetting のネスト形式（planner等のPhaseSettingsごと）またはフラット形式のキーを受け付けます。"
        ),
        json_schema_extra={
            "example": {
                "planner": {
                    "model_name": "gpt-4o-2024-08-06",
                    "model_params": {"seed": 0, "temperature": 0},
                    "prompt": {
                        "system_prompt": "あなたは優秀なプランナーです。ユーザーの質問を分析し、適切なサブタスクに分解してください。",  # noqa: E501
                        "user_prompt": "質問: {query}\n\n上記の質問に答えるために必要なサブタスクを作成してください。",  # noqa: E501
                    },
                },
                "subtask_select_tool": {
                    "model_name": "gpt-4o-2024-08-06",
                    "model_params": {"seed": 0, "temperature": 0},
                    "prompt": {
                        "system_prompt": "あなたは与えられたサブタスクを実行する専門家です。利用可能なツールを使用してタスクを完了してください。",  # noqa: E501
                        "user_prompt": "サブタスク: {subtask}\n\n上記のサブタスクを実行するために最適なツールを選択し、実行してください。",  # noqa: E501
                    },
                },
                "subtask_reflection": {
                    "model_params": {"seed": 0, "temperature": 0},
                    "model_name": "gpt-4o-2024-08-06",
                    "prompt": {
                        "user_prompt": "サブタスク: {subtask}\nツール実行結果: {tool_result}\n\n上記の結果がサブタスクの要求を満たしているか評価してください。",  # noqa: E501
                    },
                },
                "subtask_retry_answer": {
                    "model_name": "gpt-4o-2024-08-06",
                    "model_params": {"seed": 0, "temperature": 0},
                    "prompt": {
                        "user_prompt": "前回の試行が不十分でした。アドバイス: {advice}\n\n改善されたアプローチでサブタスクを再実行してください。",  # noqa: E501
                    },
                },
                "final_answer": {
                    "model_name": "gpt-4o-2024-08-06",
                    "model_params": {"seed": 0, "temperature": 0},
                    "prompt": {
                        "system_prompt": "あなたは全てのサブタスクの結果を統合し、ユーザーの質問に対する最終的な回答を作成する専門家です。",  # noqa: E501
                        "user_prompt": "質問: {query}\nサブタスク結果: {subtask_results}\n\n上記の情報を基に、質問に対する包括的で分かりやすい回答を作成してください。",  # noqa: E501
                    },
                },
            }
        },
    )

    # AgentSetting は ai_agent_setting（ネスト/フラット辞書）で受け付けます。

    # RAGas 設定（入れ子）：
    # { dataset: { reference: str }, metrics: List[str] }
    # JSONキーは "ragas_setting"（推奨）。互換のため "ragas_settging" も受理。
    ragas_setting: Optional[dict[str, Any]] = Field(
        default=None,
        validation_alias=AliasChoices("ragas_setting", "ragas_settging"),
        description=(
            "RAGas設定（入れ子）。{ dataset: { reference: str }, metrics: List[str] }"
        ),
        json_schema_extra={
            "example": {
                "dataset": {
                    "reference": (
                        "Pythonでファイルを読み込むには、open()関数を使用し、"
                        "with文と組み合わせることが推奨されます。"
                    )
                },
                "metrics": ["answer_relevancy", "answer_similarity"],
            }
        },
    )

    # RAGas評価制御パラメータ
    is_run_ragas: bool = Field(
        default=False,
        description="RAGas評価を実行するかどうか",
        examples=[False],
    )

    # 以前のトップレベル ragas_reference / ragas_metrics は廃止（入れ子に統一）

    @model_validator(mode="after")
    def validate_ragas_fields(self):
        """RAGas関連フィールドのバリデーション"""
        if self.is_run_ragas:
            # ragas_setting.dataset.reference が必須
            ragas_ref = None
            if self.ragas_setting and isinstance(self.ragas_setting, dict):
                dataset = self.ragas_setting.get("dataset") or {}
                if isinstance(dataset, dict):
                    ragas_ref = dataset.get("reference")
            if not ragas_ref:
                raise ValueError(
                    "is_run_ragas=Trueの時、ragas_setting.dataset.reference は必須です"
                )
        return self


class SubtaskDetail(BaseModel):
    """
    サブタスクの詳細情報
    """

    task_name: str = Field(
        ...,
        description="サブタスクの名前",
        examples=["Pythonファイル読み込み方法の調査"],
    )
    is_completed: bool = Field(
        ...,
        description="サブタスクが完了しているかどうか",
        examples=[True],
    )
    subtask_answer: str = Field(
        ...,
        description="サブタスクの回答",
        examples=[
            "Pythonでファイルを読み込むには、open()関数を使用します。基本的な構文は `with open('filename.txt', 'r') as file: content = file.read()` です。"  # noqa: E501
        ],  # noqa: E501
    )
    challenge_count: int = Field(
        ...,
        description="サブタスクの挑戦回数",
        examples=[1],
    )
    tool_results_count: int = Field(
        ...,
        description="使用されたツールの実行回数",
        examples=[2],
    )
    reflection_count: int = Field(
        ...,
        description="リフレクション実行回数",
        examples=[1],
    )


class PromptData(BaseModel):
    """
    プロンプトデータ
    """

    planner_system_prompt: str = Field(
        ...,
        description="プランナーシステムプロンプト",
    )
    planner_user_prompt: str = Field(
        ...,
        description="プランナーユーザープロンプト",
    )
    subtask_select_tool_system_prompt: str = Field(
        ...,
        description="サブタスクシステムプロンプト",
    )
    subtask_select_tool_user_prompt: str = Field(
        ...,
        description="サブタスクツール選択ユーザープロンプト",
    )
    subtask_reflection_user_prompt: str = Field(
        ...,
        description="サブタスクリフレクションユーザープロンプト",
    )
    subtask_retry_answer_user_prompt: str = Field(
        ...,
        description="サブタスクリトライ回答ユーザープロンプト",
    )
    final_answer_system_prompt: str = Field(
        ...,
        description="最終回答作成システムプロンプト",
    )
    final_answer_user_prompt: str = Field(
        ...,
        description="最終回答作成ユーザープロンプト",
    )


class AIAgentResult(BaseModel):
    """
    AI Agentの実行結果
    """

    prompt: PromptData = Field(..., description="使用されたプロンプト")

    plan: List[str] = Field(
        ...,
        description="実行計画",
        examples=[
            "Pythonファイル読み込み方法の調査",
            "基本的なopen()関数の使用方法の説明",
            "with文を使った安全なファイル処理の説明",
            "エラーハンドリングの方法の説明",
        ],
    )
    subtasks_detail: List[SubtaskDetail] = Field(
        ...,
        description="サブタスクの詳細情報",
        examples=[
            {
                "task_name": "Pythonファイル読み込み方法の調査",
                "is_completed": True,
                "subtask_answer": "Pythonでファイルを読み込むには、open()関数を使用します。基本的な構文は `with open('filename.txt', 'r') as file: content = file.read()` です。",  # noqa: E501
                "challenge_count": 1,
                "tool_results_count": 2,
                "reflection_count": 1,
            }
        ],
    )
    total_subtasks: int = Field(
        ...,
        description="総サブタスク数",
        examples=[4],
    )
    completed_subtasks: int = Field(
        ...,
        description="完了したサブタスク数",
        examples=[4],
    )
    total_challenge_count: int = Field(
        ...,
        description="全サブタスクの総挑戦回数",
        examples=[5],
    )


class RagasInput(BaseModel):
    """
    RAGas評価の入力データ
    """

    # ragas_retrieved_contexts: Optional[List[str]] = Field(
    #     default=None,
    #     description="検索された文脈",
    # )
    ragas_reference: Optional[str] = Field(
        default=None,
        description="正しい回答",
    )


class RagasResult(BaseModel):
    """
    RAGas評価結果
    """

    scores: dict = Field(
        ...,
        description="RAGas評価スコア",
        examples=[
            {
                "answer_relevancy": 0.95,
                "semantic_similarity": 0.88,
            }
        ],
    )

    input: RagasInput = Field(
        ...,
        description="RAGas評価の入力データ",
    )


class AIAgentResponse(BaseModel):
    """
    AI Agentの実行レスポンスモデル
    """

    query: str = Field(
        ...,
        description="処理した質問",
        examples=["Pythonでファイルを読み込む方法を教えてください"],
    )
    answer: str = Field(
        ...,
        description="AIエージェントの回答",
        examples=[
            "Pythonでファイルを読み込むには、主に以下の方法があります：\n\n1. **基本的な方法**：\n```python\nwith open('filename.txt', 'r', encoding='utf-8') as file:\n    content = file.read()\n```\n\n2. **行ごとに読み込む方法**：\n```python\nwith open('filename.txt', 'r', encoding='utf-8') as file:\n    for line in file:\n        print(line.strip())\n```\n\nwith文を使用することで、ファイルが自動的に閉じられ、メモリリークを防ぐことができます。"  # noqa: E501
        ],  # noqa: E501
    )

    ai_agent_result: AIAgentResult = Field(
        ...,
        description="AI Agentの実行結果",
    )

    ragas_result: RagasResult = Field(
        ...,
        description="RAGas評価結果",
    )

    # Langfuse情報
    langfuse_session_id: str = Field(
        ...,
        description="LangfuseセッションID",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )

    # 実行情報
    execution_time: Optional[float] = Field(
        default=None,
        description="実行時間（秒）",
        examples=[12.34],
    )
    error: Optional[str] = Field(
        default=None,
        description="エラーメッセージ",
        examples=[None],
    )


@app.post(
    "/ai_agents/chatbot/exec",
    response_model=AIAgentResponse,
    responses={
        200: {
            "description": "AI Agentの実行が成功した場合",
            "content": {
                "application/json": {
                    "example": {
                        "query": "Pythonでファイルを読み込む方法を教えてください",
                        "answer": "Pythonでファイルを読み込むには、主に以下の方法があります：\n\n1. **基本的な方法**：\n```python\nwith open('filename.txt', 'r', encoding='utf-8') as file:\n    content = file.read()\n```\n\n2. **行ごとに読み込む方法**：\n```python\nwith open('filename.txt', 'r', encoding='utf-8') as file:\n    for line in file:\n        print(line.strip())\n```\n\nwith文を使用することで、ファイルが自動的に閉じられ、メモリリークを防ぐことができます。",  # noqa: E501
                        "ai_agent_result": {
                            "prompt": {
                                "planner_system_prompt": "あなたは優秀なプランナーです。ユーザーの質問を分析し、適切なサブタスクに分解してください。",  # noqa: E501
                                "planner_user_prompt": "質問: {query}\n\n上記の質問に答えるために必要なサブタスクを作成してください。",  # noqa: E501
                                "subtask_select_tool_system_prompt": "あなたは与えられたサブタスクを実行する専門家です。利用可能なツールを使用してタスクを完了してください。",  # noqa: E501
                                "subtask_select_tool_user_prompt": "サブタスク: {subtask}\n\n上記のサブタスクを実行するために最適なツールを選択し、実行してください。",  # noqa: E501
                                "subtask_reflection_user_prompt": "サブタスク: {subtask}\nツール実行結果: {tool_result}\n\n上記の結果がサブタスクの要求を満たしているか評価してください。",  # noqa: E501
                                "subtask_retry_answer_user_prompt": "前回の試行が不十分でした。アドバイス: {advice}\n\n改善されたアプローチでサブタスクを再実行してください。",  # noqa: E501
                                "final_answer_system_prompt": "あなたは全てのサブタスクの結果を統合し、ユーザーの質問に対する最終的な回答を作成する専門家です。",  # noqa: E501
                                "final_answer_user_prompt": "質問: {query}\nサブタスク結果: {subtask_results}\n\n上記の情報を基に、質問に対する包括的で分かりやすい回答を作成してください。",  # noqa: E501
                            },
                            "plan": [
                                "Pythonファイル読み込み方法の調査",
                                "基本的なopen()関数の使用方法の説明",
                                "with文を使った安全なファイル処理の説明",
                                "エラーハンドリングの方法の説明",
                            ],
                            "subtasks_detail": [
                                {
                                    "task_name": "Pythonファイル読み込み方法の調査",
                                    "is_completed": True,
                                    "subtask_answer": "Pythonでファイルを読み込むには、open()関数を使用します。基本的な構文は `with open('filename.txt', 'r') as file: content = file.read()` です。",  # noqa: E501
                                    "challenge_count": 1,
                                    "tool_results_count": 2,
                                    "reflection_count": 1,
                                }
                            ],
                            "total_subtasks": 4,
                            "completed_subtasks": 4,
                            "total_challenge_count": 5,
                        },
                        "ragas_result": {
                            "scores": {
                                "answer_relevancy": 0.95,
                                "semantic_similarity": 0.88,
                            },
                            "input": {
                                # "ragas_retrieved_contexts": [
                                #     "Pythonでファイルを読み込むには、open()関数を使用します。",
                                #     "with文を使用することで、ファイルを自動的に閉じることができます。",
                                # ],
                                "ragas_reference": "Pythonでファイルを読み込むには、open()関数を使用し、with文と組み合わせることが推奨されます。",  # noqa: E501
                            },
                        },
                        "langfuse_session_id": "550e8400-e29b-41d4-a716-446655440000",
                        "execution_time": 12.34,
                        "error": None,
                    }
                }
            },
        },
        422: {
            "description": "バリデーションエラー",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "type": "missing",
                                "loc": ["body", "query"],
                                "msg": "Field required",
                                "input": {},
                            },
                            {
                                "type": "value_error",
                                "loc": ["body"],
                                "msg": "Value error, is_run_ragas=Trueの時、ragas_referenceは必須です",  # noqa: E501
                                "input": {
                                    "query": "テスト質問",
                                    "is_run_ragas": True,
                                    "ragas_reference": None,
                                },
                            },
                        ]
                    }
                }
            },
        },
        500: {
            "description": "サーバーエラー",
            "content": {
                "application/json": {
                    "example": {"message": "OpenAI API connection failed"}
                }
            },
        },
    },
)
async def exec_chatbot_ai_agent(
    request: AIAgentRequest,
) -> AIAgentResponse | JSONResponse:
    """
    AI Agentの実行エンドポイント
    リクエストパラメータに基づいてAIエージェントを実行し、RAGas評価も行います

    処理の流れ:
    1. 実行時間計測開始とLangfuseセッションID生成
    2. AIエージェントの設定（プランナー、サブタスク、最終回答作成用, LLMの設定）
    3. ツールの準備（HybridSearchTool）
    4. AIエージェントの実行（RAGas評価の有無に応じて分岐）
    5. レスポンスの作成 (実行結果の詳細情報構築(サブタスク詳細、統計情報), RAGasスコアのまとめ)

    ## レスポンス例

    ### 成功時 (200)
    正常にAI Agentが実行され、回答が生成された場合のレスポンス

    ### バリデーションエラー (422)
    リクエストパラメータが不正な場合のエラーレスポンス
    - 必須フィールドが不足している場合
    - RAGas評価関連のバリデーションエラー

    ### サーバーエラー (500)
    AI Agent実行中にエラーが発生した場合のレスポンス
    - OpenAI API接続エラー
    - その他の予期しないエラー
    """
    # 1. 実行時間計測開始とLangfuseセッションID生成
    start_time = time.time()

    # LangfuseセッションIDを生成
    langfuse_session_id = str(uuid.uuid4())

    try:
        # 2. AIエージェントの設定（プランナー、サブタスク、最終回答作成用, LLMの設定）
        # 優先: まとめ入力 ai_agent_setting があれば利用
        if request.ai_agent_setting is not None:
            payload = {
                k: v for k, v in request.ai_agent_setting.items() if v is not None
            }
            ai_agent_setting = AgentSetting(**payload)
        else:
            # 未指定時はデフォルト構成で初期化
            ai_agent_setting = AgentSetting()

        # 3. ツールの準備（HybridSearchTool）
        hybrid_search_tool = HybridSearchTool(
            openai_api_key=settings.openai_api_key,
            openai_base_url=settings.openai_base_url,
            openai_embedding_model=settings.openai_embedding_model,
            openai_max_retries=3,
            opensearch_base_url=settings.opensearch_base_url,
            opensearch_index_name=settings.opensearch_default_index_name,
        )
        ai_agent_tools = [hybrid_search_tool]

        # 4. AIエージェントの実行（RAGas評価の有無に応じて分岐）
        if request.is_run_ragas:

            ragas_dataset_data = None
            ragas_metrics_data = None
            if request.ragas_setting and isinstance(request.ragas_setting, dict):
                ragas_dataset_data = (
                    request.ragas_setting.get("dataset")
                    if isinstance(request.ragas_setting.get("dataset"), dict)
                    else None
                )
                ragas_metrics_data = request.ragas_setting.get("metrics")

            agent_result, ragas_scores = run_ai_agent_with_rags(
                query=request.query,
                chat_history=request.chat_history,
                ai_agent_setting=ai_agent_setting,
                ai_agent_tools=ai_agent_tools,
                langfuse_session_id=langfuse_session_id,
                ragas_dataset_data=ragas_dataset_data,
                ragas_metrics_data=ragas_metrics_data,
            )
        else:
            agent_result = run_ai_agent(
                query=request.query,
                chat_history=request.chat_history,
                ai_agent_setting=ai_agent_setting,
                ai_agent_tools=ai_agent_tools,
                langfuse_session_id=langfuse_session_id,
            )
            ragas_scores = None

        # 5. レスポンスの作成 (実行結果の詳細情報構築(サブタスク詳細、統計情報), RAGasスコアのまとめ)
        execution_time = time.time() - start_time
        return get_response(
            request,
            agent_result,
            ragas_scores,
            langfuse_session_id,
            execution_time,
        )

    except Exception as e:
        print(f"Error during AI Agent execution: {e}")
        # 期待仕様: { "message": e.message } を返却（Python3では str(e) を使用）
        return JSONResponse(status_code=500, content={"message": str(e)})


def normalize_ragas_scores(raw: Any) -> dict:
    """
    RAGasの出力を必ずdictに正規化する。
    """
    # dictそのまま
    if isinstance(raw, dict):
        return raw

    # .scoresを持つオブジェクト
    if hasattr(raw, "scores"):
        return normalize_ragas_scores(getattr(raw, "scores"))

    # listの場合
    if isinstance(raw, list):
        # list[dict]
        if raw and isinstance(raw[0], dict):
            if len(raw) == 1:
                return dict(raw[0])
            merged = {}
            for d in raw:
                merged.update(d)
            return merged

        # list[object]で metric/score を持つケース
        collected = {}
        for item in raw:
            metric = getattr(getattr(item, "metric", None), "name", None) or getattr(
                item, "name", None
            )
            value = getattr(item, "score", None) or getattr(item, "value", None)
            if metric and isinstance(value, (int, float)):
                collected[metric] = float(value)
        if collected:
            return collected

    # それ以外 → 空dict
    return {}


def get_response(
    request: AIAgentRequest,
    agent_result: AgentResult,
    ragas_scores: Optional[EvaluationResult],
    langfuse_session_id: str,
    execution_time: float,
):
    # レスポンスの作成 実行結果の詳細情報構築（サブタスク詳細、統計情報）
    subtasks_detail = []
    total_challenge_count = 0
    completed_subtasks = 0
    for subtask in agent_result.subtasks:
        # ツール実行回数を計算
        tool_results_count = sum(
            len(tool_result_list) for tool_result_list in subtask.tool_results
        )

        subtasks_detail.append(
            SubtaskDetail(
                task_name=subtask.task_name,
                is_completed=subtask.is_completed,
                subtask_answer=subtask.subtask_answer,
                challenge_count=subtask.challenge_count,
                tool_results_count=tool_results_count,
                reflection_count=len(subtask.reflection_results),
            )
        )

        total_challenge_count += subtask.challenge_count
        if subtask.is_completed:
            completed_subtasks += 1

    # レスポンスの作成 RAGasスコアの辞書形式変換
    ragas_scores_dict = {}
    if request.is_run_ragas and ragas_scores is not None:
        ragas_scores_dict = normalize_ragas_scores(ragas_scores)

    # レスポンスの作成 レスポンスモデルの構築と返却
    # レスポンスに載せるプロンプトは、ai_agent_setting（ネスト）を優先
    aset = request.ai_agent_setting or {}

    def pick(d: dict, path: list[str]) -> Optional[str]:
        cur: Any = d
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
        return cur if isinstance(cur, str) else None

    prompt_data = PromptData(
        planner_system_prompt=(
            pick(aset, ["planner", "prompt", "system_prompt"]) or PLANNER_SYSTEM_PROMPT
        ),
        planner_user_prompt=(
            pick(aset, ["planner", "prompt", "user_prompt"]) or PLANNER_USER_PROMPT
        ),
        subtask_select_tool_system_prompt=(
            pick(aset, ["subtask_select_tool", "prompt", "system_prompt"])  # noqa: E501
            or SUBTASK_TOOL_SELECTION_SYSTEM_PROMPT
        ),
        subtask_select_tool_user_prompt=(
            pick(aset, ["subtask_select_tool", "prompt", "user_prompt"])  # noqa: E501
            or SUBTASK_TOOL_SELECTION_USER_PROMPT
        ),
        subtask_reflection_user_prompt=(
            pick(aset, ["subtask_reflection", "prompt", "user_prompt"])  # noqa: E501
            or SUBTASK_REFLECTION_USER_PROMPT
        ),
        subtask_retry_answer_user_prompt=(
            pick(aset, ["subtask_retry_answer", "prompt", "user_prompt"])  # noqa: E501
            or SUBTASK_RETRY_ANSWER_USER_PROMPT
        ),
        final_answer_system_prompt=(
            pick(aset, ["final_answer", "prompt", "system_prompt"])
            or FINAL_ANSWER_SYSTEM_PROMPT  # noqa: E501
        ),
        final_answer_user_prompt=(
            pick(aset, ["final_answer", "prompt", "user_prompt"])
            or FINAL_ANSWER_USER_PROMPT  # noqa: E501
        ),
    )
    ai_agent_result = AIAgentResult(
        prompt=prompt_data,
        plan=agent_result.plan.subtasks,
        subtasks_detail=subtasks_detail,
        total_subtasks=len(agent_result.subtasks),
        completed_subtasks=completed_subtasks,
        total_challenge_count=total_challenge_count,
    )
    # ragas_reference は ragas_setting.dataset.reference
    ragas_ref = None
    if request.ragas_setting and isinstance(request.ragas_setting, dict):
        dataset = request.ragas_setting.get("dataset") or {}
        if isinstance(dataset, dict):
            ragas_ref = dataset.get("reference")

    ragas_input = RagasInput(ragas_reference=ragas_ref)
    ragas_result = RagasResult(
        scores=ragas_scores_dict,
        input=ragas_input,
    )
    return AIAgentResponse(
        query=request.query,
        answer=agent_result.answer,
        ai_agent_result=ai_agent_result,
        ragas_result=ragas_result,
        langfuse_session_id=langfuse_session_id,
        execution_time=execution_time,
    )
