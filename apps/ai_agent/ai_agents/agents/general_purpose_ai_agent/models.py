from typing import Any, Mapping

from pydantic import BaseModel, Field

DEFAULT_MODEL_NAME = "gpt-4o-2024-08-06"
DEFAULT_MODEL_PARAM: dict[str, Any] = {
    "temperature": 0,
    "seed": 0,
}


def default_params() -> dict[str, Any]:
    # 共有参照を避けるため毎回コピー
    return dict(DEFAULT_MODEL_PARAM)


class LLMConfig(BaseModel):
    """OpenAIモデルの設定（1フェーズぶん）"""

    model_name: str = Field(
        default=DEFAULT_MODEL_NAME,
        description="使用するOpenAIモデル名",
    )
    params: Mapping[str, Any] = Field(
        default_factory=default_params,
        description="OpenAIモデルの生成パラメータ（任意キー可）",
    )


class LLMPhaseConfigs(BaseModel):
    """各フェーズごとのOpenAIモデル + 生成パラメータ"""

    planner: LLMConfig = Field(default_factory=LLMConfig)
    subtask_tool_selection: LLMConfig = Field(default_factory=LLMConfig)
    subtask_answer: LLMConfig = Field(default_factory=LLMConfig)
    subtask_reflection: LLMConfig = Field(default_factory=LLMConfig)
    create_last_answer: LLMConfig = Field(default_factory=LLMConfig)


class Plan(BaseModel):
    subtasks: list[str] = Field(..., description="問題を解決するためのサブタスクリスト")


class ToolResult(BaseModel):
    tool_name: str = Field(..., description="ツールの名前")
    args: Any = Field(..., description="ツールの引数")
    results: Any = Field(..., description="ツールの結果")


class ReflectionResult(BaseModel):
    advice: str = Field(
        ...,
        description="評価がNGの場合は、別のツールを試す、別の文言でツールを試すなど、なぜNGなのかとどうしたら改善できるかを考えアドバイスを作成してください。\
アドバイスの内容は過去のアドバイスと計画内の他のサブタスクと重複しないようにしてください。\
アドバイスの内容をもとにツール選択・実行からやり直します。",
    )
    is_completed: bool = Field(
        ...,
        description="ツールの実行結果と回答から、サブタスクに対して正しく回答できているかの評価結果",
    )


class Subtask(BaseModel):
    task_name: str = Field(..., description="サブタスクの名前")
    tool_results: list[list[ToolResult]] = Field(..., description="サブタスクの結果")
    reflection_results: list[ReflectionResult] = Field(
        ..., description="サブタスクの評価結果"
    )
    is_completed: bool = Field(..., description="サブタスクが完了しているかどうか")
    subtask_answer: str = Field(..., description="サブタスクの回答")
    challenge_count: int = Field(..., description="サブタスクの挑戦回数")


class AgentResult(BaseModel):
    question: str = Field(..., description="ユーザーの元の質問")
    plan: Plan = Field(..., description="エージェントの計画")
    subtasks: list[Subtask] = Field(..., description="サブタスクのリスト")
    answer: str = Field(..., description="最終的な回答")
