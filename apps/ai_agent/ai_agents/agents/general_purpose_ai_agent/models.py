from typing import Any, Mapping, Generic, TypeVar

from pydantic import BaseModel, Field, model_validator
from pydantic.generics import GenericModel

from .settings import (
    FINAL_ANSWER_MODEL_NAME,
    FINAL_ANSWER_MODEL_PARAMS,
    FINAL_ANSWER_SYSTEM_PROMPT,
    FINAL_ANSWER_USER_PROMPT,
    PLANNER_MODEL_NAME,
    PLANNER_MODEL_PARAMS,
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT,
    SUBTASK_REFLECTION_MODEL_NAME,
    SUBTASK_REFLECTION_MODEL_PARAMS,
    SUBTASK_REFLECTION_USER_PROMPT,
    SUBTASK_RETRY_ANSWER_MODEL_NAME,
    SUBTASK_RETRY_ANSWER_MODEL_PARAMS,
    SUBTASK_RETRY_ANSWER_USER_PROMPT,
    SUBTASK_TOOL_SELECTION_MODEL_NAME,
    SUBTASK_TOOL_SELECTION_MODEL_PARAMS,
    SUBTASK_TOOL_SELECTION_SYSTEM_PROMPT,
    SUBTASK_TOOL_SELECTION_USER_PROMPT,
)

"""
エージェントの各フェーズの戻り値
"""


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
    query: str = Field(..., description="ユーザーの元の質問")
    plan: Plan = Field(..., description="エージェントの計画")
    subtasks: list[Subtask] = Field(..., description="サブタスクのリスト")
    answer: str = Field(..., description="最終的な回答")


"""
エージェントの各フェーズ設定モデル
"""


class PromptBase(BaseModel):
    pass


class PromptSystemUser(PromptBase):
    system_prompt: str
    user_prompt: str

    @model_validator(mode="after")
    def _require_both(self) -> "PromptSystemUser":
        if not self.system_prompt or not self.user_prompt:
            raise ValueError("system_prompt と user_prompt は必須です")
        return self

    pass


class PromptUserOnly(PromptBase):
    user_prompt: str

    @model_validator(mode="after")
    def _require_user(self) -> "PromptUserOnly":
        if not self.user_prompt:
            raise ValueError("user_prompt は必須です")
        return self

    pass


TPrompt = TypeVar("TPrompt", bound=PromptBase)


class PhaseSettings(GenericModel, Generic[TPrompt]):
    prompt: TPrompt
    model_name: str = ""
    model_params: Mapping[str, Any] = Field(
        default_factory=lambda: {"temperature": 0, "seed": 0}
    )


class AgentSettings(BaseModel):
    planner: PhaseSettings[PromptSystemUser]
    subtask_select_tool: PhaseSettings[PromptSystemUser]
    subtask_reflection: PhaseSettings[PromptUserOnly]
    subtask_retry_answer: PhaseSettings[PromptUserOnly]
    final_answer: PhaseSettings[PromptSystemUser]

    def __init__(
        self,
        # planner
        planner_model_name: str | None = None,
        planner_model_params: Mapping[str, Any] | None = None,
        planner_system_prompt: str | None = None,
        planner_user_prompt: str | None = None,
        # subtask_tool_selection
        subtask_tool_selection_model_name: str | None = None,
        subtask_tool_selection_model_params: Mapping[str, Any] | None = None,
        subtask_tool_selection_system_prompt: str | None = None,
        subtask_tool_selection_user_prompt: str | None = None,
        # subtask_reflection
        subtask_reflection_model_params: Mapping[str, Any] | None = None,
        subtask_reflection_user_prompt: str | None = None,
        subtask_reflection_model_name: str | None = None,
        # subtask_retry_answer
        subtask_retry_answer_model_name: str | None = None,
        subtask_retry_answer_model_params: Mapping[str, Any] | None = None,
        subtask_retry_answer_user_prompt: str | None = None,
        # final_answer
        final_answer_model_name: str | None = None,
        final_answer_model_params: Mapping[str, Any] | None = None,
        final_answer_system_prompt: str | None = None,
        final_answer_user_prompt: str | None = None,
    ):
        # planner
        planner_phase = PhaseSettings(
            model_name=(
                PLANNER_MODEL_NAME if planner_model_name is None else planner_model_name
            ),
            model_params=(
                dict(PLANNER_MODEL_PARAMS)
                if planner_model_params is None
                else dict(planner_model_params)
            ),
            prompt=PromptSystemUser(
                system_prompt=(
                    PLANNER_SYSTEM_PROMPT
                    if planner_system_prompt is None
                    else planner_system_prompt
                ),
                user_prompt=(
                    PLANNER_USER_PROMPT
                    if planner_user_prompt is None
                    else planner_user_prompt
                ),
            ),
        )
        # subtask_tool_selection
        subtask_select_tool_phase = PhaseSettings(
            model_name=(
                SUBTASK_TOOL_SELECTION_MODEL_NAME
                if subtask_tool_selection_model_name is None
                else subtask_tool_selection_model_name
            ),
            model_params=(
                dict(SUBTASK_TOOL_SELECTION_MODEL_PARAMS)
                if subtask_tool_selection_model_params is None
                else dict(subtask_tool_selection_model_params)
            ),
            prompt=PromptSystemUser(
                system_prompt=(
                    SUBTASK_TOOL_SELECTION_SYSTEM_PROMPT
                    if subtask_tool_selection_system_prompt is None
                    else subtask_tool_selection_system_prompt
                ),
                user_prompt=(
                    SUBTASK_TOOL_SELECTION_USER_PROMPT
                    if subtask_tool_selection_user_prompt is None
                    else subtask_tool_selection_user_prompt
                ),
            ),
        )
        # subtask_reflection
        subtask_reflection_phase = PhaseSettings(
            model_name=(
                SUBTASK_REFLECTION_MODEL_NAME
                if subtask_reflection_model_name is None
                else subtask_reflection_model_name
            ),
            model_params=(
                dict(SUBTASK_REFLECTION_MODEL_PARAMS)
                if subtask_reflection_model_params is None
                else dict(subtask_reflection_model_params)
            ),
            prompt=PromptUserOnly(
                user_prompt=(
                    SUBTASK_REFLECTION_USER_PROMPT
                    if subtask_reflection_user_prompt is None
                    else subtask_reflection_user_prompt
                ),
            ),
        )
        # subtask_retry_answer（モデルは subtask_answer を既定として流用する例）
        subtask_retry_answer_phase = PhaseSettings(
            model_name=(
                SUBTASK_RETRY_ANSWER_MODEL_NAME
                if subtask_retry_answer_model_name is None
                else subtask_retry_answer_model_name
            ),
            model_params=(
                dict(SUBTASK_RETRY_ANSWER_MODEL_PARAMS)
                if subtask_retry_answer_model_params is None
                else dict(subtask_retry_answer_model_params)
            ),
            prompt=PromptUserOnly(
                user_prompt=(
                    SUBTASK_RETRY_ANSWER_USER_PROMPT
                    if subtask_retry_answer_user_prompt is None
                    else subtask_retry_answer_user_prompt
                ),
            ),
        )
        # final_answer
        final_answer_phase = PhaseSettings(
            model_name=(
                FINAL_ANSWER_MODEL_NAME
                if final_answer_model_name is None
                else final_answer_model_name
            ),
            model_params=(
                dict(FINAL_ANSWER_MODEL_PARAMS)
                if final_answer_model_params is None
                else dict(final_answer_model_params)
            ),
            prompt=PromptSystemUser(
                system_prompt=(
                    FINAL_ANSWER_SYSTEM_PROMPT
                    if final_answer_system_prompt is None
                    else final_answer_system_prompt
                ),
                user_prompt=(
                    FINAL_ANSWER_USER_PROMPT
                    if final_answer_user_prompt is None
                    else final_answer_user_prompt
                ),
            ),
        )

        # BaseModel の初期化を正しく行い、
        # Pydantic v2 の内部属性（model_fields_set など）をセットする
        super().__init__(
            planner=planner_phase,
            subtask_select_tool=subtask_select_tool_phase,
            subtask_reflection=subtask_reflection_phase,
            subtask_retry_answer=subtask_retry_answer_phase,
            final_answer=final_answer_phase,
        )
