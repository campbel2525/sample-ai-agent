import os
import json
import time
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# =============================
# 定数（上部に集約）
# =============================
# FastAPIの固定URL（環境変数は無視して固定）
DEFAULT_FASTAPI_BASE_URL = "http://ai-agent:8000"

# RAGasの既定はOFF
DEFAULT_RUN_RAGAS = False

# モデル名の既定値（未入力時はAPI側の既定でも動くが、UIでは明示）
DEFAULT_PLANNER_MODEL = "gpt-4o-2024-08-06"
DEFAULT_SUBTASK_TOOL_SELECTION_MODEL = "gpt-4o-2024-08-06"
DEFAULT_SUBTASK_ANSWER_MODEL = "gpt-4o-2024-08-06"
DEFAULT_SUBTASK_REFLECTION_MODEL = "gpt-4o-2024-08-06"
DEFAULT_FINAL_ANSWER_MODEL = "gpt-4o-2024-08-06"

# UIの高さ・タイムアウト
PROMPT_TEXTAREA_HEIGHT = 120
PARAMS_TEXTAREA_HEIGHT = 80
REQUEST_TIMEOUT_SEC = 600

# APIエンドポイント
EXEC_ENDPOINT = "/ai_agents/chatbot/exec"


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, str]] = []  # {role, content}
    if "last_request" not in st.session_state:
        st.session_state.last_request: Optional[Dict[str, Any]] = None
    if "last_response" not in st.session_state:
        st.session_state.last_response: Optional[Dict[str, Any]] = None
    if "turns" not in st.session_state:
        # 各ターンの詳細（user発話・assistant応答・詳細結果）を保持
        st.session_state.turns: List[Dict[str, Any]] = []


def to_chat_history(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    # FastAPIのChatCompletionMessageParamに合わせた最低限の形式
    out = []
    for m in messages:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            out.append({"role": m["role"], "content": m["content"]})
    return out


def parse_json_or_none(label: str, raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        data = json.loads(raw)
        if data is None:
            return None
        if not isinstance(data, dict):
            st.warning(f"{label} はオブジェクト(JSON)で指定してください。無視します。")
            return None
        return data
    except Exception as e:
        st.error(f"{label} のJSONが不正です: {e}")
        return None


def main():
    st.set_page_config(page_title="AI Agent Chat UI", page_icon="🤖", layout="wide")
    init_state()

    # 固定URL（入力欄は廃止）
    with st.sidebar:
        # 参考リンク（FastAPIのAPI仕様）
        st.caption("詳しくは http://localhost:8000/docs 参照")

        st.subheader("RAGas")
        is_run_ragas = st.checkbox("RAGasを実行する", value=DEFAULT_RUN_RAGAS)
        ragas_reference = st.text_area("RAGas reference", value="")

        st.subheader("モデル設定 (未入力はAPI既定)")
        planner_model_name = st.text_input(
            "planner_model_name", value=DEFAULT_PLANNER_MODEL
        )
        subtask_tool_selection_model_name = st.text_input(
            "subtask_tool_selection_model_name",
            value=DEFAULT_SUBTASK_TOOL_SELECTION_MODEL,
        )
        subtask_answer_model_name = st.text_input(
            "subtask_answer_model_name", value=DEFAULT_SUBTASK_ANSWER_MODEL
        )
        subtask_reflection_model_name = st.text_input(
            "subtask_reflection_model_name", value=DEFAULT_SUBTASK_REFLECTION_MODEL
        )
        final_answer_model_name = st.text_input(
            "final_answer_model_name", value=DEFAULT_FINAL_ANSWER_MODEL
        )

        st.subheader("モデルパラメータ(JSON) (未入力はNone)")
        planner_params_raw = st.text_area(
            "planner_params", height=PARAMS_TEXTAREA_HEIGHT
        )
        subtask_tool_selection_params_raw = st.text_area(
            "subtask_tool_selection_params", height=PARAMS_TEXTAREA_HEIGHT
        )
        subtask_answer_params_raw = st.text_area(
            "subtask_answer_params", height=PARAMS_TEXTAREA_HEIGHT
        )
        subtask_reflection_params_raw = st.text_area(
            "subtask_reflection_params", height=PARAMS_TEXTAREA_HEIGHT
        )
        final_answer_params_raw = st.text_area(
            "final_answer_params", height=PARAMS_TEXTAREA_HEIGHT
        )

        st.subheader("プロンプト上書き (未入力は既定)")
        with st.expander("Planner prompts"):
            ai_agent_planner_system_prompt = st.text_area(
                "ai_agent_planner_system_prompt", height=PROMPT_TEXTAREA_HEIGHT
            )
            ai_agent_planner_user_prompt = st.text_area(
                "ai_agent_planner_user_prompt", height=PROMPT_TEXTAREA_HEIGHT
            )
        with st.expander("Subtask prompts"):
            ai_agent_subtask_system_prompt = st.text_area(
                "ai_agent_subtask_system_prompt", height=PROMPT_TEXTAREA_HEIGHT
            )
            ai_agent_subtask_tool_selection_user_prompt = st.text_area(
                "ai_agent_subtask_tool_selection_user_prompt",
                height=PROMPT_TEXTAREA_HEIGHT,
            )
            ai_agent_subtask_reflection_user_prompt = st.text_area(
                "ai_agent_subtask_reflection_user_prompt", height=PROMPT_TEXTAREA_HEIGHT
            )
            ai_agent_subtask_retry_answer_user_prompt = st.text_area(
                "ai_agent_subtask_retry_answer_user_prompt",
                height=PROMPT_TEXTAREA_HEIGHT,
            )
        with st.expander("Final answer prompts"):
            ai_agent_create_last_answer_system_prompt = st.text_area(
                "ai_agent_create_last_answer_system_prompt",
                height=PROMPT_TEXTAREA_HEIGHT,
            )
            ai_agent_create_last_answer_user_prompt = st.text_area(
                "ai_agent_create_last_answer_user_prompt", height=PROMPT_TEXTAREA_HEIGHT
            )

    st.title("🤖 Chatbot AI Agent")

    # 既存メッセージ表示（詳細ごと保持している場合はそちらを優先）
    if st.session_state.turns:
        for t in st.session_state.turns:
            user_text = t.get("user", "")
            asst_text = t.get("assistant", "")
            detail = t.get("detail", {})

            if user_text:
                with st.chat_message("user"):
                    st.markdown(user_text)
            if asst_text:
                with st.chat_message("assistant"):
                    st.markdown(asst_text)
                    if detail:
                        with st.expander("詳細結果 (plan, subtasks, RAGas, Langfuse)"):
                            if "latency_sec" in detail:
                                st.write({"latency_sec": detail["latency_sec"]})
                            if "plan" in detail and detail["plan"] is not None:
                                st.subheader("Plan")
                                st.write(detail["plan"])
                            if "subtasks" in detail and detail["subtasks"] is not None:
                                st.subheader("Subtasks")
                                st.write(detail["subtasks"])
                            if "ragas_scores" in detail:
                                st.subheader("RAGas scores")
                                st.write(detail.get("ragas_scores", {}))
                            if "langfuse_session_id" in detail and detail["langfuse_session_id"]:
                                st.subheader("Langfuse session id")
                                st.code(str(detail["langfuse_session_id"]))
    else:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    user_input = st.chat_input("メッセージを入力...")
    if user_input:
        # 送信: 直前までの履歴をchat_historyに
        st.session_state.messages.append({"role": "user", "content": user_input})
        # 直近のユーザー入力は即時表示（次回リロード待ちにしない）
        with st.chat_message("user"):
            st.markdown(user_input)

        chat_history = to_chat_history(st.session_state.messages[:-1])

        # JSONパラメータの解析
        planner_params = parse_json_or_none("planner_params", planner_params_raw)
        subtask_tool_selection_params = parse_json_or_none(
            "subtask_tool_selection_params", subtask_tool_selection_params_raw
        )
        subtask_answer_params = parse_json_or_none(
            "subtask_answer_params", subtask_answer_params_raw
        )
        subtask_reflection_params = parse_json_or_none(
            "subtask_reflection_params", subtask_reflection_params_raw
        )
        final_answer_params = parse_json_or_none(
            "final_answer_params", final_answer_params_raw
        )

        # 空文字はNoneへ
        def nvl(s: str) -> Optional[str]:
            return s if s else None

        # RAGasの必須チェック（未入力なら今回だけ自動無効化）
        ragas_ref_trim = (ragas_reference or "").strip()
        ragas_enabled = bool(is_run_ragas and ragas_ref_trim)
        # メッセージは出さず静かに無効化

        payload: Dict[str, Any] = {
            "question": user_input,
            "chat_history": chat_history,
            "planner_model_name": nvl(planner_model_name),
            "subtask_tool_selection_model_name": nvl(subtask_tool_selection_model_name),
            "subtask_answer_model_name": nvl(subtask_answer_model_name),
            "subtask_reflection_model_name": nvl(subtask_reflection_model_name),
            "final_answer_model_name": nvl(final_answer_model_name),
            "planner_params": planner_params,
            "subtask_tool_selection_params": subtask_tool_selection_params,
            "subtask_answer_params": subtask_answer_params,
            "subtask_reflection_params": subtask_reflection_params,
            "final_answer_params": final_answer_params,
            "ai_agent_planner_system_prompt": nvl(ai_agent_planner_system_prompt),
            "ai_agent_planner_user_prompt": nvl(ai_agent_planner_user_prompt),
            "ai_agent_subtask_system_prompt": nvl(ai_agent_subtask_system_prompt),
            "ai_agent_subtask_tool_selection_user_prompt": nvl(
                ai_agent_subtask_tool_selection_user_prompt
            ),
            "ai_agent_subtask_reflection_user_prompt": nvl(
                ai_agent_subtask_reflection_user_prompt
            ),
            "ai_agent_subtask_retry_answer_user_prompt": nvl(
                ai_agent_subtask_retry_answer_user_prompt
            ),
            "ai_agent_create_last_answer_system_prompt": nvl(
                ai_agent_create_last_answer_system_prompt
            ),
            "ai_agent_create_last_answer_user_prompt": nvl(
                ai_agent_create_last_answer_user_prompt
            ),
            "is_run_ragas": ragas_enabled,
            "ragas_reference": ragas_ref_trim if ragas_enabled else None,
        }

        st.session_state.last_request = payload

        with st.chat_message("assistant"):
            with st.spinner("エージェント実行中..."):
                try:
                    t0 = time.time()
                    resp = requests.post(
                        f"{DEFAULT_FASTAPI_BASE_URL}{EXEC_ENDPOINT}",
                        json=payload,
                        timeout=REQUEST_TIMEOUT_SEC,
                    )
                    latency = time.time() - t0
                    if resp.status_code != 200:
                        st.error(f"APIエラー: {resp.status_code} {resp.text}")
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": "すみません、エラーが発生しました。",
                            }
                        )
                    else:
                        data = resp.json()
                        st.session_state.last_response = data
                        answer = data.get("answer") or ""
                        st.markdown(answer)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer}
                        )

                        with st.expander("詳細結果 (plan, subtasks, RAGas, Langfuse)"):
                            latency_sec = round(latency, 2)
                            st.write({"latency_sec": latency_sec})
                            plan = (data.get("ai_agent_result") or {}).get("plan")
                            if plan:
                                st.subheader("Plan")
                                st.write(plan)
                            subtasks = (data.get("ai_agent_result") or {}).get(
                                "subtasks_detail"
                            )
                            if subtasks:
                                st.subheader("Subtasks")
                                st.write(subtasks)
                            ragas_scores = (data.get("ragas_result") or {}).get(
                                "scores"
                            )
                            st.subheader("RAGas scores")
                            st.write(ragas_scores)
                            sid = data.get("langfuse_session_id")
                            st.subheader("Langfuse session id")
                            st.code(sid)

                        # ターン詳細を履歴に保存（次回以降の再描画でも保持）
                        st.session_state.turns.append(
                            {
                                "user": user_input,
                                "assistant": answer,
                                "detail": {
                                    "latency_sec": round(latency, 2),
                                    "plan": plan,
                                    "subtasks": subtasks,
                                    "ragas_scores": ragas_scores or {},
                                    "langfuse_session_id": sid,
                                },
                            }
                        )
                except Exception as e:
                    st.error(f"通信エラー: {e}")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "すみません、通信エラーが発生しました。",
                        }
                    )

    st.divider()
    cols = st.columns(3)
    if cols[0].button("履歴クリア"):
        st.session_state.messages = []
        st.session_state.last_request = None
        st.session_state.last_response = None
        st.session_state.turns = []
        st.experimental_rerun()
    if cols[1].button("最後のリクエスト表示"):
        st.json(st.session_state.last_request)
    if cols[2].button("最後のレスポンス表示"):
        st.json(st.session_state.last_response)


if __name__ == "__main__":
    main()
