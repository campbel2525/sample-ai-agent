import os
import json
import time
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from streamlit.components.v1 import html as st_html

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
    # ペンディング送信の処理（先に実行してから描画）
    pending_payload = st.session_state.pop("pending_payload", None)
    if pending_payload is not None:
        # API呼び出し（会話は上側に描画され、その下に入力欄が来る）
        try:
            t0 = time.time()
            resp = requests.post(
                f"{DEFAULT_FASTAPI_BASE_URL}{EXEC_ENDPOINT}",
                json=pending_payload,
                timeout=REQUEST_TIMEOUT_SEC,
            )
            latency = time.time() - t0
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.last_request = pending_payload
                st.session_state.last_response = data
                answer = data.get("answer") or ""
                # 会話にassistantを追加
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
                # ターン履歴
                st.session_state.turns.append(
                    {
                        "user": pending_payload.get("query", ""),
                        "assistant": answer,
                        "detail": {
                            "latency_sec": round(latency, 2),
                            "raw_response": data,
                            "plan": (data.get("ai_agent_result") or {}).get("plan"),
                            "subtasks": (data.get("ai_agent_result") or {}).get(
                                "subtasks_detail"
                            ),
                            "ragas_scores": (data.get("ragas_result") or {}).get(
                                "scores"
                            )
                            or {},
                            "langfuse_session_id": data.get("langfuse_session_id"),
                        },
                    }
                )
            else:
                st.error(f"APIエラー: {resp.status_code} {resp.text}")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "すみません、エラーが発生しました。",
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

    # 入力欄の固定は描画崩れのため一旦オフ（最下部に通常表示）

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
                        # フルAPIレスポンスを表示
                        raw_resp = detail.get("raw_response")
                        if raw_resp is not None:
                            with st.expander("APIレスポンス（raw）"):
                                st.json(raw_resp)
                        # 参考：レイテンシ等の軽量メタ
                        if "latency_sec" in detail:
                            st.caption(f"latency: {detail['latency_sec']}s")
    else:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    # 入力欄（チャットの直下に配置・画面最下部に固定）
    # StreamlitはHTMLのネストを維持しないため、stFormを直接固定化するCSSを適用
    st.markdown(
        """
        <style>
        :root { --footer-height: 140px; }
        /* 本文がフッターで隠れないように下余白 */
        section.main > div.block-container { padding-bottom: var(--footer-height); }
        /* ページ内のstFormを固定フッター化（このアプリでは1つのみ） */
        section.main div[data-testid="stForm"] {
          position: fixed; left: 0; right: 0; bottom: 0; z-index: 1000;
          padding: 10px 16px; background: var(--footer-bg, rgba(255,255,255,0.97));
          box-shadow: 0 -2px 10px rgba(0,0,0,0.12);
        }
        /* 中身を中央寄せ（コンテンツ幅と揃えるための控えめな最大幅） */
        section.main div[data-testid="stForm"] > div { max-width: 1000px; margin: 0 auto; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.form("chat_form", clear_on_submit=True):
        chat_value = st.text_area(
            "メッセージ",
            key="chat_input_area",
            height=100,
            placeholder="メッセージを入力… (送信: ⌘/Ctrl + Enter)",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("送信", type="primary")

    # Cmd/Ctrl+Enter で送信ボタンをクリックするJS（簡易）
    st_html(
        """
        <script>
        (function(){
          // 背景色をテーマに合わせる
          try{
            const bg = getComputedStyle(parent.document.body).backgroundColor;
            const forms = parent.document.querySelectorAll('section.main div[data-testid="stForm"]');
            forms.forEach(f => f.style.background = bg);
          }catch(_){ }

          function clickSend(){
            const btns = parent.document.querySelectorAll('button');
            for(let i=btns.length-1;i>=0;i--){
              const t = (btns[i].innerText||'').trim();
              if(t === '送信'){ btns[i].click(); break; }
            }
          }
          window.addEventListener('keydown', function(e){
            if ((e.metaKey||e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); clickSend(); }
          }, true);
        })();
        </script>
        """,
        height=0,
    )

    # 送信処理（フォームはページ最下部に1つだけ）。即時APIは叩かずpayloadを保存→再描画の先頭で処理
    if submitted and chat_value and chat_value.strip():
        user_input: str = chat_value.strip()

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

        def nvl(s: str) -> Optional[str]:
            return s if s else None

        # RAGas 実行可否と参照の整形
        ragas_ref_trim = (ragas_reference or "").strip()
        # チェックが入っているのに参照が空なら、APIバリデーションで422になるため事前に警告して送信しない
        if is_run_ragas and not ragas_ref_trim:
            st.warning("RAGasを実行するには 'RAGas reference' の入力が必要です。")
            return

        # 送信可となった段階でユーザー発話を履歴に追加し、履歴を作成
        st.session_state.messages.append({"role": "user", "content": user_input})
        chat_history = to_chat_history(st.session_state.messages[:-1])

        payload: Dict[str, Any] = {
            "query": user_input,
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
            # チェックボックスの値をそのまま渡す（事前に参照の必須チェック済み）
            "is_run_ragas": is_run_ragas,
            "ragas_reference": ragas_ref_trim if ragas_ref_trim else None,
        }

        st.session_state["pending_payload"] = payload
        st.rerun()

    # （操作ボタン省略）


if __name__ == "__main__":
    main()
