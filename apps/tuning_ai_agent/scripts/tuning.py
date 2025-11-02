import os
import shutil
from datetime import datetime
from typing import Any, Dict, List
from urllib.parse import urljoin

import anthropic
import requests
import yaml

from config.prompts import (
    AI_AGENT_UPDATE_PROMPT,
    ANSWER_EVALUATION_PROMPT,
    IMPROVEMENT_POINT_PROMPT,
)
from config.settings import Settings
from utils.file_utils import load_yaml_data, save_json, save_yaml


def load_current_prompts(
    execution_datetime: str, test_execution_no: str
) -> Dict[str, str]:
    """
    現在のプロンプトを読み込む関数
    """
    # チューニングされたプロンプトは実行結果フォルダに保存されている
    prompt_path = f"data/tuning_result/{execution_datetime}/実行結果/{test_execution_no}/prompt.yml"
    if os.path.exists(prompt_path):
        data = load_yaml_data(prompt_path)
        return data.get("prompt", {})
    else:
        raise FileNotFoundError(f"プロンプトファイルが見つかりません: {prompt_path}")


def ensure_result_dirs(execution_datetime: str, test_execution_no: str):
    """
    テスト結果用のディレクトリを作成する関数
    """
    base_dir = f"data/tuning_result/{execution_datetime}/実行結果/{test_execution_no}"
    result_dir = f"{base_dir}/result"

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)


def call_ai_agent_api(
    query: str,
    ragas_reference: str,
    current_prompts: Dict[str, str],
) -> Dict[str, Any]:
    """
    AIエージェントAPIを呼び出す関数
    """
    settings = Settings()

    # APIリクエストのペイロード
    payload = {
        "query": query,
        "ai_agent_planner_system_prompt": current_prompts.get(
            "ai_agent_planner_system_prompt", ""
        ),
        "ai_agent_planner_user_prompt": current_prompts.get(
            "ai_agent_planner_user_prompt", ""
        ),
        "ai_agent_subtask_system_prompt": current_prompts.get(
            "ai_agent_subtask_system_prompt", ""
        ),
        "ai_agent_subtask_tool_selection_user_prompt": current_prompts.get(
            "ai_agent_subtask_tool_selection_user_prompt", ""
        ),
        "ai_agent_subtask_reflection_user_prompt": current_prompts.get(
            "ai_agent_subtask_reflection_user_prompt", ""
        ),
        "ai_agent_subtask_retry_answer_user_prompt": current_prompts.get(
            "ai_agent_subtask_retry_answer_user_prompt", ""
        ),
        "ai_agent_create_last_answer_system_prompt": current_prompts.get(
            "ai_agent_create_last_answer_system_prompt", ""
        ),
        "ai_agent_create_last_answer_user_prompt": current_prompts.get(
            "ai_agent_create_last_answer_user_prompt", ""
        ),
        "is_run_ragas": True,
        "ragas_reference": ragas_reference,
    }

    api_url = urljoin(settings.ai_agent_api_url, "ai_agents/chatbot/exec")
    try:
        print(f"AIエージェントAPIを呼び出し中: {api_url}")
        print(f"質問: {query}")

        # AIエージェントAPIを呼び出し
        response = requests.post(
            api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300,  # 5分のタイムアウト
        )
        response.raise_for_status()

        result = response.json()
        print(f"API呼び出し成功: ステータスコード {response.status_code}")
        return result

    except Exception as e:
        print(f"API呼び出しエラー: {str(e)}")
        # エラーの場合はダミーデータを返す
        return {
            "query": query,
            "answer": f"エラーが発生しました: {str(e)}",
            "plan": [],
            "subtasks_detail": [],
            "total_subtasks": 0,
            "completed_subtasks": 0,
            "total_challenge_count": 0,
            "ragas_scores": {
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
            },
            "langfuse_session_id": "error",
            "execution_time": 0.0,
        }


def generate_llm_judge_evaluation(api_result: Dict[str, Any]) -> Dict[str, str]:
    """
    LLMを使用してAPIの結果を評価する関数
    """
    settings = Settings()

    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

        # 回答評価の生成
        answer_evaluation_prompt = f"""
        以下のAIエージェントの回答について考察してください：

        質問: {api_result.get('query', '')}
        回答: {api_result.get('answer', '')}

        {ANSWER_EVALUATION_PROMPT}
        """

        response = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=4000,
            messages=[{"role": "user", "content": answer_evaluation_prompt}],
        )
        answer_evaluation_result = response.content

        # AIMessageオブジェクトを文字列に変換
        if hasattr(answer_evaluation_result, "content"):
            answer_evaluation = answer_evaluation_result.content
        else:
            answer_evaluation = str(answer_evaluation_result)

        # 改善点の生成
        improvement_prompt = f"""
        以下のAIエージェントの実行結果について改善点を考察してください：

        質問: {api_result.get('query', '')}
        回答: {api_result.get('answer', '')}
        実行時間: {api_result.get('execution_time', 0)}秒
        完了したサブタスク: {api_result.get('completed_subtasks', 0)}/{api_result.get('total_subtasks', 0)}

        {IMPROVEMENT_POINT_PROMPT}
        """

        response = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=4000,
            messages=[{"role": "user", "content": improvement_prompt}],
        )
        improvement_points_result = response.content

        # AIMessageオブジェクトを文字列に変換
        if hasattr(improvement_points_result, "content"):
            improvement_points = improvement_points_result.content
        else:
            improvement_points = str(improvement_points_result)

        return {
            "answer_evaluation": answer_evaluation,
            "improvement_points": improvement_points,
        }

    except Exception as e:
        return {
            "answer_evaluation": f"評価中にエラーが発生しました: {str(e)}",
            "improvement_points": f"改善点の生成中にエラーが発生しました: {str(e)}",
        }


def update_prompts_with_ai(
    current_prompts: Dict[str, str], test_results: List[Dict[str, Any]]
) -> Dict[str, str]:
    """
    AIを使用してプロンプトを更新する関数
    """
    settings = Settings()

    try:
        # OpenAIクライアントを取得
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

        # テスト結果の要約を作成
        results_summary = ""
        for i, result in enumerate(test_results, 1):
            api_result = result.get("api_result", {})
            llm_judge = result.get("llm_as_a_judge", {})

            results_summary += f"""
            テスト{i}:
            質問: {api_result.get('query', '')}
            回答: {api_result.get('answer', '')}
            実行時間: {api_result.get('execution_time', 0)}秒
            評価: {llm_judge.get('answer_evaluation', '')}
            改善点: {llm_judge.get('improvement_points', '')}

            """

        # プロンプト更新の指示
        update_prompt = f"""
        以下の現在のプロンプトとテスト結果を基に、より良い結果を得られるようにプロンプトを改善してください：

        現在のプロンプト:
        {yaml.dump(current_prompts, default_flow_style=False, allow_unicode=True)}

        テスト結果:
        {results_summary}

        {AI_AGENT_UPDATE_PROMPT}

        改善されたプロンプトをYAML形式で出力してください。
        """
        response = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=4000,
            messages=[{"role": "user", "content": update_prompt}],
        )
        updated_prompts_result = response.content

        # AIMessageオブジェクトを文字列に変換
        if hasattr(updated_prompts_result, "content"):
            updated_prompts_text = updated_prompts_result.content
        else:
            updated_prompts_text = str(updated_prompts_result)

        # YAMLパースを試行
        try:
            print(f"OpenAIからの応答:\n{updated_prompts_text}")

            # YAML部分を抽出（```yaml で囲まれている場合）
            if "```yaml" in updated_prompts_text:
                start = updated_prompts_text.find("```yaml") + 7
                end = updated_prompts_text.find("```", start)
                if end != -1:
                    yaml_content = updated_prompts_text[start:end].strip()
                else:
                    yaml_content = updated_prompts_text[start:].strip()
            else:
                yaml_content = updated_prompts_text

            print(f"抽出されたYAML:\n{yaml_content}")

            updated_prompts = yaml.safe_load(yaml_content)
            if isinstance(updated_prompts, dict):
                print("プロンプト更新成功")
                return updated_prompts
            else:
                print(f"パース結果が辞書ではありません: {type(updated_prompts)}")
        except Exception as e:
            print(f"YAMLパースエラー: {str(e)}")

        # パースに失敗した場合は現在のプロンプトを返す
        print("プロンプト更新失敗、現在のプロンプトを維持")
        return current_prompts

    except Exception as e:
        print(f"プロンプト更新中にエラーが発生しました: {str(e)}")
        return current_prompts


def run_api_with_test_data(
    execution_datetime: str,
    test_execution_no: str,
):
    """
    テストデータを使用してAIエージェントAPIを実行し、結果を保存する関数
    """
    # テストデータを読み込む
    test_data = load_yaml_data("data/test_data/test_data.yml")
    test_data_list = test_data.get("test_data", [])

    if len(test_data_list) == 0:
        raise ValueError("テストデータが空です: data/test_data/test_data.yml")

    # 現在のプロンプトを読み込む
    current_prompts = load_current_prompts(execution_datetime, test_execution_no)

    # ディレクトリを作成
    ensure_result_dirs(execution_datetime, test_execution_no)

    # テスト結果を保存するリスト
    test_results = []

    # テストデータごとにapiを実行する
    for index, test_data in enumerate(test_data_list):
        print(f"テスト{test_execution_no} - データ{index+1}を実行中...")

        # APIを実行
        api_result = call_ai_agent_api(
            query=test_data.get("query", ""),
            ragas_reference=test_data.get("reference", ""),
            current_prompts=current_prompts,
        )

        # LLMを使用した評価を生成
        llm_judge_result = generate_llm_judge_evaluation(api_result)

        # 結果をまとめる
        result_data = {"api_result": api_result, "llm_as_a_judge": llm_judge_result}

        test_results.append(result_data)

        # 結果を保存
        result_path = f"data/tuning_result/{execution_datetime}/実行結果/{test_execution_no}/result/test_no_{test_data['test_no']}.json"  # noqa E501
        save_json(result_path, result_data)

        print(f"結果を保存しました: {result_path}")

    # プロンプトは既に保存済み（run_tuning関数で管理）
    return test_results


def run_tuning():
    run_count = 3  # 実行回数

    # 実行日時を生成
    execution_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    print(f"チューニング開始: {execution_datetime}")

    # 初回プロンプトをdata/test_data/initial_prompt.ymlから読み込み、テストフォルダにinitial_prompt.ymlとしてコピー
    # initial_prompt_path = f"data/tuning_result/{execution_datetime}/initial_prompt.yml"
    # target_prompt_path = (
    #     f"data/tuning_result/{execution_datetime}/実行結果/1/prompt.yml"
    # )

    # source_prompt_path = "data/test_data/initial_prompt.yml"
    # if not os.path.exists(source_prompt_path):
    #     raise FileNotFoundError(
    #         f"初回プロンプトのソースファイルが見つかりません: {source_prompt_path}"
    #     )

    # # ソースファイルを読み込み
    # data = load_yaml_data(source_prompt_path)

    # テストフォルダにinitial_prompt.ymlとしてコピー
    # os.makedirs(f"data/tuning_result/{execution_datetime}", exist_ok=True)

    # save_yaml(initial_prompt_path, data)
    # print(f"初回プロンプトを保存しました: {initial_prompt_path}")

    # # 実行結果1のフォルダにコピー
    # os.makedirs(f"data/tuning_result/{execution_datetime}/実行結果/1", exist_ok=True)
    # save_yaml(target_prompt_path, data)
    # print(f"実行用プロンプトを保存しました: {target_prompt_path}")

    # 実行回数分のフォルダ作成とテスト実行
    all_test_results = []

    for index in range(run_count):
        test_execution_no = index + 1
        print(f"\n=== テスト実行 {test_execution_no}/{run_count} ===")

        # 実行結果用のディレクトリを作成
        base_dir = (
            f"data/tuning_result/{execution_datetime}/実行結果/{test_execution_no}/"
        )
        os.makedirs(base_dir, exist_ok=True)

        # プロンプトコピー
        if index == 0:
            initial_prompt_path = "data/test_data/initial_prompt.yml"
            target_prompt_path = f"{base_dir}prompt.yml"  # noqa E501
            shutil.copy2(initial_prompt_path, target_prompt_path)

        else:
            # 1個前のプロンプトを次の実行用にコピー
            prev_prompt_path = f"data/tuning_result/{execution_datetime}/実行結果/{test_execution_no - 1}/prompt.yml"  # noqa E501
            next_prompt_path = f"{base_dir}prompt.yml"  # noqa E501
            shutil.copy2(prev_prompt_path, next_prompt_path)

        # テストデータに対してapiを実行し保存
        test_results = run_api_with_test_data(execution_datetime, test_execution_no)
        all_test_results.extend(test_results)

        # 次の実行結果のフォルダに保存する
        if test_execution_no < run_count:
            # 現在のプロンプトを読み込み
            current_prompts = load_current_prompts(
                execution_datetime, test_execution_no
            )

            # AIを使用してプロンプトを更新
            print("AIを使用してプロンプトを更新中...")
            updated_prompts = update_prompts_with_ai(current_prompts, test_results)

            # 次回実行用のプロンプトを保存
            next_prompt_path = f"data/tuning_result/{execution_datetime}/実行結果/{test_execution_no}/prompt.yml"  # noqa E501
            os.makedirs(
                f"data/tuning_result/{execution_datetime}/実行結果/{test_execution_no}",
                exist_ok=True,
            )

            save_yaml(next_prompt_path, {"prompt": updated_prompts})
            print(f"更新されたプロンプトを保存しました: {next_prompt_path}")
        else:
            # 最終結果を保存
            final_result_path = (
                f"data/tuning_result/{execution_datetime}/final_result.yml"
            )
            final_prompts = load_current_prompts(execution_datetime, test_execution_no)

            final_result = {
                "execution_datetime": execution_datetime,
                "total_tests": len(all_test_results),
                "final_prompts": final_prompts,
                "summary": "チューニング完了",
            }

            save_yaml(final_result_path, final_result)
            print(f"最終結果を保存しました: {final_result_path}")

    print(f"\nチューニング完了: {execution_datetime}")
    print(f"結果は data/tuning_result/{execution_datetime}/ に保存されました")


if __name__ == "__main__":
    run_tuning()
