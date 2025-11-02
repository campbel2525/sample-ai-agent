# """
# # 概要

# サンプルのコードです

# # プロンプトについて

# `from ai_agents.agents.general_purpose_ai_agent.prompts import AgentPrompts`
# は汎用的なプロンプトです。これと同じ構成であればプロンプトの内容は変えてしまっていいです。
# むしろエージェントごとにプロンプトは変えるべきです。
# """

# from langchain.tools import BaseTool

# from ai_agents.agents.general_purpose_ai_agent.agent import Agent
# from ai_agents.agents.general_purpose_ai_agent.models import LLMPhaseConfigs
# from ai_agents.agents.general_purpose_ai_agent.prompts import AgentPrompts
# from services.langfuse.wrapper import run_agent_with_langfuse
# from ai_agents.tools.hybrid_search_tool import HybridSearchTool
# from ai_agents.tools.random_tool import RandomTool
# from config.settings import Settings
# from services.opensearch_service import hybrid_search


# def test1_raw_ai_agent() -> None:
#     """
#     AIエージェントの基本的な動作テスト。

#     Langfuseなしで純粋なAIエージェントを作成し、
#     RandomToolを使用して乱数生成のテストを実行する。

#     Returns:
#         None: 結果はコンソールに出力される
#     """

#     settings = Settings()

#     # ツールの準備
#     tools: list[BaseTool] = [
#         RandomTool(),
#     ]

#     query = "ランダムな整数を1つ答えてください"

#     # 純粋なエージェントを作成（Langfuseなし）
#     agent = Agent(
#         openai_api_key=settings.openai_api_key,
#         openai_base_url=settings.openai_base_url,
#         llm_phase_configs=LLMPhaseConfigs(),
#         tools=tools,
#         prompts=AgentPrompts(),
#         max_challenge_count=3,
#     )
#     result = agent.run_agent(query)

#     print(f"計画: {result.plan.subtasks}")
#     print(f"質問: {result.query}")
#     print(f"回答: {result.answer}")


# def test2_langfuge_ai_agent() -> None:
#     """
#     Langfuseトレーシング機能付きAIエージェントのテスト。

#     AIエージェントをLangfuseラッパー経由で実行し、
#     実行過程のトレーシングと計画の詳細を確認する。
#     RandomToolを使用して乱数生成のテストを実行。

#     Returns:
#         None: 結果はコンソールに出力される
#     """

#     settings = Settings()

#     # ツールを準備
#     tools: list[BaseTool] = [
#         RandomTool(),
#     ]

#     query = "ランダムな整数を1つ答えてください"

#     # 純粋なエージェントを作成（Langfuse一切なし）
#     agent = Agent(
#         openai_api_key=settings.openai_api_key,
#         openai_base_url=settings.openai_base_url,
#         llm_phase_configs=LLMPhaseConfigs(),
#         tools=tools,
#         prompts=AgentPrompts(),
#         max_challenge_count=3,
#     )

#     # Langfuse付きで実行
#     result = run_agent_with_langfuse(
#         agent=agent,
#         query=query,
#         langfuse_public_key=settings.langfuse_public_key,
#         langfuse_secret_key=settings.langfuse_secret_key,
#         langfuse_host=settings.langfuse_host,
#     )

#     print(f"計画: {result.plan.subtasks}")
#     print(f"質問: {result.query}")
#     print(f"回答: {result.answer}")


# def test3_chatbot() -> None:
#     """
#     ハイブリッド検索機能付きチャットボットのテスト。

#     OpenSearchのハイブリッド検索（BM25 + ベクトル検索）を使用して
#     知識ベースから情報を検索し、質問に回答するチャットボットの
#     動作をテストする。

#     Returns:
#         None: 結果はコンソールに出力される
#     """

#     settings = Settings()

#     # ツールを準備
#     hybrid_search_tool = HybridSearchTool(
#         openai_api_key=settings.openai_api_key,
#         openai_base_url=settings.openai_base_url,
#         openai_embedding_model=settings.openai_embedding_model,
#         openai_max_retries=3,
#         opensearch_base_url=settings.opensearch_base_url,
#         opensearch_index_name=settings.opensearch_default_index_name,
#     )
#     tools: list[BaseTool] = [
#         hybrid_search_tool,
#     ]

#     query = "キアヌリーブスについて教えてください"

#     # 純粋なエージェントを作成
#     agent = Agent(
#         openai_api_key=settings.openai_api_key,
#         openai_base_url=settings.openai_base_url,
#         llm_phase_configs=LLMPhaseConfigs(),
#         tools=tools,
#         prompts=AgentPrompts(),
#         max_challenge_count=3,
#     )

#     # Langfuse付きで実行
#     result = run_agent_with_langfuse(
#         agent=agent,
#         query=query,
#         langfuse_public_key=settings.langfuse_public_key,
#         langfuse_secret_key=settings.langfuse_secret_key,
#         langfuse_host=settings.langfuse_host,
#         langfuse_session_id="test_session_id1",
#         langfuse_user_id=1,
#     )

#     # Langfuse付きで実行
#     result = run_agent_with_langfuse(
#         agent=agent,
#         query=query,
#         langfuse_public_key=settings.langfuse_public_key,
#         langfuse_secret_key=settings.langfuse_secret_key,
#         langfuse_host=settings.langfuse_host,
#         langfuse_session_id="test_session_id1",
#         langfuse_user_id=1,
#     )

#     # Langfuse付きで実行
#     result = run_agent_with_langfuse(
#         agent=agent,
#         query=query,
#         langfuse_public_key=settings.langfuse_public_key,
#         langfuse_secret_key=settings.langfuse_secret_key,
#         langfuse_host=settings.langfuse_host,
#         langfuse_session_id="test_session_id2",
#         langfuse_user_id=1,
#     )

#     print(f"計画: {result.plan.subtasks}")
#     print(f"質問: {result.query}")
#     print(f"回答: {result.answer}")


# def test4_opensearch_hybrid() -> None:
#     """
#     OpenSearchハイブリッド検索の直接テスト。

#     AIエージェントを介さずに、OpenSearchのハイブリッド検索機能を
#     直接呼び出してテストする。BM25とベクトル検索を組み合わせた
#     検索結果から_sourceフィールドのみを抽出して表示。

#     Returns:
#         None: 検索結果はコンソールに出力される
#     """

#     query = "キアヌリーブスについて教えてください"

#     settings = Settings()
#     result = hybrid_search(
#         openai_api_key=settings.openai_api_key,
#         openai_base_url=settings.openai_base_url,
#         openai_embedding_model=settings.openai_embedding_model,
#         openai_max_retries=3,
#         opensearch_base_url=settings.opensearch_base_url,
#         opensearch_index_name=settings.opensearch_default_index_name,
#         query=query,
#         k=50,
#         size=20,
#     )

#     sources = []
#     for hit in result.get("hits", {}).get("hits", []):
#         sources.append(hit["_source"])

#     print(sources)


# def test5_ragas():
#     from ragas import evaluate
#     from ragas.dataset_schema import EvaluationDataset
#     from ragas.embeddings import LangchainEmbeddingsWrapper
#     from ragas.llms import LangchainLLMWrapper
#     from ragas.metrics import answer_relevancy, answer_similarity

#     from config.settings import Settings
#     from services.openai_service import get_embedding_client, get_openai_client

#     settings = Settings()

#     # RAGAS評価の実行
#     ragas_data = [
#         {
#             "user_input": "キアヌリーブスについて教えてください",
#             "retrieved_contexts": [
#                 "キアヌ・チャールズ・リーブス（Keanu Charles Reeves, 1964年9月2日 - ）は、"
#                 "カナダの俳優・ロックバンド「ドッグスター」のメンバー。俳優としての代表作は"
#                 "『スピード』や『マトリックス』シリーズ、『ジョン・ウィック』シリーズなど。",
#                 "1994年の映画『スピード』の大ヒットにより国際的スターとなった。"
#                 "1999年の映画『マトリックス』が世界的に大ヒットし再ブレイク、"
#                 "3部作に主演し、人気を不動のものとした。",
#                 "「聖人」と呼ばれるほどの人格者で知られており、2010年代後半になると"
#                 "キアヌの聖人エピソードがインターネット・ミームとまでなっている。",
#                 "ギャラの大部分は小児科病院や癌治療の研究などへ寄付している。"
#                 "代表作としても知られる映画「マトリックス」で得た報酬の70％を"
#                 "ガン研究に寄付していた。",
#             ],
#             "response": "マトリックスの映画に出演していました。",
#             "reference": (
#                 "キアヌ・リーブスの代表作には『スピード』（1994年）、『マトリックス』シリーズ（1999年〜）、"
#                 "『ジョン・ウィック』シリーズ（2014年〜）があります。彼は「聖人」と呼ばれるほどの人格者として知られ、"
#                 "映画の報酬の大部分を慈善事業に寄付するなど、その優しい人柄でも有名です。"
#                 "特に『マトリックス』の報酬の70％をガン研究に寄付したエピソードは広く知られています。"
#             ),
#             "reference_contexts": None,
#         }
#     ]

#     ragas_metrics = [
#         answer_relevancy,
#         answer_similarity,
#     ]

#     openai_client = get_openai_client(
#         settings.openai_base_url,
#         settings.openai_api_key,
#         settings.openai_model,
#     )
#     embedding_client = get_embedding_client(
#         settings.openai_base_url,
#         settings.openai_api_key,
#         settings.openai_embedding_model,
#         max_retries=3,
#     )

#     evaluate(
#         dataset=EvaluationDataset.from_list(ragas_data),
#         metrics=ragas_metrics,
#         llm=LangchainLLMWrapper(openai_client),
#         embeddings=LangchainEmbeddingsWrapper(embedding_client),
#         raise_exceptions=True,
#         show_progress=True,
#     )


# if __name__ == "__main__":
#     # from config.debug import *

#     test1_raw_ai_agent()
#     test2_langfuge_ai_agent()
#     test3_chatbot()
#     test4_opensearch_hybrid()
#     test5_ragas()

#     pass
