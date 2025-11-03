## [English README](https://github.com/campbel2525/sample-ai-agent/blob/main/README-ENGLISH.md)

# 概要

以下のシステムを作りました

- チャットボット AI エージェントシステム(apps/ai_agent)

  - chatbot を実行する AI エージェント機能

    - 資料から検索を行い返答するチャットボット
    - ユーザーの質問が曖昧の場合は「追い質問」を行う
    - ~~資料検索の結果によってユーザーに質問をした方がよければユーザーに「追い質問」を行う~~
    - ユーザーの質問が資料の中にない情報の場合は「回答なし」という旨の返答を行う

  - API 化

- チューニング AI システム(apps/tuning_ai_agent)
  - AI エージェントをチューニングする機能
  - まだ精度があまり良くなく調整の余地があります。

## 全体の構成図

<img width="2232" height="1268" alt="image" src="https://github.com/user-attachments/assets/0a42a70b-a508-4cd0-b7ee-81caf80db9f3" />

# AI エージェントシステム

AI エージェントの機能になります

AI エージェントは FastAPI にて API 化されているため API を叩くことで AI エージェントが実行できます。

リクエスト Body に実行するプロンプトを指定できるため AI エージェントのプロンプトのチューニングを行えるようになっています。

プロンプトを変更しながら API を叩いてプロンプトのチューニングをしてください。

## 主な機能

- メインの機能は chatbot の AI エージェント API
  - シングルエージェント
  - Plan-and-Execute 型
- プログラムのディレクトリ: `apps/ai_agent`
- FastAPI で API 化
  - 詳しい API 仕様書は[FastAPI の Docs](http://localhost:8000/docs)を見ること
  - リクエスト Body
    - AI エージェントの実行に必要なプロンプト
    - RAGas の実行に必要な情報
  - レスポンス
    - チューニングに必要な情報一式
    - AI エージェントの最終出力結果
    - AI エージェントの実行記録
    - Langfuse の ID など
    - RAGas の結果
  - 今回は API の機能はサブ機能と位置付けているためフォルダ構成などにこだわらず 1 ファイル(`apps/ai_agent/run_fastapi.py`)にまとめている。また認証機能などもなし。
- streamli-ui
  - AI エージェント API を使用する UI 機能
- Langfuse を使用しているため AI エージェントの実行ログをブラウザで確認可能
  - https://langfuse.com/
  - `answer_relevancy`、`answer_similarity`のみ返すようになっている
- 検索 DB には OpenSearch を利用してハイブリッド検索(フルテキスト検索+ベクトル検索)を実行
  - サンプルデータとして[ウィキペディア キアヌ・リーブス](https://ja.wikipedia.org/wiki/%E3%82%AD%E3%82%A2%E3%83%8C%E3%83%BB%E3%83%AA%E3%83%BC%E3%83%96%E3%82%B9)を利用
  - 今回は AI エージェントの動きを確かめることが目的なため、特にこだわらずチャンクは 512 文字、128 文字の重複で挿入
  - ファイルは`project/data/test_data.txt`

## その他

以下の技術書を参考にしました。(かなりいい本なのでおすすめです)

- AI エージェントの開発は[現場で活用するための AI エージェント実践入門](https://www.amazon.co.jp/%E7%8F%BE%E5%A0%B4%E3%81%A7%E6%B4%BB%E7%94%A8%E3%81%99%E3%82%8B%E3%81%9F%E3%82%81%E3%81%AEAI%E3%82%A8%E3%83%BC%E3%82%B8%E3%82%A7%E3%83%B3%E3%83%88%E5%AE%9F%E8%B7%B5%E5%85%A5%E9%96%80-KS%E6%83%85%E5%A0%B1%E7%A7%91%E5%AD%A6%E5%B0%82%E9%96%80%E6%9B%B8-%E5%A4%AA%E7%94%B0-%E7%9C%9F%E4%BA%BA/dp/4065401402)を参考にしました。

# チューニング AI システム

エージェントのプロンプトを持続的に AI が改良していくための機能になります。

AI エージェントは API 化されているため、API を実行しその結果を元に LLM がプロンプトを調整して、再度 API を叩くといったことができます

あまり精度が良くないです。

## 主な機能

- メインの機能は AI エージェントのプロンプトチューニング
- プログラムのディレクトリ: `apps/tuning_ai_agent`
- 実行回数をあらかじめ決めておいてその回数 AI エージェント API を叩いてプロンプトをチューニングする。

# 技術スタック

- Python
- OpenSearch
  - https://opensearch.org/
- LangChain
  - https://www.langchain.com/
- LangGraph
  - https://www.langchain.com/langgraph
- Langfuse
  - https://langfuse.com/
- Docker
- OpenAI API

# 環境構築方法

1. .env の設定

- `apps/ai_agent/.env.example.example`を参考に`apps/ai_agent/.env`を作成
- `apps/tuning_ai_agent/.env.example`を参考に`apps/tuning_ai_agent/.env`を作成

2. Docker 環境の作成

```
make init
```

3. 動作確認

以下の URL をブラウザで開いて画面が表示されれば問題なし

- FastAPI の Swagger 画面
  - [http://localhost:8000/docs](http://localhost:8000/docs)
- Langfuse の管理画面
  - [http://localhost:3000/](http://localhost:3000/)
  - ID: `admin@example.com`
  - Password: `secret1234!`
- OpenSearch の管理画面
  - [http://localhost:5601/](http://localhost:5601/)

3. OpenSearch にデータのセットアップ

```
make opensearch-setup
```

キアヌリーブスについての内容(`data/insert_data/test_data.txt`の中のテキスト)が 512 文字、128 文字の重複のチャンクされて OpenSearch の index にインサート

4. AI エージェント API の起動

```
make ai-agent-run
```

AI エージェント API のテストの方法

[http://localhost:8000/docs](http://localhost:8000/docs)を開いて`Exec Chatbot Ai Agent`の API を実行して適切にレスポンスが返ってくれば OK

リクエスト Body には以下を指定する

```json
{
  "query": "Pythonでファイルを読み込む方法を教えてください",
  "chat_history": [
    {
      "content": "Pythonについて教えて",
      "role": "user",
      "timestamp": "2025-01-17T10:00:00Z"
    },
    {
      "content": "Pythonは汎用プログラミング言語です。",
      "role": "assistant",
      "timestamp": "2025-01-17T10:00:30Z"
    }
  ],
  "planner_model_name": "gpt-4o-2024-08-06",
  "planner_model_params": {
    "seed": 0,
    "temperature": 0
  },
  "subtask_tool_selection_model_name": "gpt-4o-2024-08-06",
  "subtask_tool_selection_model_params": {
    "seed": 0,
    "temperature": 0
  },
  "subtask_retry_answer_model_params": {
    "seed": 0,
    "temperature": 0
  },
  "final_answer_model_name": "gpt-4o-2024-08-06",
  "subtask_reflection_model_params": {
    "seed": 0,
    "temperature": 0
  },
  "final_answer_model_params": {
    "seed": 0,
    "temperature": 0
  },
  "planner_system_prompt": "あなたは優秀なプランナーです。ユーザーの質問を分析し、適切なサブタスクに分解してください。",
  "planner_user_prompt": "質問: {query}\n\n上記の質問に答えるために必要なサブタスクを作成してください。",
  "subtask_tool_selection_system_prompt": "あなたは与えられたサブタスクを実行する専門家です。利用可能なツールを使用してタスクを完了してください。",
  "subtask_tool_selection_user_prompt": "サブタスク: {subtask}\n\n上記のサブタスクを実行するために最適なツールを選択し、実行してください。",
  "subtask_reflection_user_prompt": "サブタスク: {subtask}\nツール実行結果: {tool_result}\n\n上記の結果がサブタスクの要求を満たしているか評価してください。",
  "subtask_reflection_model_name": "gpt-4o-2024-08-06",
  "subtask_retry_answer_user_prompt": "前回の試行が不十分でした。アドバイス: {advice}\n\n改善されたアプローチでサブタスクを再実行してください。",
  "subtask_retry_answer_model_name": "gpt-4o-2024-08-06",
  "final_answer_system_prompt": "あなたは全てのサブタスクの結果を統合し、ユーザーの質問に対する最終的な回答を作成する専門家です。",
  "final_answer_user_prompt": "質問: {query}\nサブタスク結果: {subtask_results}\n\n上記の情報を基に、質問に対する包括的で分かりやすい回答を作成してください。",
  "is_run_ragas": true,
  "ragas_reference": "Pythonでファイルを読み込むには、open()関数を使用し、with文と組み合わせることが推奨されます。"
}
```

5. streamlit の起動

```
make streamlit-ui-run
```

[http://localhost:8501/](http://localhost:8501/)を開いて画面が表示されれば OK

# チューニング AI を使ったチューニング方法

**まだまだ精度が良くなく改善の余地があります**

AI エージェントで使用するプロンプトをチューニングすることを想定しています

プロンプトを変えながら AI エージェント API を実行して結果を元にプロンプトをチューニングしていきます

## 手順

1. 質問と回答の組み合わせを複数用意する。保存ファイル名は`data/test_data/test_data.yml`
2. 初期プロンプトは`data/test_data/initial_prompt.yml`を利用
3. AI エージェント API を起動。

```
make ai-agent-run
```

4. チューニング実行

```
docker compose -f "./docker/local/docker-compose.yml" -p chatbot-ai-agent exec -it tuning-ai-agent pipenv run python scripts/tuning.py
```

5. 結果は`data/tuning_result`に出力される。どのようなフォルダ構成で出力されるかは`data/tuning_result/0sample`を参照

# 今後の課題、やってみたいこと

- AI エージェント自体のプログラムの修正機能はないのでなんかしら導入したい。
  - Cline を利用する？
  - Claude Code を利用する？
- AI エージェントにはメモリ機能があるといいので使用したい
  - 成功した内容を保存するようにすれば使えば使うほど性能が良くなると言いたことも可能らしい
- AI エージェントの仕組み自体も変えてみるのも良さそう
  - chatbot なので対話形式を想定しているが、自分で用意したファイル群から DeepResearch をするのもいいかもしれない
- LLM のモデルの比較を検討してみたい
  - 特にプラン作成の LLM は gpt-5 とか熟考モードを使用したい
- チューニング AI が精度が悪いので改善したい
  - そもそもプロンプトのチューニングで使用していない。
    - 現状は Codex を使用して開発を進めている
  - チューニング AI が実行するプロンプトが良くない。
    - どんな AI エージェントなのかの情報がプロンプトに入っていない
    - ここも外部から入れれるようにするといいと思う
  - テストフォルダを作ってコマンド実行の際に引数で渡すようにする？

# 参考文献

- [現場で活用するための AI エージェント実践入門](https://www.amazon.co.jp/%E7%8F%BE%E5%A0%B4%E3%81%A7%E6%B4%BB%E7%94%A8%E3%81%99%E3%82%8B%E3%81%9F%E3%82%81%E3%81%AEAI%E3%82%A8%E3%83%BC%E3%82%B8%E3%82%A7%E3%83%B3%E3%83%88%E5%AE%9F%E8%B7%B5%E5%85%A5%E9%96%80-KS%E6%83%85%E5%A0%B1%E7%A7%91%E5%AD%A6%E5%B0%82%E9%96%80%E6%9B%B8-%E5%A4%AA%E7%94%B0-%E7%9C%9F%E4%BA%BA/dp/4065401402)
- OpenAI のモデル一覧
  - https://platform.openai.com/docs/models
- OpenSearch の Docker の設定ファイル
  - https://github.com/codelibs/docker-opensearch
- Langfuse
  - セッションを用いて複数のトレースをまとめる
    - https://langfuse.com/docs/observability/features/sessions?utm_source=chatgpt.com
  - 1 エージェント 1 トレース
    - https://langfuse.com/integrations/model-providers/openai-py?utm_source=chatgpt.com
- sudachi
  - https://github.com/WorksApplications/elasticsearch-sudachi/
  - http://sudachi.s3-website-ap-northeast-1.amazonaws.com/sudachidict/

# チャットボットの検証

## 検証内容

AI エージェントは

- 資料から検索を行い返答するチャットボット
- ユーザーの質問が曖昧の場合は「追い質問」を行う
- ユーザーの質問が資料の中にない情報の場合は「回答なし」という旨の返答を行う
- ~~資料検索の結果によってユーザーに質問をした方がよければユーザーに「追い質問」を行う~~

ということを目指しています

OpenSearch には「キアヌリーブス」についての情報が入っています

キアヌリーブスの情報以外は答えないか確認します

## 検証結果

### ケース 1

質問が「キアヌリーブスの生い立ち」についての質問のため 1 回のやり取りで回答できていることを確認できました

#### チャットボットとのやり取り

ユーザー: キアヌリーブスの生い立ちについて知りたい

チャットボット: キアヌ・リーブスは 1964 年 9 月 2 日にレバノンで生まれ、カナダのトロントで育ちました。彼の父はハワイ出身のアメリカ人地質学者で、母はイングランド人の元ダンサーでデザイナーです。幼少期は世界各地を転々とし、最終的にトロントで育ちました。彼は 5 年間で 4 つの高校に通い、アイスホッケーで優れた能力を発揮しましたが、怪我によりその道を断念しました。9 歳で演劇に出演し、1986 年に映画デビューを果たしました。

### ケース 2

ユーザー:

### ケース 3

1 回目の質問が曖昧のためチャットボットは追い質問をしていることを確認。

その後ユーザーから詳細を取得できたのですが資料にない質問だったため「答えられない」という旨の返答をていることが確認できました。

#### チャットボットとのやり取り

ユーザー: 日本について知りたい

チャットボット:日本について具体的にどのような情報を知りたいですか。

ユーザー: 戦国時代の歴史について知りたいです

チャットボット: お答えできません。ご提示の資料の範囲では確認できません。

## 考察

概ね正しい動きであることが確認できました。
