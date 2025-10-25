# Overview

This repository contains the following systems:

- AI Agent System (`apps/ai_agent`)
  - Chatbot AI agent
  - Exposed as an API
- Tuning AI System (`apps/tuning_ai_agent`)
  - Tunes the AI agent
  - Accuracy still needs improvement

## Architecture

![architecture](https://github.com/campbel2525/sample-ai-agent/blob/main/docs/%E6%A7%8B%E6%88%90%E5%9B%B3.png?raw=true)

# AI Agent System

Provides AI agent functionality.

The AI agent is exposed via FastAPI, so you can execute it by calling the API.

You can specify the prompts to be used in the request body, enabling prompt tuning for the AI agent.

Tune the prompts while calling the API repeatedly.

## Main Features

- Core feature: Chatbot AI agent
  - Single agent
  - Plan-and-Execute pattern
- Program directory: `apps/ai_agent`
- Exposed as a FastAPI API
  - See FastAPI Docs for detailed API specifications: http://localhost:8000/docs
  - Request body includes:
    - Prompts required to run the AI agent
    - Information needed to run RAGas
  - Response includes:
    - Information required for tuning
    - Final output of the AI agent
    - Execution trace of the AI agent
    - Langfuse IDs, etc.
    - RAGas results
  - For simplicity, API code is consolidated into a single file (`apps/ai_agent/run_fastapi.py`) without authentication.
- Uses Langfuse, so you can review agent execution logs in the browser
  - https://langfuse.com/
  - Only `answer_relevancy` and `answer_similarity` are returned
- Uses OpenSearch for hybrid search (full-text + vector search)
  - Sample data: [Wikipedia: Keanu Reeves](https://ja.wikipedia.org/wiki/%E3%82%AD%E3%82%A2%E3%83%8C%E3%83%BB%E3%83%AA%E3%83%BC%E3%83%96%E3%82%B9)
  - Since the goal is to verify the agent, text is chunked into 512 characters with 128-character overlap
  - Sample file: `project/data/test_data.txt`

## Notes

Reference (highly recommended book):

- [現場で活用するための AI エージェント実践入門](https://www.amazon.co.jp/%E7%8F%BE%E5%A0%B4%E3%81%A7%E6%B4%BB%E7%94%A8%E3%81%99%E3%82%8B%E3%81%9F%E3%82%81%E3%81%AEAI%E3%82%A8%E3%83%BC%E3%82%B8%E3%82%A7%E3%83%B3%E3%83%88%E5%AE%9F%E8%B7%B5%E5%85%A5%E9%96%80-KS%E6%83%85%E5%A0%B1%E7%A7%91%E5%AD%A6%E5%B0%82%E9%96%80%E6%9B%B8-%E5%A4%AA%E7%94%B0-%E7%9C%9F%E4%BA%BA/dp/4065401402)

# Tuning AI System

This system continuously refines the agent prompts using AI.

Because the AI agent is exposed as an API, the system can call the API, adjust prompts based on results with an LLM, and call the API again.

Current accuracy is not ideal and needs improvement.

## Main Features

- Main feature: Prompt tuning for the AI agent
- Program directory: `apps/tuning_ai_agent`
- Predefine the number of iterations, call the AI agent API that many times, and tune prompts based on results

# Tech Stack

- Python
- OpenSearch (https://opensearch.org/)
- LangChain (https://www.langchain.com/)
- LangGraph (https://www.langchain.com/langgraph)
- Langfuse (https://langfuse.com/)
- Docker
- OpenAI API

# Setup

1. Configure `.env`

- Create `apps/ai_agent/.env` from `apps/ai_agent/.env.example.example`
- Create `apps/tuning_ai_agent/.env` from `apps/tuning_ai_agent/.env.example`

2. Create Docker environment

```
make init
```

3. Verify

Open the following in your browser and confirm the pages load:

- FastAPI Swagger
  - http://localhost:8000/docs
- Langfuse console
  - http://localhost:3000/
  - ID: `admin@example.com`
  - Password: `secret1234!`
- OpenSearch console
  - http://localhost:5601/

3. Set up data in OpenSearch

```
docker compose -f "./docker/local/docker-compose.yml" -p chatbot-ai-agent exec -it ai-agent pipenv run python scripts/setup.py
```

This chunks the text in `data/insert_data/test_data.txt` into 512 characters with 128-character overlap and inserts it into an OpenSearch index.

4. Start the AI Agent API

```
make ai-agent-run
```

Testing the AI Agent API:

Open http://localhost:8000/docs and execute the `Exec Chatbot Ai Agent` endpoint. If you get a valid response, it is working.

Example request body:

```json
{
  "question": "キアヌ・リーブスの代表作と彼の人柄について教えてください",
  "chat_history": [
    { "role": "user", "content": "映画マトリックスの主人公は？" },
    { "role": "assistant", "content": "キアヌリーブスです" }
  ],
  "planner_model_name": "gpt-4o-2024-08-06",
  "subtask_tool_selection_model_name": "gpt-4o-2024-08-06",
  "subtask_answer_model_name": "gpt-4o-2024-08-06",
  "subtask_reflection_model_name": "gpt-4o-2024-08-06",
  "final_answer_model_name": "gpt-4o-2024-08-06",
  "planner_params": null,
  "subtask_tool_selection_params": null,
  "subtask_answer_params": null,
  "subtask_reflection_params": null,
  "final_answer_params": null,
  "ai_agent_planner_system_prompt": null,
  "ai_agent_planner_user_prompt": null,
  "ai_agent_subtask_system_prompt": null,
  "ai_agent_subtask_tool_selection_user_prompt": null,
  "ai_agent_subtask_reflection_user_prompt": null,
  "ai_agent_subtask_retry_answer_user_prompt": null,
  "ai_agent_create_last_answer_system_prompt": null,
  "ai_agent_create_last_answer_user_prompt": null,
  "is_run_ragas": true,
  "ragas_reference": "キアヌ・リーブスの代表作には『スピード』（1994年）、『マトリックス』シリーズ（1999年〜）、『ジョン・ウィック』シリーズ（2014年〜）があります。彼は「聖人」と呼ばれるほどの人格者として知られ、映画の報酬の大部分を慈善事業に寄付するなど、その優しい人柄でも有名です。特に『マトリックス』の報酬の70％をガン研究に寄付したエピソードは広く知られています。"
}
```

# How to Tune with the Tuning AI

Accuracy still needs improvement.

The goal is to tune prompts used by the AI agent.

Change prompts and run the AI agent API repeatedly, tuning prompts based on results.

## Steps

1. Prepare multiple question/answer pairs. Save to `data/test_data/test_data.yml`.
2. Use `data/test_data/initial_prompt.yml` as the initial prompt.
3. Start the AI agent API:

```
make ai-agent-run
```

4. Run tuning:

```
docker compose -f "./docker/local/docker-compose.yml" -p chatbot-ai-agent exec -it tuning-ai-agent pipenv run python scripts/tuning.py
```

5. Results are output to `data/tuning_result`. See `data/tuning_result/0sample` for the folder structure.

# Future Work / Ideas

- Add functions to modify the agent’s own code
  - Use Cline?
  - Use Claude Code?
- Add memory to the AI agent
  - If successful results are saved, performance can improve with usage
