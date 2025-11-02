# Overview

This repository provides the following systems:

- AI Agent System (`apps/ai_agent`)
  - AI agent that runs a chatbot
  - Exposed as an API
- Tuning AI System (`apps/tuning_ai_agent`)
  - Functionality to tune the AI agent
  - Accuracy is still limited and has room for improvement

## Architecture

<img width="2232" height="1268" alt="image" src="https://github.com/user-attachments/assets/0a42a70b-a508-4cd0-b7ee-81caf80db9f3" />

# AI Agent System

This provides the AI agent functionality.

The AI agent is exposed via FastAPI, so you can run it by calling the API.

You can specify the prompt to execute in the request body, enabling prompt tuning of the AI agent.

Tune the prompts while calling the API.

## Main Features

- Core feature: Chatbot AI agent API
  - Single-agent
  - Plan-and-Execute type
- Program directory: `apps/ai_agent`
- Exposed as a FastAPI API
  - See FastAPI Docs for detailed API specifications: http://localhost:8000/docs
  - Request body
    - Prompt(s) required to run the AI agent
    - Information required to run RAGas
  - Response
    - Full set of information needed for tuning
    - Final output of the AI agent
    - Execution record of the AI agent
    - Langfuse IDs, etc.
    - RAGas results
  - For this project the API is treated as a sub-feature, so it is consolidated into a single file (`apps/ai_agent/run_fastapi.py`) without authentication.
- streamlit-ui
  - UI feature that uses the AI agent API
- Uses Langfuse, allowing you to view AI agent execution logs in the browser
  - https://langfuse.com/
  - Only `answer_relevancy` and `answer_similarity` are returned
- Uses OpenSearch as the search DB to perform hybrid search (full-text + vector search)
  - Uses [Wikipedia: Keanu Reeves](https://ja.wikipedia.org/wiki/%E3%82%AD%E3%82%A2%E3%83%8C%E3%83%BB%E3%83%AA%E3%83%BC%E3%83%96%E3%82%B9) as sample data
  - Since the goal here is to check how the AI agent behaves, we insert chunks of 512 characters with 128-character overlap without particular tuning
  - File: `project/data/test_data.txt`

## Other

The following technical book was used as a reference (highly recommended):

- [現場で活用するための AI エージェント実践入門](https://www.amazon.co.jp/%E7%8F%BE%E5%A0%B4%E3%81%A7%E6%B4%BB%E7%94%A8%E3%81%99%E3%82%8B%E3%81%9F%E3%82%81%E3%81%AEAI%E3%82%A8%E3%83%BC%E3%82%B8%E3%82%A7%E3%83%B3%E3%83%88%E5%AE%9F%E8%B7%B5%E5%85%A5%E9%96%80-KS%E6%83%85%E5%A0%B1%E7%A7%91%E5%AD%A6%E5%B0%82%E9%96%80%E6%9B%B8-%E5%A4%AA%E7%94%B0-%E7%9C%9F%E4%BA%BA/dp/4065401402)

# Tuning AI System

This provides functionality for AI to continuously refine the agent’s prompts.

Since the AI agent is exposed as an API, you can call the API, have an LLM adjust the prompts based on the results, and call the API again.

Accuracy is not very good.

## Main Features

- Core feature: Prompt tuning for the AI agent
- Program directory: `apps/tuning_ai_agent`
- Predefine the number of iterations and call the AI agent API that many times to tune the prompts

# Tech Stack

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

# Setup

1. Configure `.env`

- Create `apps/ai_agent/.env` from `apps/ai_agent/.env.example.example`
- Create `apps/tuning_ai_agent/.env` from `apps/tuning_ai_agent/.env.example`

2. Create the Docker environment

```
make init
```

3. Verify operation

Open the following URLs in your browser and confirm the screens display:

- FastAPI Swagger UI
  - http://localhost:8000/docs
- Langfuse console
  - http://localhost:3000/
  - ID: `admin@example.com`
  - Password: `secret1234!`
- OpenSearch console
  - http://localhost:5601/

3. Set up data in OpenSearch

```
make opensearch-setup
```

The content about Keanu Reeves (the text in `data/insert_data/test_data.txt`) is chunked into 512 characters with 128-character overlap and inserted into an OpenSearch index.

4. Start the AI Agent API

```
make ai-agent-run
```

How to test the AI Agent API

Open http://localhost:8000/docs and execute the `Exec Chatbot Ai Agent` API. If an appropriate response is returned, it is OK.

Specify the following in the request body:

```json
{
  "query": "キアヌ・リーブスの代表作と彼の人柄について教えてください",
  "chat_history": [
    {
      "role": "user",
      "content": "映画マトリックスの主人公は？"
    },
    {
      "role": "assistant",
      "content": "キアヌリーブスです"
    }
  ],
  "planner_model_name": "gpt-4o-2024-08-06",
  "subtask_tool_selection_model_name": "gpt-4o-2024-08-06",
  "subtask_answer_model_name": "gpt-4o-2024-08-06",
  "subtask_reflection_model_name": "gpt-4o-2024-08-06",
  "create_last_answer_model_name": "gpt-4o-2024-08-06",
  "planner_params": null,
  "subtask_tool_selection_model_params": null,
  "subtask_answer_model_params": null,
  "subtask_reflection_model_params": null,
  "create_last_answer_model_params": null,
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

5. Start streamlit

```
make streamlit-ui-run
```

Open http://localhost:8501/ and confirm the screen displays.

# How to Tune Using the Tuning AI

**Accuracy still needs improvement.**

We envisage tuning the prompts used by the AI agent.

Change the prompts while running the AI agent API and tune them based on the results.

## Steps

1. Prepare multiple query-and-answer pairs. Save file name: `data/test_data/test_data.yml`
2. Use `data/test_data/initial_prompt.yml` as the initial prompt
3. Start the AI agent API.

```
make ai-agent-run
```

4. Run tuning

```
docker compose -f "./docker/local/docker-compose.yml" -p chatbot-ai-agent exec -it tuning-ai-agent pipenv run python scripts/tuning.py
```

5. Results are output to `data/tuning_result`. For the output folder structure, see `data/tuning_result/0sample`.

# Future Work / Ideas

- There is no functionality to modify the agent’s own program; we would like to introduce something.
  - Use Cline?
  - Use Claude Code?
- It would be good for the AI agent to have memory
  - If successful results are saved, performance can reportedly improve the more it is used
- It may be good to change the mechanism of the AI agent itself
  - Although it is a chatbot assuming a conversational form, performing Deep Research over a set of prepared files might also be good
- Consider comparing LLM models
  - Especially for planning LLMs, would like to use GPT-5 or a reasoning mode
- The Tuning AI has poor accuracy and should be improved
  - We are not actually using it for prompt tuning at the moment
    - Currently development is proceeding using Codex
  - The prompts executed by the Tuning AI are not good
    - The prompt does not include information about what kind of AI agent it is
    - It would be good to make this injectable from outside
  - Create a test folder and pass it as an argument when running the command?

# References

- [現場で活用するための AI エージェント実践入門](https://www.amazon.co.jp/%E7%8F%BE%E5%A0%B4%E3%81%A7%E6%B4%BB%E7%94%A8%E3%81%99%E3%82%8B%E3%81%9F%E3%82%81%E3%81%AEAI%E3%82%A8%E3%83%BC%E3%82%B8%E3%82%A7%E3%83%B3%E3%83%88%E5%AE%9F%E8%B7%B5%E5%85%A5%E9%96%80-KS%E6%83%85%E5%A0%B1%E7%A7%91%E5%AD%A6%E5%B0%82%E9%96%80%E6%9B%B8-%E5%A4%AA%E7%94%B0-%E7%9C%9F%E4%BA%BA/dp/4065401402)
- OpenAI model list
  - https://platform.openai.com/docs/models
- OpenSearch Docker configuration files
  - https://github.com/codelibs/docker-opensearch
- Langfuse
  - Group multiple traces using sessions
    - https://langfuse.com/docs/observability/features/sessions?utm_source=chatgpt.com
  - One agent per trace
    - https://langfuse.com/integrations/model-providers/openai-py?utm_source=chatgpt.com
- sudachi
  - https://github.com/WorksApplications/elasticsearch-sudachi/
  - http://sudachi.s3-website-ap-northeast-1.amazonaws.com/sudachidict/

# Validation

## What We Verified

The AI agent aims to:

- Act as a chatbot that searches OpenSearch and responds
- Ask “follow-up querys” when the user’s query is ambiguous
- Respond that there is “no answer” when the user’s query is not present in OpenSearch

OpenSearch contains information about “Keanu Reeves”.

We check that it does not answer beyond the information about Keanu Reeves.

## Results

### Case 1

Since the query was about “the early life of Keanu Reeves,” we confirmed it could answer in a single interaction.

#### Conversation with the chatbot

User: I’d like to know about Keanu Reeves’s early life.

Chatbot: Keanu Reeves was born on September 2, 1964 in Lebanon and grew up in Toronto, Canada. His father was an American geologist from Hawaii, and his mother was an English former dancer and designer. In his childhood he moved around the world with his mother’s remarried partners and grew up mainly in Toronto. He attended four high schools over five years, but ultimately did not graduate from high school.

### Case 2

Because the first query was ambiguous, the chatbot asked a follow-up query.

Afterward the user provided details, but because it was not in the materials, we confirmed it responded that it could not answer.

#### Conversation with the chatbot

User: I want to know about Japan.

Chatbot: What specific information about Japan would you like to know?

User: I’d like to know about the history of the Sengoku period.

Chatbot: I cannot answer. This cannot be confirmed within the provided materials.

## Discussion

We confirmed that the behavior is generally correct.
