# Overview

## [Japanese README](https://github.com/campbel2525/sample-ai-agent/blob/main/README.md)

I built the following systems:

- Chatbot AI Agent System (`apps/ai_agent`)

  - AI agent functionality to run a chatbot

    - A chatbot that searches documents and answers
    - If the user's question is ambiguous, ask a follow-up question
    - ~~If document retrieval suggests it would help, ask a follow-up question~~
    - If the user's question requires information not found in the documents, respond that no answer is available

  - Exposed as an API

- Tuning AI System (`apps/tuning_ai_agent`)
  - Functionality to tune the AI agent
  - Accuracy is still limited and there is room for improvement.

## Architecture

<img width="2232" height="1268" alt="image" src="https://github.com/user-attachments/assets/0a42a70b-a508-4cd0-b7ee-81caf80db9f3" />

# AI Agent System

This provides the AI agent functionality.

The AI agent is exposed via FastAPI, so you can run it by calling the API.

You can specify the prompt to execute in the request body, enabling prompt tuning of the AI agent.

Tune the prompts while calling the API.

## Main Features

- Core feature: Chatbot AI agent API
  - Single agent
  - Plan-and-Execute type
- Program directory: `apps/ai_agent`
- Exposed via FastAPI
  - See the FastAPI Docs for detailed API specifications: http://localhost:8000/docs
  - Request body
    - Prompt(s) required to run the AI agent
    - Information required to run RAGas
  - Response
    - Full set of information needed for tuning
    - Final output of the AI agent
    - Execution record of the AI agent
    - Langfuse IDs, etc.
    - RAGas results
  - In this project the API is positioned as a sub-feature, so the code is consolidated into a single file (`apps/ai_agent/run_fastapi.py`) without authentication.
- streamlit-ui
  - UI that uses the AI agent API
- Uses Langfuse, allowing you to view AI agent execution logs in the browser
  - https://langfuse.com/
  - Returns only `answer_relevancy` and `answer_similarity`
- Uses OpenSearch as the search DB to run hybrid search (full-text + vector)
  - Uses [Wikipedia: Keanu Reeves](https://ja.wikipedia.org/wiki/%E3%82%AD%E3%82%A2%E3%83%8C%E3%83%BB%E3%83%AA%E3%83%BC%E3%83%96%E3%82%B9) as sample data
  - Since the goal here is to check how the AI agent behaves, chunks of 512 characters with 128-character overlap are inserted without particular tuning
  - File: `project/data/test_data.txt`

## Others

The following technical book was used as a reference (highly recommended).

- We referenced [現場で活用するための AI エージェント実践入門](https://www.amazon.co.jp/%E7%8F%BE%E5%A0%B4%E3%81%A7%E6%B4%BB%E7%94%A8%E3%81%99%E3%82%8B%E3%81%9F%E3%82%81%E3%81%AEAI%E3%82%A8%E3%83%BC%E3%82%B8%E3%82%A7%E3%83%B3%E3%83%88%E5%AE%9F%E8%B7%B5%E5%85%A5%E9%96%80-KS%E6%83%85%E5%A0%B1%E7%A7%91%E5%AD%A6%E5%B0%82%E9%96%80%E6%9B%B8-%E5%A4%AA%E7%94%B0-%E7%9C%9F%E4%BA%BA/dp/4065401402) for AI agent development.

# Tuning AI System

This provides functionality for the AI to continuously refine the agent’s prompts.

Because the AI agent is exposed as an API, you can call the API, have an LLM adjust the prompts based on the results, and call the API again.

Accuracy is not very good.

## Main Features

- Core feature: Prompt tuning for the AI agent
- Program directory: `apps/tuning_ai_agent`
- Predefine the number of iterations and call the AI agent API that many times to tune the prompts.

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

Data will be inserted into OpenSearch:

- Content about Keanu Reeves (text in `data/insert_data/test_data.txt`) is chunked into 512 characters with 128-character overlap and inserted into an OpenSearch index.

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

5. Start the AI Agent API

```
make ai-agent-run
```

How to test the AI Agent API

Open http://localhost:8000/docs and execute the `Exec Chatbot Ai Agent` API. If an appropriate response is returned, it is OK.

Specify the following in the request body:

```json
{
  "query": "Please tell me how to read a file in Python.",
  "chat_history": [
    {
      "content": "Tell me about Python.",
      "role": "user",
      "timestamp": "2025-01-17T10:00:00Z"
    },
    {
      "content": "Python is a general-purpose programming language.",
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
  "planner_system_prompt": "You are an excellent planner. Analyze the user's question and break it down into appropriate subtasks.",
  "planner_user_prompt": "Question: {query}\n\nCreate the subtasks necessary to answer the above question.",
  "subtask_tool_selection_system_prompt": "You are a specialist who executes the given subtask. Use the available tools to complete the task.",
  "subtask_tool_selection_user_prompt": "Subtask: {subtask}\n\nSelect the best tool to execute the above subtask and run it.",
  "subtask_reflection_user_prompt": "Subtask: {subtask}\nTool execution result: {tool_result}\n\nEvaluate whether the above result meets the subtask requirements.",
  "subtask_reflection_model_name": "gpt-4o-2024-08-06",
  "subtask_retry_answer_user_prompt": "The previous attempt was insufficient. Advice: {advice}\n\nRe-run the subtask with an improved approach.",
  "subtask_retry_answer_model_name": "gpt-4o-2024-08-06",
  "final_answer_system_prompt": "You are an expert who integrates the results of all subtasks and composes the final answer to the user's question.",
  "final_answer_user_prompt": "Question: {query}\nSubtask results: {subtask_results}\n\nBased on the above information, produce a comprehensive and easy-to-understand answer to the question.",
  "is_run_ragas": true,
  "ragas_reference": "To read a file in Python, use the open() function, preferably combined with a with statement."
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

1. Prepare multiple question–answer pairs. Save file name: `data/test_data/test_data.yml`
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
- The AI agent would benefit from memory functionality
  - If successful outcomes are saved, performance can reportedly improve the more it is used
- It may be good to change the mechanism of the AI agent itself
  - While this assumes a conversational chatbot, it might also be good to perform deep research over a set of self-prepared files
- Consider comparing LLM models
  - Especially for planning, we may want to try models like GPT‑5 or “reasoning” modes
- Improve the Tuning AI as accuracy is currently poor
  - Not actually used for prompt tuning yet
    - Currently development is progressing using Codex
  - The prompts run by the Tuning AI are not good
    - The prompt lacks information about what kind of AI agent it is
    - It would be better to allow injecting this information externally
  - Create a dedicated tests folder and pass it as an argument when running the command?

# References

- [現場で活用するための AI エージェント実践入門](https://www.amazon.co.jp/%E7%8F%BE%E5%A0%B4%E3%81%A7%E6%B4%BB%E7%94%A8%E3%81%99%E3%82%8B%E3%81%9F%E3%82%81%E3%81%AEAI%E3%82%A8%E3%83%BC%E3%82%B8%E3%82%A7%E3%83%B3%E3%83%88%E5%AE%9F%E8%B7%B5%E5%85%A5%E9%96%80-KS%E6%83%85%E5%A0%B1%E7%A7%91%E5%AD%A6%E5%B0%82%E9%96%80%E6%9B%B8-%E5%A4%AA%E7%94%B0-%E7%9C%9F%E4%BA%BA/dp/4065401402)
- OpenAI model list
  - https://platform.openai.com/docs/models
- OpenSearch Docker configuration files
  - https://github.com/codelibs/docker-opensearch
- Langfuse
  - Group multiple traces with sessions
    - https://langfuse.com/docs/observability/features/sessions?utm_source=chatgpt.com
  - One agent, one trace
    - https://langfuse.com/integrations/model-providers/openai-py?utm_source=chatgpt.com
- sudachi
  - https://github.com/WorksApplications/elasticsearch-sudachi/
  - http://sudachi.s3-website-ap-northeast-1.amazonaws.com/sudachidict/

# Chatbot Verification

## What We Verified

The AI agent aims to:

- A chatbot that searches documents and answers
- If the user's question is ambiguous, ask a follow-up question
- If the user's question requires information not found in the documents, respond that no answer is available
- ~~If document retrieval suggests it would help, ask a follow-up question~~

OpenSearch contains information about "Keanu Reeves".

We verify that it does not answer questions outside that scope.

```
Model used: gpt-4o-2024-08-06
Parameters: {"temperature": 0, "seed": 0}
```

## Results

### Case 1

Since the first input already allowed an answer, we confirmed it can answer in a single turn.

#### Chat Transcript

User: Please tell me about Keanu Reeves' background.

Chatbot:

```
Here is an overview of Keanu Reeves' background.

Birthplace and date: Keanu Reeves was born on September 2, 1964, in Lebanon.

Family: His father is an American geologist from Hawaii, and his paternal grandmother is of Chinese-Hawaiian descent. His mother is an English former dancer and designer. Keanu has a sister, Kim. He also has a half-sister, Karina Miller, and a half-sister, Emma Rose Reeves.

Childhood environment: His childhood was unstable, moving around the world and living with his mother's successive spouses. After his parents divorced, his mother worked as a costume designer and moved to Australia and New York City. The family later relocated to Toronto, followed by another divorce. Keanu and his sister Kim were raised by their grandparents and a nanny.

Education: He grew up in Toronto and attended four high schools over five years. He struggled academically due to dyslexia but developed a strong interest in theater. He ultimately did not complete high school.

Early career and interests: He became interested in acting at age nine and was drawn to Shakespeare and other plays. He also excelled as an ice hockey goaltender but gave it up due to injury. The 1989 film "Bill & Ted's Excellent Adventure," in which he played a leading role, was a hit and launched his career as an actor. He also played bass in the grunge rock band Dogstar.

These points show that Keanu Reeves has a diverse background and grew up with strong interests in acting and music.
```

### Case 2

Because the first input was ambiguous, the agent asked a follow-up question, and then answered based on the user's subsequent input. (Due to the knowledge cutoff, the year is treated as 2023.) We also confirmed it referred to prior conversation history.

#### Chat Transcript

User: Keanu

Chatbot: Regarding the name "Keanu," what specifically would you like to know? For example, are you looking for information on Keanu Reeves' background and activities, or something else?

User: Tell me his age.

Chatbot: Keanu Reeves was born on September 2, 1964. Assuming the current year is 2023, his age is 59.

### Case 3

Because the information was not found in the documents, the agent replied that it could not answer.

#### Chat Transcript

User: Please summarize Japan's Sengoku period.

Chatbot: I cannot answer. This is not within the scope of the provided documents.

## Discussion

Overall, the behavior is generally correct:

- A chatbot that searches documents and answers
- If the user's question is ambiguous, ask a follow-up question
- If the user's question requires information not found in the documents, respond that no answer is available
- ~~If document retrieval suggests it would help, ask a follow-up question~~

Determining when to ask a follow-up question was challenging.

As an improvement, as in Case 3, it would be better to return "cannot answer" earlier without digging too deeply. This is tricky because there is often a possibility that asking the user a follow-up could enable an answer.

- OpenSearch Docker configuration
  - https://github.com/codelibs/docker-opensearch
- Langfuse
  - Group multiple traces using sessions
    - https://langfuse.com/docs/observability/features/sessions?utm_source=chatgpt.com
  - One trace per agent
    - https://langfuse.com/integrations/model-providers/openai-py?utm_source=chatgpt.com
- sudachi
  - https://github.com/WorksApplications/elasticsearch-sudachi/
  - http://sudachi.s3-website-ap-northeast-1.amazonaws.com/sudachidict/

# Chatbot Verification

## What We Verify

The AI agent aims to be:

- A chatbot that searches documents and answers
- If the user's question is ambiguous, it asks a follow-up question
- If the user's question requires information not found in the documents, it replies that it cannot answer
- ~~If document retrieval suggests it would help, it asks a follow-up question~~

OpenSearch contains information about “Keanu Reeves”.

We check that it does not answer questions outside Keanu Reeves.

```
Model: gpt-4o-2024-08-06
Parameters: {"temperature": 0, "seed": 0}
```

## Results

### Case 1

The question can be answered on the first input, so it answered in a single turn.

#### Conversation with the chatbot

User: Tell me about Keanu’s background.

Chatbot: Keanu Reeves was born on September 2, 1964 in Lebanon and raised in Toronto, Canada. His father was an American geologist from Hawaii, and his mother is an English former dancer and designer. Keanu grew up surrounded by Chinese art and Chinese cuisine. He attended four high schools in five years and struggled academically due to dyslexia, but he excelled at ice hockey. He developed a strong interest in theater and took the stage at age nine in the play “Damn Yankees.” The 1989 comedy film “Bill & Ted's Excellent Adventure,” in which he starred, became a hit and helped establish his career.

### Case 2

Because the first input is ambiguous, the chatbot asks a follow-up question, and then answers based on the subsequent user input. It also references conversation history.

#### Conversation with the chatbot

User: About Keanu.

Chatbot: What specific information about Keanu Reeves are you looking for? If you have a particular topic such as his background, notable works, or recent activities, please let me know.

User: Tell me his age.

Chatbot: Keanu Reeves was born on September 2, 1964, so as of 2023 he is 59 years old.

### Case 3

Because the first input is ambiguous, the chatbot asks a follow-up question, and then correctly replies that it cannot answer based on the subsequent input.

#### Conversation with the chatbot

User: About Japan.

Chatbot: What specific information about Japan are you looking for? If you can specify a domain such as geography, history, economy, politics, or society and culture, I will look into that.

User: Summarize the Sengoku period.

Chatbot: I cannot answer. It is outside the scope of the provided documents.

## Discussion

We confirmed the behavior is generally correct.

- A chatbot that searches documents and answers
- If the user's question is ambiguous, it asks a follow-up question
- If the user's question requires information not found in the documents, it replies that it cannot answer
- ~~If document retrieval suggests it would help, it asks a follow-up question~~

Determining when to ask follow-up questions is difficult.

As an improvement, like in “Case 3,” it would be nice to return “cannot answer” earlier without digging too deeply. This is tricky because there is often a possibility that asking the user a clarifying question would enable an answer.
  - Especially for planning, we would like to use GPT-5 or a “reasoning mode”
- The Tuning AI has poor accuracy and should be improved
  - We are not actually using it for prompt tuning at the moment
    - Currently, development is proceeding using Codex
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

# Chatbot Validation

## What We Verified

The AI agent aims to:

- Act as a chatbot that searches documents and responds
- Ask follow-up questions when the user’s query is ambiguous
- Respond that there is “no answer” when the user’s question is not present in the documents
- ~~Ask a follow-up question when document retrieval suggests it would help~~

OpenSearch contains information about “Keanu Reeves”.

We verify that it does not answer outside the scope of information about Keanu Reeves.

## Results

### Case 1

We confirmed it could answer in a single interaction because the first input already allowed an answer.

#### Conversation with the chatbot

User: Tell me about Keanu’s early life.

Chatbot: Keanu Reeves was born on September 2, 1964 in Lebanon and grew up in Toronto, Canada. His father is an American geologist from Hawaii, and his mother is an English former dancer and designer. He grew up around Chinese art and Chinese cuisine. He attended four high schools over five years and, due to dyslexia, struggled academically, but he excelled at ice hockey. He developed a strong interest in theater and, at age 9, appeared on stage in the play “Damn Yankees.” In 1989, the comedy “Bill & Ted’s Excellent Adventure,” in which he starred, became a hit and laid the foundation for his career.

### Case 2

We confirmed that because the first input was ambiguous, it asked a follow-up question, and then answered based on the subsequent user input. We also confirmed it referenced previous conversation history.

#### Conversation with the chatbot

User: About Keanu.

Chatbot: What specific information about Keanu Reeves are you looking for? His background, major works, recent activities, etc.? If you have a specific topic, please let me know.

User: Please tell me his age.

Chatbot: Keanu Reeves was born on September 2, 1964, so he is 59 years old as of 2023.

### Case 3

We confirmed that because the first input was ambiguous, it asked a follow-up question and then responded that it could not answer based on the subsequent user input.

#### Conversation with the chatbot

User: About Japan.

Chatbot: What specific information about Japan would you like to know? For example, geography, history, economy, politics, society and culture, etc. If you can specify an area, I will look it up.

User: Please summarize the Sengoku period.

Chatbot: I cannot answer. This cannot be confirmed within the provided documents.

## Discussion

We confirmed that the behavior is largely correct.

- Act as a chatbot that searches documents and responds
- Ask follow-up questions when the user’s query is ambiguous
- Respond that there is “no answer” when the user’s question is not present in the documents
- ~~Ask a follow-up question when document retrieval suggests it would help~~

It was difficult to judge when to ask a follow-up question.

As an improvement, like in Case 3, it would be better to return “cannot answer” earlier without probing too deeply. This is quite difficult because there is often a possibility that asking probing questions would make an answer possible.
