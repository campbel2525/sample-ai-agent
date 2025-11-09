export PROJECT_NAME=chatbot-ai-agent
pn = chatbot-ai-agent
pf := "./docker/local/docker-compose.yml"

init: ## 開発作成
	make destroy
	docker compose -f $(pf) -p $(pn) build --no-cache
	docker compose -f $(pf) -p $(pn) down --volumes
	docker compose -f $(pf) -p $(pn) up -d
	docker compose -f $(pf) -p $(pn) exec -it ai-agent pipenv install --dev
	docker compose -f $(pf) -p $(pn) exec -it tuning-ai-agent pipenv install --dev
	docker compose -f $(pf) -p $(pn) exec -it streamlit-ui pipenv install --dev
	make opensearch-setup

opensearch-setup:
	docker compose -f $(pf) -p $(pn) exec -it ai-agent pipenv run python scripts/opensearch_setup.py

up: ## 開発立ち上げ
	docker compose -f $(pf) -p $(pn) up -d

down: ## 開発down
	docker compose -f $(pf) -p $(pn) down

ai-agent-shell: ## dockerのshellに入る
	docker compose -f $(pf) -p $(pn) exec ai-agent bash

tuning-ai-agent-shell: ## dockerのshellに入る
	docker compose -f $(pf) -p $(pn) exec tuning-ai-agent bash

streamlit-ui-shell: ## streamlitのshellに入る
	docker compose -f $(pf) -p $(pn) exec streamlit-ui bash

check: ## コードのフォーマット
# ai-agent
	docker compose -f $(pf) -p $(pn) exec -it ai-agent pipenv run isort .
	docker compose -f $(pf) -p $(pn) exec -it ai-agent pipenv run black .
	docker compose -f $(pf) -p $(pn) exec -it ai-agent pipenv run flake8 .
	docker compose -f $(pf) -p $(pn) exec -it ai-agent pipenv run mypy .
# tuning-ai-agent
	docker compose -f $(pf) -p $(pn) exec -it tuning-ai-agent pipenv run isort .
	docker compose -f $(pf) -p $(pn) exec -it tuning-ai-agent pipenv run black .
	docker compose -f $(pf) -p $(pn) exec -it tuning-ai-agent pipenv run flake8 .
	docker compose -f $(pf) -p $(pn) exec -it tuning-ai-agent pipenv run mypy .

ai-agent-run:
	docker compose -f $(pf) -p $(pn) exec -it ai-agent pipenv run uvicorn run_fastapi:app --reload --host 0.0.0.0 --port 8000

streamlit-ui-run: ## streamlitを起動（コンテナ内コマンド、--server.portに合わせる）
	docker compose -f $(pf) -p $(pn) exec -it streamlit-ui pipenv run streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501

destroy: ## 環境削除
	make down
	docker network ls -qf name=$(pn) | xargs docker network rm
	docker container ls -a -qf name=$(pn) | xargs docker container rm
	docker volume ls -qf name=$(pn) | xargs docker volume rm

push:
	git add .
	git commit -m "Commit at $$(date +'%Y-%m-%d %H:%M:%S')"
	git push origin head


reset-commit: ## mainブランチのコミット履歴を1つにする 使用は控える
	git checkout --orphan new-branch-name
	git add .
	git branch -D main
	git branch -m main
	git commit -m "first commit"
	git push origin -f main
