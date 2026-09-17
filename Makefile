.PHONY: help install seed run mcp mcp-http sync demo test eval lint docker clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:  ## create a venv and install the project with dev extras
	python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e ".[dev]"

seed:  ## write the demo dataset to data/mock_jira
	.venv/bin/python scripts/seed_mock_jira.py

run:  ## start the API and demo UI on http://localhost:8000
	.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

mcp:  ## run the MCP server on stdio (for Claude Desktop or another MCP host)
	.venv/bin/python -m app.mcp_server.server

mcp-http:  ## run the MCP server over Streamable HTTP on :8765
	.venv/bin/python -m app.mcp_server.server --http

sync:  ## pull changes from the live source into the local index
	.venv/bin/python scripts/sync.py

demo:  ## run the full end-to-end demo in the terminal
	.venv/bin/python scripts/run_demo.py

test:  ## run the test suite
	.venv/bin/python -m pytest tests/ -q

eval:  ## run the evaluation suite
	.venv/bin/python evaluation/run_eval.py --out evaluation/results/latest.json

lint:  ## ruff check
	.venv/bin/ruff check app tests evaluation scripts

docker:  ## build and start the stack
	docker compose up --build

clean:  ## remove the local database and caches
	rm -f data/app.db data/app.db-wal data/app.db-shm
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
