.PHONY: run lint format

run:
	python3 main.py

lint:
	@echo "---------------------"
	@echo "--> Running Ruff lint checks..."
	uv run ruff check .

format: ## [Quality] Format code with Ruff
	@echo "--> Formatting with Ruff..."
	uv run ruff format .
