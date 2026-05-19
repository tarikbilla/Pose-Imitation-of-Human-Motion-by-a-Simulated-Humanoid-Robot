PYTHON ?= python3.11
VENV   ?= .venv
ACT    := source $(VENV)/bin/activate

.PHONY: setup run demo test lint format clean

setup:
	bash scripts/setup_ubuntu.sh

run:
	$(ACT) && python run.py

demo:
	$(ACT) && python run.py --no-webots

headless:
	$(ACT) && python run.py --no-webots --no-display --max-frames 100

test:
	$(ACT) && pytest

lint:
	$(ACT) && ruff check src tests

format:
	$(ACT) && black src tests && ruff check --fix src tests

clean:
	rm -rf logs __pycache__ .pytest_cache .ruff_cache
	find . -name "__pycache__" -type d -exec rm -rf {} +
