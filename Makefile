CONDA_ENV ?= py312
CONDA_RUN := conda run -n $(CONDA_ENV)

.PHONY: setup conda-setup run demo headless test lint format clean help

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  setup          - Alias for conda-setup (recommended)"
	@echo "  conda-setup    - Create and configure conda environment"
	@echo "  run            - Run full pipeline with Webots (activate env first)"
	@echo "  demo           - Run camera-only demo (no Webots)"
	@echo "  headless       - Run headless for 100 frames (no camera window)"
	@echo "  test           - Run pytest"
	@echo "  lint           - Run ruff linter"
	@echo "  format         - Format code with black and ruff"
	@echo "  clean          - Remove artifacts and caches"
	@echo ""
	@echo "Conda environment: $(CONDA_ENV)"
	@echo "Activate with: conda activate $(CONDA_ENV)"

setup: conda-setup

conda-setup:
	bash scripts/setup_conda.sh

run:
	$(CONDA_RUN) python run.py

demo:
	$(CONDA_RUN) python run.py --no-webots

headless:
	$(CONDA_RUN) python run.py --no-webots --no-display --max-frames 100

test:
	$(CONDA_RUN) pytest

lint:
	$(CONDA_RUN) ruff check src tests

format:
	$(CONDA_RUN) black src tests && $(CONDA_RUN) ruff check --fix src tests

clean:
	rm -rf logs __pycache__ .pytest_cache .ruff_cache
	find . -name "__pycache__" -type d -exec rm -rf {} +
