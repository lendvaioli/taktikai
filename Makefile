.DEFAULT_GOAL := help
.PHONY: help install test lint eval eval-compare tag run

help:
	@echo "Targets: install | test | lint | eval [SET=<golden set>] | eval-compare A=<file> B=<file>"
	@echo "         tag | run STAGE=<stage> VIDEO=<path> [ARGS=...]"

install:
	uv sync --all-extras

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

eval:
	uv run python -m match_tagger.evals.cli eval $(if $(SET),--set $(SET))

eval-compare:
	uv run python -m match_tagger.evals.cli compare $(A) $(B)

tag:
	uv run streamlit run apps/tagger.py

run:
	uv run python -m match_tagger.cli run --stage $(STAGE) --video $(VIDEO) $(ARGS)
