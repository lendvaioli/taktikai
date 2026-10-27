.DEFAULT_GOAL := help
.PHONY: help install test lint eval run

help:
	@echo "Targets: install | test | lint | eval MATCH=<id> | run STAGE=<stage> VIDEO=<path> [ARGS=...]"

install:
	uv sync --all-extras

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

eval:
	@echo "eval MATCH=$(MATCH): not implemented yet"

run:
	uv run python -m match_tagger.cli run --stage $(STAGE) --video $(VIDEO) $(ARGS)
