.DEFAULT_GOAL := help
.PHONY: help install test lint eval run

help:
	@echo "Targets: install | test | lint | eval MATCH=<id> | run STAGE=<stage> VIDEO=<path>"

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

eval:
	@echo "eval MATCH=$(MATCH): not implemented yet"

run:
	@echo "run STAGE=$(STAGE) VIDEO=$(VIDEO): not implemented yet"
