.DEFAULT_GOAL := help
.PHONY: help install test lint eval eval-compare tune team-sheets tag review run statsbomb-list statsbomb

help:
	@echo "Targets: install | test | lint | eval [SET=<golden set>] | eval-compare A=<file> B=<file>"
	@echo "         tune | team-sheets | tag | review"
	@echo "         run STAGE=<stage> VIDEO=<path> | MATCH=<id> [ARGS=...]"
	@echo "         statsbomb-list [COMP=<id> SEASON=<id>] | statsbomb COMP=<id> SEASON=<id> MATCH=<id>"

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

tune:
	uv run python -m match_tagger.evals.cli tune $(if $(LABEL),--label $(LABEL))

team-sheets:
	uv run python -m match_tagger.evals.cli team-sheets

tag:
	uv run streamlit run apps/tagger.py

review:
	uv run streamlit run apps/review.py

run:
	uv run python -m match_tagger.cli run --stage $(STAGE) $(if $(VIDEO),--video $(VIDEO)) $(if $(MATCH),--match $(MATCH)) $(ARGS)

statsbomb-list:
	uv run python -m match_tagger.metrics.statsbomb_fetch $(if $(COMP),matches --competition $(COMP) --season $(SEASON),competitions)

statsbomb:
	uv run python -m match_tagger.metrics.statsbomb_fetch import --competition $(COMP) --season $(SEASON) --match $(MATCH)
