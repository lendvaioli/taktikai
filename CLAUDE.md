# TaktikAI (Python package: match_tagger)

AI pipeline that turns a football club's own match footage into tagged events, video clips and a coaching report. Portfolio project for AI engineer roles; target users are video analysts at Hungarian NB I / NB II clubs.

## Pipeline
Each stage reads and writes files under data/processed/<match>/ and can run on its own:
video -> detection (detections.parquet) -> pitch (positions.parquet) -> events (events.json) -> clips (clips/) -> metrics -> report

## Layout
- src/match_tagger/{video,detection,tracking,pitch,events,clips,metrics,report,evals}
- apps/ Streamlit apps (tagger, review, demo)
- notebooks/ Kaggle/Colab runners for GPU stages
- configs/ YAML thresholds and parameters, one file per stage
- prompts/ LLM prompts, versioned like code
- docs/ MODELS.md (models, datasets, licences), GOLDEN.md, analyses
- tests/ pytest, mirrors src layout

## Commands
make install | make test | make lint | make eval MATCH=<id> | make run STAGE=<stage> VIDEO=<path>

## Rules
- Numbers are computed by code. LLMs only interpret values returned by tools.
- Every stage is idempotent, configured via configs/*.yaml, and writes its output to a file.
- Never commit video, model weights or anything under data/. Never modify data/raw/ or evals/golden/.
- Open-source dependencies only. Record every model and dataset with source URL and licence in docs/MODELS.md. If a licence is unclear, stop and ask.
- Python 3.11, type hints everywhere, English docstrings, functions under ~40 lines, pydantic models for every data contract.
- Tests: pytest, no network, small synthetic fixtures with hand-checkable expected values.
- UI text in Hungarian; code, comments and commit messages in English.

## Workflow for every task
1. Read the relevant code and docs first.
2. Propose a plan: files, functions, data schemas, tests. Wait for my approval.
3. Implement in small, reviewable steps.
4. Run make test and make lint; fix failures before reporting.
5. Finish with a short report: what changed, files touched, how it was verified (with numbers), open questions, suggested next step.

If anything is ambiguous, ask. Never guess IDs, thresholds, API signatures or licences.