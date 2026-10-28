"""Human scores of generated reports (rubric: docs/REPORT_RUBRIC.md).

Scores are appended to evals/results/report_scores.jsonl, one line per scored report; a later
line by the same scorer for the same run replaces the earlier one in summaries (the file itself
is never rewritten). Terminal prompts are in Hungarian.

    python -m match_tagger.report.score list
    python -m match_tagger.report.score add <run_id> [--usefulness 4 --clarity 3 --terms 5]
    python -m match_tagger.report.score summary
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, Field

from match_tagger.config import CONFIG_DIR, DefaultConfig, load_config
from match_tagger.evals.run import EvalConfig, git_commit
from match_tagger.report.runs import RunRecord, all_records

SCORES_FILE = "report_scores.jsonl"
CRITERIA = {
    "usefulness": "Hasznosság edzőnek",
    "clarity": "Érthetőség",
    "terms": "Szakszerű futballkifejezések",
}
Score = int


class ReportScore(BaseModel):
    """One analyst's scores for one report run."""

    scored_at: datetime
    scorer: str = "Oli"
    run_id: str
    match_id: str
    backend: str
    model: str
    prompt_version: str | None
    prompt_sha256: str
    usefulness: Score = Field(ge=1, le=5)
    clarity: Score = Field(ge=1, le=5)
    terms: Score = Field(ge=1, le=5)
    comment: str = ""
    git_commit: str | None = None


class SummaryRow(BaseModel):
    """Acceptance and mean scores of one backend, model and prompt version."""

    backend: str
    model: str
    prompt_version: str | None
    runs: int
    accepted: int
    scored: int
    usefulness: float | None
    clarity: float | None
    terms: float | None


def read_scores(path: Path) -> list[ReportScore]:
    """All score lines (empty if the file does not exist)."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ReportScore.model_validate_json(x) for x in lines if x.strip()]


def latest(scores: Sequence[ReportScore]) -> list[ReportScore]:
    """The last score per (run, scorer)."""
    by_key = {(s.run_id, s.scorer): s for s in scores}
    return list(by_key.values())


def append_score(path: Path, score: ReportScore, replace: bool = False) -> None:
    """Append a score; refuse a second score of the same run by the same scorer unless replace."""
    if not replace and any(
        s.run_id == score.run_id and s.scorer == score.scorer for s in read_scores(path)
    ):
        raise ValueError(f"{score.scorer} already scored {score.run_id}; use --replace")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(score.model_dump_json() + "\n")


def unscored(
    records: Sequence[RunRecord], scores: Sequence[ReportScore], scorer: str
) -> list[RunRecord]:
    """Accepted runs without a score by this scorer."""
    done = {s.run_id for s in scores if s.scorer == scorer}
    return [r for r in records if r.status == "ok" and r.run_id not in done]


def ask_score(ask: Callable[[str], str], label: str) -> int:
    """Ask until the answer is an integer 1-5."""
    while True:
        answer = ask(f"{label} (1-5): ").strip()
        if answer in {"1", "2", "3", "4", "5"}:
            return int(answer)


def summarize(records: Sequence[RunRecord], scores: Sequence[ReportScore]) -> list[SummaryRow]:
    """Per backend, model and prompt version: runs, accepted, scored and mean scores."""
    keys = sorted({(r.backend, r.model, r.prompt_version or "") for r in records})
    current = latest(scores)
    rows = []
    for backend, model, version in keys:
        runs = [
            r
            for r in records
            if (r.backend, r.model, r.prompt_version or "") == (backend, model, version)
        ]
        ids = {r.run_id for r in runs}
        mine = [s for s in current if s.run_id in ids]

        def avg(field: str, scored: list[ReportScore] = mine) -> float | None:
            return round(mean(getattr(s, field) for s in scored), 2) if scored else None

        rows.append(
            SummaryRow(
                backend=backend,
                model=model,
                prompt_version=version or None,
                runs=len(runs),
                accepted=sum(r.status == "ok" for r in runs),
                scored=len(mine),
                usefulness=avg("usefulness"),
                clarity=avg("clarity"),
                terms=avg("terms"),
            )  # fmt: skip
        )
    return rows


def score_run(
    record: RunRecord, args: argparse.Namespace, ask: Callable[[str], str]
) -> ReportScore:
    """Scores from the flags, asking in the terminal for the missing ones."""
    values = {key: getattr(args, key) or ask_score(ask, label) for key, label in CRITERIA.items()}
    comment = args.comment if args.comment is not None else ask("Megjegyzés (nem kötelező): ")
    return ReportScore(
        scored_at=datetime.now(UTC).replace(microsecond=0),
        scorer=args.scorer,
        **record.model_dump(
            include={"run_id", "match_id", "backend", "model", "prompt_version", "prompt_sha256"}
        ),
        **values,
        comment=comment.strip(),
        git_commit=git_commit(),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="report.score")
    parser.add_argument("command", choices=["list", "add", "summary"])
    parser.add_argument("run_id", nargs="?")
    parser.add_argument("--scorer", default="Oli")
    for key in CRITERIA:
        parser.add_argument(f"--{key}", type=int, choices=range(1, 6))
    parser.add_argument("--comment")
    parser.add_argument("--replace", action="store_true")
    return parser


def main(argv: list[str] | None = None, ask: Callable[[str], str] = input) -> int:
    """Command-line entry point (Hungarian output)."""
    args = _parser().parse_args(argv)
    processed = load_config("default", DefaultConfig, CONFIG_DIR).paths.processed_dir
    path = load_config("evals", EvalConfig, CONFIG_DIR).results_dir / SCORES_FILE
    records, scores = all_records(processed), read_scores(path)
    if args.command == "list":
        for r in unscored(records, scores, args.scorer):
            md = processed / r.match_id / "report" / "runs" / (r.markdown or "")
            print(f"{r.run_id}  {r.match_id:<10} {r.backend}/{r.model}  {r.prompt_version}  {md}")
    elif args.command == "summary":
        for row in summarize(records, scores):
            print(row.model_dump_json())
    else:
        record = next((r for r in records if r.run_id == args.run_id and r.status == "ok"), None)
        if record is None:
            print(f"Nincs elfogadott riport ezzel az azonosítóval: {args.run_id}", file=sys.stderr)
            return 1
        append_score(path, score_run(record, args, ask), replace=args.replace)
        print(f"Mentve: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
