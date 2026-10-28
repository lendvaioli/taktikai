"""Publication guard: nothing derived from licensed footage or StatsBomb data leaks into git.

Rules (CLAUDE.md, docs/EXPERIMENTS.md):
- DFL development clips: no clip ids, tagged events, positions or per-clip results in any
  tracked file; such files live in evals/private/ or *.private.yaml / *_private.md (gitignored).
- StatsBomb: analysis only in space/, next to the logo; elsewhere at most the team-name mapping
  and tests of the team-name rule.
- space/ holds only allow-listed files, and every page and chart carries the StatsBomb logo.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DFL_CLIP = re.compile(r"\b(08fd33|0bfacc|121364|2e57b9|573e61)_\d\b")
STATSBOMB_TEAMS = re.compile(r"\b(France|Croatia|Franciaország|Horvátország)\b")
TEAM_NAME_ALLOWED = (
    "configs/team_names_hu.yaml",
    "tests/report/",
    "tests/test_publication.py",
    "space/",
)
PRIVATE = re.compile(r"(^|/)evals/private/|\.private\.yaml$|_private\.md$")
SPACE_ALLOWED = re.compile(r"^space/(index\.html|README\.md|assets/[\w.-]+\.png)$")
LOGO = "hudl-statsbomb-logo"


def tracked() -> list[str]:
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git work tree")
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [p for p in out.splitlines() if (ROOT / p).is_file()]


def _text(path: str) -> str | None:
    try:
        return (ROOT / path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None  # binary


def test_no_private_files_are_tracked():
    assert [p for p in tracked() if PRIVATE.search(p)] == []


def test_no_dfl_clip_ids_or_golden_files():
    leaks = [p for p in tracked() if p.endswith("dfl.json") or DFL_CLIP.search(_text(p) or "")]
    assert leaks == []


def test_no_positions_in_tracked_eval_files():
    leaks = [p for p in tracked() if p.startswith("evals/") and '"x_m"' in (_text(p) or "")]
    assert leaks == []


def test_statsbomb_team_names_only_where_allowed():
    leaks = [
        p
        for p in tracked()
        if not p.startswith(TEAM_NAME_ALLOWED)
        and p != "uv.lock"
        and STATSBOMB_TEAMS.search(_text(p) or "")
    ]
    assert leaks == []


def test_space_holds_only_allowed_files_with_the_logo():
    files = [p for p in tracked() if p.startswith("space/")]
    assert [p for p in files if not SPACE_ALLOWED.match(p)] == []
    if "space/index.html" in files:
        assert LOGO in (ROOT / "space/index.html").read_text(encoding="utf-8")
    charts = [p for p in files if p.startswith("space/assets/chart_")]
    # the build script stamps the logo into every chart and writes the logo name into the PNG
    # metadata; a chart without it was not made by the build script
    unstamped = [c for c in charts if LOGO.encode() not in (ROOT / c).read_bytes()]
    assert unstamped == []
