"""Download StatsBomb Open Data JSON and import one match as data/processed/sb_<id>/.

Uses only the standard library (statsbombpy states no software licence). Files are cached
under <raw_dir>/statsbomb/ with the repository's own layout and never committed; the data is
for research and analysis only under the StatsBomb Public Data User Agreement (docs/MODELS.md).

    python -m match_tagger.metrics.statsbomb_fetch competitions
    python -m match_tagger.metrics.statsbomb_fetch matches --competition 43 --season 3
    python -m match_tagger.metrics.statsbomb_fetch import --competition 43 --season 3 --match N
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from match_tagger.config import CONFIG_DIR, DefaultConfig, load_config
from match_tagger.events.schema import EventsFile
from match_tagger.metrics.config import TeamNamesConfig
from match_tagger.metrics.statsbomb import Raw, context, convert, nicknames

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


def fetch(rel_path: str, cache_dir: Path) -> list[Raw]:
    """JSON of ``data/<rel_path>`` from the repository, downloaded once into ``cache_dir``."""
    path = cache_dir / rel_path
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{BASE_URL}/{rel_path}", timeout=60) as resp:  # noqa: S310
            path.write_bytes(resp.read())
    return json.loads(path.read_text(encoding="utf-8"))


def import_match(competition: int, season: int, match_id: int, paths: DefaultConfig) -> Path:
    """Write events.json and match.json of one match; return its directory."""
    cache = paths.paths.raw_dir / "statsbomb"
    matches = fetch(f"matches/{competition}/{season}.json", cache)
    match = next((m for m in matches if m["match_id"] == match_id), None)
    if match is None:
        raise ValueError(f"match {match_id} not in competition {competition} season {season}")
    raw_events = fetch(f"events/{match_id}.json", cache)
    names = nicknames(fetch(f"lineups/{match_id}.json", cache))
    teams_hu = load_config("team_names_hu", TeamNamesConfig, CONFIG_DIR).names
    ctx = context(raw_events, match, teams_hu)
    out = paths.paths.processed_dir / ctx.match_id
    out.mkdir(parents=True, exist_ok=True)
    events = EventsFile(clip_id=ctx.match_id, events=convert(raw_events, match, names=names))
    (out / "events.json").write_text(events.model_dump_json(indent=2), encoding="utf-8")
    (out / "match.json").write_text(ctx.model_dump_json(indent=2), encoding="utf-8")
    return out


def _print_competitions(cache: Path) -> None:
    for c in fetch("competitions.json", cache):
        print(
            f"{c['competition_id']:>4} {c['season_id']:>4}  {c['competition_name']} "
            f"{c['season_name']}"
        )


def _print_matches(competition: int, season: int, cache: Path) -> None:
    for m in sorted(
        fetch(f"matches/{competition}/{season}.json", cache), key=lambda m: m["match_date"]
    ):
        print(
            f"{m['match_id']:>8}  {m['match_date']}  {m['home_team']['home_team_name']} "
            f"{m['home_score']}-{m['away_score']} {m['away_team']['away_team_name']}"
            f"  ({(m.get('competition_stage') or {}).get('name', '')})"
        )


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(prog="statsbomb_fetch")
    parser.add_argument("command", choices=["competitions", "matches", "import"])
    parser.add_argument("--competition", type=int)
    parser.add_argument("--season", type=int)
    parser.add_argument("--match", type=int)
    args = parser.parse_args(argv)
    paths = load_config("default", DefaultConfig, CONFIG_DIR)
    cache = paths.paths.raw_dir / "statsbomb"
    if args.command == "competitions":
        _print_competitions(cache)
    elif args.command == "matches":
        _print_matches(args.competition, args.season, cache)
    else:
        print(import_match(args.competition, args.season, args.match, paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
