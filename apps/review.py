"""Review app (Streamlit, Hungarian UI): correct predicted events. Run with ``make review``.

events.json and the video are never modified: every correction is appended to
evals/corrections/<match>.jsonl and replayed over events.json by ``merge``. The export writes
a golden candidate to evals/candidates/<match>.json for manual promotion to evals/golden/.
"""

import uuid
from pathlib import Path

import streamlit as st
from video_component import PROCESSED_DIR, RAW_DIR, video_events, video_url

from match_tagger.clips.plan import ClipsConfig
from match_tagger.clips.stage import load_index
from match_tagger.config import load_config
from match_tagger.evals.golden import GoldenClip
from match_tagger.evals.run import EvalConfig
from match_tagger.events.config import MatchesConfig
from match_tagger.events.corrections import (
    MergeResult,
    append_entry,
    merge,
    read_log,
)
from match_tagger.events.review import (
    ReviewRow,
    correction_for,
    filter_rows,
    find_golden_clip,
    golden_candidate,
    review_rows,
    write_candidate,
)
from match_tagger.events.schema import EventsFile
from match_tagger.pitch.config import PitchConfig
from match_tagger.video.frames import read_meta

REVIEW_TYPES = ["pass", "ball_loss"]  # events v1 scope (docs/GOLDEN.md)
TYPE_LABELS = {
    "pass": "Passz",
    "ball_loss": "Labdavesztés",
    "shot": "Lövés",
    "set_piece": "Pontrúgás",
}
TEAM_LABELS = {"home": "Hazai", "away": "Vendég"}
NO_ZONE = "nincs"


def find_video(match_id: str) -> Path | None:
    """The source video of a match: data/raw/**/<match_id>.mp4."""
    return next(iter(sorted(RAW_DIR.rglob(f"{match_id}.mp4"))), None)


def choose_match() -> tuple[str, Path] | None:
    """Sidebar: matches with an events.json and a source video."""
    matches = sorted(p.parent.name for p in PROCESSED_DIR.glob("*/events.json"))
    matches = [m for m in matches if find_video(m) is not None]
    if not matches:
        st.warning("Nincs olyan mérkőzés, amelyhez events.json és videó is tartozik.")
        return None
    match_id = st.sidebar.selectbox("Mérkőzés / klip", matches, key="match")
    return match_id, find_video(match_id)


def filters(golden: GoldenClip | None) -> dict[str, set | None]:
    """Sidebar filters: type, team, zone and (with a golden clip) TP / FP / FN."""
    st.sidebar.markdown("**Szűrők**")
    types = st.sidebar.multiselect("Típus", REVIEW_TYPES, format_func=TYPE_LABELS.get)
    teams = st.sidebar.multiselect("Csapat", list(TEAM_LABELS), format_func=TEAM_LABELS.get)
    zones = st.sidebar.multiselect(
        "Zóna (1–18: bal kapuvonaltól, alsó oldalvonaltól)", [*range(1, 19), NO_ZONE]
    )
    evals = st.sidebar.multiselect("Értékelés", ["TP", "FP", "FN"]) if golden else []
    return {
        "types": set(types) or None,
        "teams": set(teams) or None,
        "zones": {None if z == NO_ZONE else z for z in zones} or None,
        "evals": set(evals) or None,
    }


def handle(value: dict | None, log: Path, merged: MergeResult, match_id: str) -> None:
    """Apply a component message once (deduplicated by nonce) and rerun."""
    seen = st.session_state.setdefault("nonces", set())
    if not value or value.get("nonce") in seen:
        return
    seen.add(value["nonce"])
    fields = correction_for(value, merged, match_id, read_log(log), lambda: uuid.uuid4().hex[:8])
    if fields is not None:
        append_entry(log, **fields)
    st.rerun()


def export(match_id: str, video: Path, merged: MergeResult, candidates_dir: Path) -> None:
    """Sidebar button: write the merged events as a golden candidate."""
    if not st.sidebar.button("Exportálás golden-jelöltként"):
        return
    meta = read_meta(video)
    kits = load_config("matches", MatchesConfig).matches.get(match_id)
    candidate = golden_candidate(
        merged,
        match_id,
        video.as_posix(),
        meta.fps,
        meta.n_frames / meta.fps,
        kits.kits if kits else {},
    )
    path = write_candidate(candidate, candidates_dir)
    st.sidebar.success(f"Mentve: {path} – a golden készletbe kézzel kell áthelyezni.")


def summary(rows: list[ReviewRow], merged: MergeResult) -> None:
    """Counts of review status and evaluation labels; conflicts as a warning."""
    status = {
        s: sum(r.status == s for r in rows)
        for s in ("unreviewed", "accepted", "corrected", "added")
    }
    labels = {e: sum(r.eval == e for r in rows) for e in ("TP", "FP", "FN")}
    st.caption(
        f"Új: {status['unreviewed']} · elfogadva: {status['accepted']} · javítva: "
        f"{status['corrected']} · hozzáadva: {status['added']}"
        + (
            f" · TP {labels['TP']} · FP {labels['FP']} · FN {labels['FN']}"
            if any(labels.values())
            else ""
        )
    )
    if merged.conflicts:
        st.warning(
            f"{len(merged.conflicts)} javítás nem illeszkedik az events.json-hoz "
            f"(pl. újragenerált események): {[c.seq for c in merged.conflicts]}"
        )


def clip_links(match_id: str) -> None:
    """Expander with the clip library of the match (served by the media server)."""
    index = load_index(PROCESSED_DIR / match_id)
    if index is None:
        return
    with st.expander(f"Klipkönyvtár: {len(index.clips)} klip"):
        for c in index.clips:
            url = video_url(PROCESSED_DIR / match_id / "clips" / c.path)
            st.markdown(
                f"[{c.t_sec:6.2f} s · {TYPE_LABELS.get(c.type, c.type)} · "
                f"{TEAM_LABELS.get(c.team, c.team)}]({url}) ({c.method})"
            )


def main() -> None:
    """Streamlit entry point."""
    st.set_page_config(page_title="TaktikAI – eseményellenőrzés", layout="wide")
    st.title("Eseményellenőrzés")
    chosen = choose_match()
    if chosen is None:
        return
    match_id, video = chosen
    cfg = load_config("evals", EvalConfig)
    log = cfg.corrections_dir / f"{match_id}.jsonl"
    events = EventsFile.model_validate_json(
        (PROCESSED_DIR / match_id / "events.json").read_text(encoding="utf-8")
    )
    merged = merge(events.events, read_log(log))
    golden = find_golden_clip(cfg.golden_dir, match_id)
    dims = load_config("pitch", PitchConfig).dims_for(match_id)
    rows = review_rows(merged, golden, dims, cfg.tolerance_sec)
    shown = filter_rows(rows, **filters(golden))
    export(match_id, video, merged, cfg.candidates_dir)
    clips = load_config("clips", ClipsConfig)
    kits = load_config("matches", MatchesConfig).matches.get(match_id)
    value = video_events(
        mode="review",
        video_url=video_url(video),
        fps=read_meta(video).fps,
        kits=kits.kits if kits else {},
        types=REVIEW_TYPES,
        events=[r.model_dump() for r in shown],
        window={"pre": clips.pre_sec, "post": clips.post_sec},
        key=f"review-{match_id}",
    )
    handle(value, log, merged, match_id)
    summary(rows, merged)
    st.caption(f"Javítási napló: {log} · golden: {'van' if golden else 'nincs'}")
    clip_links(match_id)


main()
