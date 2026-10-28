"""Manual event tagger (Streamlit, Hungarian UI). Run with ``make tag``.

The only writer of evals/golden/. Clips of golden set ``<set_id>`` are read from
data/raw/<set_id>/*.mp4; each tag is saved immediately to evals/golden/<set_id>.json.
"""

import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st
from video_component import RAW_DIR, video_events, video_url

from match_tagger.config import load_config
from match_tagger.evals.golden import (
    GoldenClip,
    GoldenSet,
    list_golden_sets,
    load_golden_set,
    save_golden_set,
)
from match_tagger.evals.run import EvalConfig
from match_tagger.evals.tagging import TagAction, apply_action
from match_tagger.events.schema import EVENT_TYPES
from match_tagger.video.frames import read_meta

TYPE_LABELS = {
    "pass": "Passz",
    "ball_loss": "Labdavesztés",
    "shot": "Lövés",
    "set_piece": "Pontrúgás",
}
SET_ID_PATTERN = re.compile(r"^[a-z0-9_-]+$")


def choose_set(golden_dir: Path) -> GoldenSet | None:
    """Sidebar: pick an existing golden set or create a new one."""
    existing = list_golden_sets(golden_dir)
    choice = st.sidebar.selectbox("Golden készlet", [*existing, "+ új készlet"])
    if choice != "+ új készlet":
        return edit_source(load_golden_set(golden_dir, choice), golden_dir)
    set_id = st.sidebar.text_input("Készlet azonosító (pl. dfl)").strip()
    source = st.sidebar.text_input("Forrás (honnan származnak a klipek)")
    if not st.sidebar.button("Készlet létrehozása"):
        return None
    if not SET_ID_PATTERN.match(set_id) or not (RAW_DIR / set_id).is_dir():
        st.sidebar.error(f"Érvénytelen azonosító, vagy hiányzik a {RAW_DIR / set_id} mappa.")
        return None
    golden = GoldenSet(set_id=set_id, source=source)
    save_golden_set(golden, golden_dir)
    st.rerun()
    return golden


def edit_source(golden: GoldenSet, golden_dir: Path) -> GoldenSet:
    """Sidebar: edit the source description of an existing set; saved when it changes."""
    source = st.sidebar.text_input(
        "Forrás (honnan származnak a klipek)", golden.source, key=f"source-{golden.set_id}"
    ).strip()
    if source != golden.source:
        golden = golden.model_copy(update={"source": source})
        save_golden_set(golden, golden_dir)
    return golden


def choose_clip(golden: GoldenSet) -> Path | None:
    """Sidebar: pick a clip from data/raw/<set_id>/ and show its tagging status."""
    videos = sorted((RAW_DIR / golden.set_id).glob("*.mp4"))
    if not videos:
        st.warning(f"Nincs videó a {RAW_DIR / golden.set_id} mappában.")
        return None

    # Labels must stay stable: a changing label would reset the selection on every tag.
    choice = st.sidebar.selectbox(
        "Klip", videos, format_func=lambda p: p.stem, key=f"clip-{golden.set_id}"
    )

    def status(path: Path) -> dict[str, str | int]:
        clip = golden.clip(path.stem)
        state = "új" if clip is None else ("kész" if clip.complete else "folyamatban")
        return {"Klip": path.stem, "Állapot": state, "Esemény": len(clip.events) if clip else 0}

    st.sidebar.dataframe([status(p) for p in videos], hide_index=True)
    return choice


def get_clip(golden: GoldenSet, video: Path) -> GoldenClip:
    """Return the golden clip for a video, creating it from the video's metadata if new."""
    clip = golden.clip(video.stem)
    if clip is None:
        meta = read_meta(video)
        clip = GoldenClip(
            clip_id=video.stem,
            video=video.as_posix(),
            fps=meta.fps,
            duration_sec=meta.n_frames / meta.fps,
        )
        golden.clips.append(clip)
    return clip


def store(golden: GoldenSet, clip: GoldenClip, golden_dir: Path) -> None:
    """Replace the clip in the set, stamp it and save the set."""
    clip.updated_at = datetime.now(UTC).replace(microsecond=0)
    golden.clips = [clip if c.clip_id == clip.clip_id else c for c in golden.clips]
    save_golden_set(golden, golden_dir)


def edit_metadata(clip: GoldenClip) -> GoldenClip:
    """Sidebar: kit descriptions, tagger name and the complete flag."""
    st.sidebar.markdown("**Klip adatai**")
    home = st.sidebar.text_input("Hazai mez", clip.kits.get("home", ""), key=f"home-{clip.clip_id}")
    away = st.sidebar.text_input(
        "Vendég mez", clip.kits.get("away", ""), key=f"away-{clip.clip_id}"
    )
    tagged_by = st.sidebar.text_input("Tagelő", clip.tagged_by or "", key=f"by-{clip.clip_id}")
    complete = st.sidebar.checkbox(
        "Klip kész (bekerül az értékelésbe)", clip.complete, key=f"done-{clip.clip_id}"
    )
    kits = {k: v.strip() for k, v in (("home", home), ("away", away)) if v.strip()}
    return clip.model_copy(
        update={"kits": kits, "tagged_by": tagged_by.strip() or None, "complete": complete}
    )


def handle_action(golden: GoldenSet, clip: GoldenClip, value: dict, golden_dir: Path) -> None:
    """Apply a component message once (deduplicated by nonce), save and rerun."""
    seen = st.session_state.setdefault("nonces", set())
    if not value or value.get("nonce") in seen:
        return
    seen.add(value["nonce"])
    history_key = f"history-{golden.set_id}-{clip.clip_id}"
    history = st.session_state.get(history_key, [])
    clip, history = apply_action(
        clip,
        TagAction.model_validate(value),
        history,
        lambda: f"{clip.clip_id}-{uuid.uuid4().hex[:8]}",
    )
    st.session_state[history_key] = history
    store(golden, clip, golden_dir)
    st.rerun()


def summary(clip: GoldenClip) -> None:
    """Counts per event type and team below the player."""
    rows = [
        {
            "Esemény": TYPE_LABELS[t],
            "Hazai": sum(e.type == t and e.team == "home" for e in clip.events),
            "Vendég": sum(e.type == t and e.team == "away" for e in clip.events),
        }
        for t in EVENT_TYPES
    ]
    st.dataframe(rows, hide_index=True)


def main() -> None:
    """Streamlit entry point."""
    st.set_page_config(page_title="TaktikAI – eseménytagelő", layout="wide")
    st.title("Eseménytagelő")
    golden_dir = load_config("evals", EvalConfig).golden_dir
    golden = choose_set(golden_dir)
    video = choose_clip(golden) if golden else None
    if golden is None or video is None:
        return
    clip = get_clip(golden, video)
    edited = edit_metadata(clip)
    if edited.model_dump(exclude={"updated_at"}) != clip.model_dump(exclude={"updated_at"}):
        store(golden, edited, golden_dir)
        st.rerun()  # refresh the sidebar status table
    value = video_events(
        mode="tag",
        video_url=video_url(video),
        fps=clip.fps,
        kits=clip.kits,
        types=list(EVENT_TYPES),
        events=[e.model_dump(include={"id", "t_sec", "type", "team"}) for e in clip.events],
        key=f"tagger-{golden.set_id}-{clip.clip_id}",
    )
    handle_action(golden, clip, value, golden_dir)
    st.caption(f"Mentés: {golden_dir / (golden.set_id + '.json')} · {len(clip.events)} esemény")
    summary(clip)


main()
