"""Shared by the tagger and review apps: video + event-list component and the media server."""

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from match_tagger.video.media_server import MediaServer

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
COMPONENT_DIR = Path(__file__).parent / "components" / "video_events"

_video_events = components.declare_component("video_events", path=str(COMPONENT_DIR))


@st.cache_resource
def media_server() -> MediaServer:
    """One local media server per Streamlit process (127.0.0.1, read-only, range requests)."""
    return MediaServer({"raw": RAW_DIR, "processed": PROCESSED_DIR}).start()


def video_url(video: Path) -> str:
    """URL under which the media server streams a video below data/raw or data/processed."""
    return media_server().url_for(video)


def video_events(**kwargs: object) -> dict | None:
    """Render the component; returns the last action message (with a nonce) or None."""
    return _video_events(default=None, **kwargs)
