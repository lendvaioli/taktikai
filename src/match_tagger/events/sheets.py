"""Team contact sheets: sample crops per cluster, to confirm the kit-colour mapping by eye."""

import unicodedata
from pathlib import Path

import cv2
import numpy as np

from match_tagger.detection.schema import read_detections
from match_tagger.events.teams import TeamMapping
from match_tagger.tracking.teams import crop_box
from match_tagger.video.frames import FrameRef, iter_frames

TILE_W, TILE_H, PER_ROW = 64, 128, 12


def ascii_text(text: str) -> str:
    """Drop accents for OpenCV's Hershey fonts (fehér -> feher)."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def team_sheet(
    video: Path, match_dir: Path, mapping: TeamMapping, kits: dict[str, str], seed: int = 0
) -> np.ndarray:
    """One row of crops per cluster, labelled with the predicted team and its kit name."""
    det = read_detections(match_dir / "detections.parquet")
    players = det[(det.cls == "player") & det.team.notna()]
    rows = []
    for cluster in (0, 1):
        g = players[players.team == cluster]
        sample = g.sample(min(len(g), PER_ROW), random_state=seed).sort_values("frame")
        refs = [
            FrameRef(index=int(r.frame), t_sec=float(r.t_sec))
            for r in sample.drop_duplicates("frame").itertuples()
        ]
        tiles = []
        for ref, image in iter_frames(video, refs):
            for r in sample[sample.frame == ref.index].itertuples():
                crop = crop_box(image, (r.x1, r.y1, r.x2, r.y2))
                if crop is not None:
                    tiles.append(cv2.resize(crop, (TILE_W, TILE_H)))
        tiles += [np.zeros((TILE_H, TILE_W, 3), np.uint8)] * (PER_ROW - len(tiles))
        team = mapping.mapping[cluster]
        label = np.full((28, TILE_W * PER_ROW, 3), 255, np.uint8)
        text = f"cluster {cluster} -> {team} ({ascii_text(kits[team])})"
        cv2.putText(label, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        rows += [label, np.hstack(tiles[:PER_ROW])]
    return np.vstack(rows)
