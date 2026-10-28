"""Map the pipeline's team clusters (0/1) to home/away by kit colour.

Each cluster's kit colour is the median Lab colour of the upper body of sampled player crops;
the analyst's kit names (configs/matches.yaml) are looked up in a fixed sRGB palette, and the
assignment with the smaller total Lab distance wins.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from match_tagger.events.schema import Team


@dataclass(frozen=True)
class TeamMapping:
    """Cluster label -> team, the cluster kit colours and the decision margin (Lab units)."""

    mapping: dict[int, Team]
    cluster_lab: dict[int, tuple[float, float, float]]
    margin: float


def rgb_to_lab(rgb: Sequence[float]) -> np.ndarray:
    """Convert one sRGB colour (0-255) to CIE Lab (L 0-100)."""
    pixel = np.array([[rgb]], dtype=np.float32) / 255.0
    return cv2.cvtColor(pixel, cv2.COLOR_RGB2Lab)[0, 0].astype(np.float64)


def parse_kit(name: str, palette: dict[str, tuple[int, int, int]]) -> np.ndarray:
    """Return the Lab colour of a kit name; the name must be a palette key."""
    key = name.strip().lower()
    if key not in palette:
        raise ValueError(f"Unknown kit colour {name!r}; known: {sorted(palette)}")
    return rgb_to_lab(palette[key])


def kit_region(crop: np.ndarray) -> np.ndarray:
    """Upper-body pixels of a BGR player crop: 20-45 % of height, middle 50 % of width."""
    h, w = crop.shape[:2]
    return crop[
        int(0.20 * h) : max(int(0.45 * h), int(0.20 * h) + 1),
        int(0.25 * w) : max(int(0.75 * w), int(0.25 * w) + 1),
    ]


def kit_colour(crops: Sequence[np.ndarray]) -> np.ndarray:
    """Median Lab colour of the kit regions of BGR crops."""
    pixels = np.concatenate([kit_region(c).reshape(-1, 3) for c in crops]).astype(np.float32)
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3) / 255.0, cv2.COLOR_BGR2Lab).reshape(-1, 3)
    return np.median(lab, axis=0).astype(np.float64)


def map_clusters(cluster_lab: dict[int, np.ndarray], kits: dict[Team, np.ndarray]) -> TeamMapping:
    """Assign clusters 0/1 to home/away by minimal total Lab distance."""
    straight = {0: "home", 1: "away"}
    swapped = {0: "away", 1: "home"}

    def cost(m: dict[int, Team]) -> float:
        return sum(float(np.linalg.norm(cluster_lab[c] - kits[t])) for c, t in m.items())

    c_straight, c_swapped = cost(straight), cost(swapped)
    best = straight if c_straight <= c_swapped else swapped
    return TeamMapping(
        mapping=best,
        cluster_lab={c: tuple(round(float(v), 1) for v in lab) for c, lab in cluster_lab.items()},
        margin=abs(c_straight - c_swapped),
    )
