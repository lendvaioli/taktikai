"""Unsupervised team assignment, after roboflow/sports.

Player crops are embedded with SigLIP, reduced with UMAP and clustered with KMeans into two
teams. Goalkeepers are assigned to the team whose players' mean position is nearest.
"""

from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, Field

from match_tagger.detection.schema import DetectionRecord


class TeamConfig(BaseModel):
    """Team classifier parameters from configs/tracking.yaml."""

    embed_model: str = "google/siglip-base-patch16-224"
    crops_per_track: int = Field(10, ge=1)
    crop_every: int = Field(5, ge=1)
    batch_size: int = Field(32, ge=1)
    umap_components: int = Field(3, ge=1)
    umap_neighbors: int = Field(15, ge=2)
    seed: int = 0


def majority_vote(labels: Sequence[int]) -> int | None:
    """Return the most frequent label; ties go to the label seen first. Empty -> None."""
    if not labels:
        return None
    counts = Counter(labels)
    best = max(counts.values())
    return next(label for label in labels if counts[label] == best)


def assign_goalkeeper(
    keeper_xy: np.ndarray, player_xy: np.ndarray, player_team: np.ndarray
) -> int | None:
    """Return the team whose players' mean position is closest to the goalkeeper.

    Positions are bottom-centre anchors in pixels. Teams with no players in the frame are
    skipped; with no labelled players at all the result is None.
    """
    best_team, best_dist = None, np.inf
    for team in (0, 1):
        members = player_xy[player_team == team]
        if len(members) == 0:
            continue
        dist = float(np.linalg.norm(keeper_xy - members.mean(axis=0)))
        if dist < best_dist:
            best_team, best_dist = team, dist
    return best_team


def crop_box(image: np.ndarray, xyxy: Sequence[float]) -> np.ndarray | None:
    """Return the image region inside the box, clipped to the image; None if empty."""
    height, width = image.shape[:2]
    x1, y1 = max(int(xyxy[0]), 0), max(int(xyxy[1]), 0)
    x2, y2 = min(int(np.ceil(xyxy[2])), width), min(int(np.ceil(xyxy[3])), height)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2].copy()


class CropCollector:
    """Keeps up to ``crops_per_track`` crops per track, one per ``crop_every`` sightings.

    Crops are collected for every person track; ``subset`` keeps those of the tracks whose
    resolved class is player before the team fit.
    """

    def __init__(self, cfg: TeamConfig) -> None:
        """Start with no crops."""
        self.cfg = cfg
        self.sightings: Counter[int] = Counter()
        self.track_ids: list[int] = []
        self.crops: list[np.ndarray] = []

    def add(self, image: np.ndarray, track_id: int, xyxy: Sequence[float]) -> None:
        """Register one sighting of ``track_id``; store its crop if it is due."""
        seen = self.sightings[track_id]
        self.sightings[track_id] += 1
        if seen % self.cfg.crop_every or seen // self.cfg.crop_every >= self.cfg.crops_per_track:
            return
        crop = crop_box(image, xyxy)
        if crop is not None:
            self.track_ids.append(track_id)
            self.crops.append(crop)

    def subset(self, keep: set[int]) -> tuple[list[int], list[np.ndarray]]:
        """Return the track ids and crops of the tracks in ``keep``."""
        pairs = [(t, c) for t, c in zip(self.track_ids, self.crops, strict=True) if t in keep]
        return [t for t, _ in pairs], [c for _, c in pairs]


def vote_per_track(track_ids: Sequence[int], labels: Sequence[int]) -> dict[int, int]:
    """Collapse per-crop labels into one team per track by majority vote."""
    by_track: dict[int, list[int]] = defaultdict(list)
    for track_id, label in zip(track_ids, labels, strict=True):
        by_track[track_id].append(int(label))
    return {
        tid: team for tid, votes in by_track.items() if (team := majority_vote(votes)) is not None
    }


def _anchor(record: DetectionRecord) -> np.ndarray:
    return np.array([(record.x1 + record.x2) / 2, record.y2])


def _keeper_teams(
    records: Sequence[DetectionRecord], player_team: dict[int, int]
) -> dict[int, int]:
    """Assign each goalkeeper track a team: per-frame nearest centroid, then majority vote."""
    by_frame: dict[int, list[DetectionRecord]] = defaultdict(list)
    for record in records:
        by_frame[record.frame].append(record)
    votes: dict[int, list[int]] = defaultdict(list)
    for frame_records in by_frame.values():
        players = [r for r in frame_records if r.cls == "player" and r.track_id in player_team]
        xy = np.array([_anchor(r) for r in players]).reshape(-1, 2)
        teams = np.array([player_team[r.track_id] for r in players], dtype=int)
        for keeper in (r for r in frame_records if r.cls == "goalkeeper"):
            team = assign_goalkeeper(_anchor(keeper), xy, teams)
            if team is not None:
                votes[keeper.track_id].append(team)
    return {tid: team for tid, v in votes.items() if (team := majority_vote(v)) is not None}


def label_teams(
    records: Sequence[DetectionRecord], player_team: dict[int, int]
) -> list[DetectionRecord]:
    """Return records with ``team`` set for players and goalkeepers; others are unchanged.

    Updated records are re-validated so the schema rules still hold.
    """
    keeper_team = _keeper_teams(records, player_team)
    lookup = {"player": player_team, "goalkeeper": keeper_team}
    return [
        DetectionRecord.model_validate({**r.model_dump(), "team": lookup[r.cls].get(r.track_id)})
        if r.cls in lookup
        else r
        for r in records
    ]


class TeamClassifier:
    """SigLIP embeddings -> UMAP -> KMeans(2). Heavy libraries are imported lazily."""

    def __init__(self, cfg: TeamConfig, device: str) -> None:
        """Load the embedding model onto ``device``."""
        from transformers import AutoImageProcessor, SiglipVisionModel

        self.cfg = cfg
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(cfg.embed_model)
        self.model = SiglipVisionModel.from_pretrained(cfg.embed_model).to(device).eval()

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Return one mean-pooled embedding per BGR crop."""
        import torch

        out = []
        size = self.cfg.batch_size
        for start in range(0, len(crops), size):
            batch = [np.ascontiguousarray(c[:, :, ::-1]) for c in crops[start : start + size]]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                hidden = self.model(**inputs).last_hidden_state
            out.append(hidden.mean(dim=1).cpu().numpy())
        return np.concatenate(out)

    def fit_predict(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Cluster crops into two teams; returns one label (0 or 1) per crop."""
        import umap
        from sklearn.cluster import KMeans

        features = self.embed(crops)
        neighbors = min(self.cfg.umap_neighbors, len(features) - 1)
        reducer = umap.UMAP(
            n_components=self.cfg.umap_components,
            n_neighbors=neighbors,
            random_state=self.cfg.seed,
        )
        projected = reducer.fit_transform(features)
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=self.cfg.seed)
        return kmeans.fit_predict(projected)
