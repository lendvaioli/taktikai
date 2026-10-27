"""Runtime download of detector weights into models/ (never committed)."""

from pathlib import Path

from match_tagger.detection.config import WeightsConfig


def ensure_weights(cfg: WeightsConfig, models_dir: Path) -> Path:
    """Return the local weights path, downloading from Google Drive if it is missing."""
    path = models_dir / cfg.filename
    if path.exists():
        return path
    if cfg.gdrive_id is None:
        raise FileNotFoundError(f"{path} is missing and no gdrive_id is configured")
    import gdown

    models_dir.mkdir(parents=True, exist_ok=True)
    if gdown.download(id=cfg.gdrive_id, output=str(path), quiet=False) is None:
        raise RuntimeError(f"Download of {cfg.filename} from Google Drive failed")
    return path
