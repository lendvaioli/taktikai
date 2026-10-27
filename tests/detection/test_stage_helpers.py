import pytest

from match_tagger.cli import main
from match_tagger.config import DefaultConfig, load_config
from match_tagger.detection.config import DetectionConfig
from match_tagger.detection.detector import build_detector, class_lookup
from match_tagger.detection.schema import DetectionRecord
from match_tagger.detection.summary import coverage
from match_tagger.tracking.config import TrackingConfig
from match_tagger.video.frames import VideoConfig

# Class order of the roboflow/sports player model.
ROBOFLOW_NAMES = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}


def _rec(frame, cls, track_id=None, team=None):
    return DetectionRecord(
        frame=frame, t_sec=frame / 25, track_id=track_id, cls=cls, team=team,
        x1=0, y1=0, x2=1, y2=1, conf=0.5,
    )  # fmt: skip


def test_class_lookup_maps_model_order_to_pipeline_order():
    lookup = class_lookup(ROBOFLOW_NAMES, DetectionConfig().class_map)
    # pipeline order: player=0, goalkeeper=1, referee=2, ball=3
    assert lookup.tolist() == [3, 1, 0, 2]


def test_class_lookup_rejects_model_without_ball():
    names = {0: "player", 1: "goalkeeper", 2: "referee"}
    with pytest.raises(ValueError, match="ball"):
        class_lookup(names, DetectionConfig().class_map)


def test_ball_model_flag_is_a_stub(tmp_path):
    cfg = DetectionConfig.model_validate({"ball_model": {"enabled": True}})
    with pytest.raises(NotImplementedError):
        build_detector(cfg, tmp_path, "cpu")


def test_coverage_shares():
    records = [
        _rec(0, "player", 1, 0),
        _rec(0, "player", 2, None),
        _rec(5, "player", 1, 0),
        _rec(5, "player", 3, 1),
        _rec(0, "referee", 4),
        _rec(0, "ball"),
    ]
    stats = coverage(records, n_sampled_frames=4)
    assert stats["player_team_share"] == 0.75
    assert stats["ball_frame_share"] == 0.25
    assert stats["n_tracks"] == 4
    assert stats["records_per_class"] == {"player": 4, "referee": 1, "ball": 1}


def test_repo_configs_parse():
    assert load_config("video", VideoConfig).stride_fps == 5.0
    assert load_config("detection", DetectionConfig).conf == 0.3
    assert load_config("tracking", TrackingConfig).bytetrack.minimum_consecutive_frames == 3
    assert str(load_config("default", DefaultConfig).paths.models_dir) == "models"


def test_cli_rejects_unimplemented_stage(tmp_path):
    video = tmp_path / "x.mp4"
    video.write_bytes(b"")
    assert main(["run", "--stage", "events", "--video", str(video)]) == 2
