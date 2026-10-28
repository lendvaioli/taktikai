import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from match_tagger.clips.plan import ClipPlan, ClipsConfig, ffmpeg_args, plan_clip
from match_tagger.clips.stage import ClipRecord, cut, keyframes, make_clips
from match_tagger.events.schema import Event

CFG = ClipsConfig()  # 5 s before, 3 s after, copy if keyframe <= 0.5 s before
KEYS = [0.0, 10.0, 20.0]  # DFL clips: a keyframe every 10 s


def test_stream_copy_when_a_keyframe_is_close_before_the_window():
    plan = plan_clip(15.3, 30.0, KEYS, CFG)  # window 10.3-18.3, keyframe 10.0 is 0.3 s earlier
    assert plan == ClipPlan(requested_start=10.3, requested_end=18.3, start=10.0, end=18.3,
                            method="copy")  # fmt: skip


def test_reencode_when_the_last_keyframe_is_too_far_back():
    plan = plan_clip(16.2, 30.0, KEYS, CFG)  # window 11.2-19.2, keyframe 10.0 is 1.2 s earlier
    assert (plan.method, plan.start, plan.end) == ("reencode", pytest.approx(11.2), 19.2)


def test_window_is_clipped_to_the_video():
    start_clip = plan_clip(2.0, 30.0, KEYS, CFG)  # 0-5 s, keyframe at 0 -> copy
    end_clip = plan_clip(28.0, 30.0, KEYS, CFG)  # 23-30 s
    assert (start_clip.start, start_clip.end, start_clip.method) == (0.0, 5.0, "copy")
    assert (end_clip.requested_start, end_clip.end) == (23.0, 30.0)


def test_ffmpeg_arguments_are_exact():
    plan = ClipPlan(requested_start=10.3, requested_end=18.3, start=10.0, end=18.3, method="copy")
    src, dst = Path("in.mp4"), Path("out.mp4")
    assert ffmpeg_args(plan, src, dst, CFG, "copy") == [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", "10.000", "-i", "in.mp4",
        "-t", "8.300", "-map", "0:v:0", "-map", "0:a?", "-c", "copy",
        "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", "out.mp4",
    ]  # fmt: skip
    reencode = ffmpeg_args(plan, src, dst, CFG, "reencode")
    assert reencode[reencode.index("-c:v") : reencode.index("-movflags")] == [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac",
    ]  # fmt: skip


def _fake_runner(calls, fail_copy=False):
    def run(args):
        calls.append("copy" if "copy" in args else "reencode")
        if fail_copy and "copy" in args:
            return 1
        Path(args[-1]).write_bytes(b"clip")
        return 0

    return run


def test_failed_stream_copy_falls_back_to_reencode_from_the_requested_start(tmp_path):
    calls = []
    plan = plan_clip(15.3, 30.0, KEYS, CFG)
    method = cut(plan, Path("in.mp4"), tmp_path / "c.mp4", CFG, _fake_runner(calls, True))
    assert (method, calls) == ("reencode", ["copy", "reencode"])


def test_make_clips_is_idempotent(tmp_path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"video")
    events = [Event(id="e1", t_sec=15.3, type="pass", team="home"),
              Event(id="e2", t_sec=16.2, type="ball_loss", team="away")]  # fmt: skip
    calls = []
    records, skipped = make_clips(events, src, tmp_path / "clips", CFG, KEYS, 30.0,
                                  _fake_runner(calls))  # fmt: skip
    assert ([r.method for r in records], skipped) == (["copy", "reencode"], 0)
    previous = {r.event_id: r for r in records}
    again, skipped = make_clips(events, src, tmp_path / "clips", CFG, KEYS, 30.0,
                                _fake_runner(calls), previous)  # fmt: skip
    assert skipped == 2 and len(calls) == 2  # nothing re-cut
    assert all(isinstance(r, ClipRecord) for r in again)


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                    reason="ffmpeg not installed")  # fmt: skip
def test_real_ffmpeg_cuts_a_synthetic_video(tmp_path):
    src = tmp_path / "in.mp4"
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 48))
    for i in range(25 * 12):  # 12 s
        writer.write(np.full((48, 64, 3), i % 255, np.uint8))
    writer.release()
    keys = keyframes(src, "ffprobe")
    assert keys and keys[0] == 0.0
    event = Event(id="e1", t_sec=6.0, type="pass", team="home")
    records, _ = make_clips([event], src, tmp_path / "clips", CFG, keys, 12.0)
    out = tmp_path / "clips" / "e1.mp4"
    cap = cv2.VideoCapture(str(out))
    seconds = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    assert records[0].method in ("copy", "reencode")
    assert seconds == pytest.approx(8.0, abs=0.6)  # 1 s .. 9 s (a copy may start earlier)
