import pytest

from match_tagger.video.frames import (
    VideoConfig,
    VideoMeta,
    processing_frames,
    select_frame_indices,
)


def _meta(fps, n_frames=10_000):
    return VideoMeta(fps=fps, n_frames=n_frames, width=1920, height=1080)


def test_25fps_to_5fps_takes_every_fifth_frame():
    refs = select_frame_indices(_meta(25), VideoConfig(stride_fps=5, duration_sec=1))
    assert [r.index for r in refs] == [0, 5, 10, 15, 20]


def test_timestamps_are_index_over_source_fps():
    refs = select_frame_indices(_meta(25), VideoConfig(stride_fps=5, duration_sec=1))
    assert [r.t_sec for r in refs] == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8])


def test_non_integer_fps_rounds_to_nearest_frame():
    # step = 29.97 / 5 = 5.994 -> 0, 5.994, 11.988, 17.982, 23.976; end = round(29.97) = 30
    refs = select_frame_indices(_meta(29.97), VideoConfig(stride_fps=5, duration_sec=1))
    assert [r.index for r in refs] == [0, 6, 12, 18, 24]


def test_stride_above_source_fps_uses_every_frame():
    refs = select_frame_indices(_meta(10), VideoConfig(stride_fps=30, duration_sec=0.5))
    assert [r.index for r in refs] == [0, 1, 2, 3, 4]


def test_start_and_duration_window():
    refs = select_frame_indices(_meta(25), VideoConfig(stride_fps=5, start_sec=2, duration_sec=0.6))
    assert [r.index for r in refs] == [50, 55, 60]


def test_window_is_clipped_to_video_length():
    refs = select_frame_indices(_meta(25, n_frames=12), VideoConfig(stride_fps=5))
    assert [r.index for r in refs] == [0, 5, 10]


def test_processing_frames_default_equals_output_grid():
    refs, output = processing_frames(_meta(25), VideoConfig(stride_fps=5, duration_sec=1))
    assert [r.index for r in refs] == [0, 5, 10, 15, 20]
    assert output == {0, 5, 10, 15, 20}


def test_processing_frames_dense_grid_includes_output_grid():
    # 12.5 fps on 25 fps -> every 2nd frame; the 5 fps output frames 5 and 15 are added.
    cfg = VideoConfig(stride_fps=5, detect_fps=12.5, duration_sec=1)
    refs, output = processing_frames(_meta(25), cfg)
    assert [r.index for r in refs] == [0, 2, 4, 5, 6, 8, 10, 12, 14, 15, 16, 18, 20, 22, 24]
    assert output == {0, 5, 10, 15, 20}


def test_start_beyond_end_yields_nothing():
    cfg = VideoConfig(stride_fps=5, start_sec=100)
    assert select_frame_indices(_meta(25, n_frames=100), cfg) == []
