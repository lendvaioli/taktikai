import itertools

import pytest

from match_tagger.evals.golden import GoldenClip
from match_tagger.evals.tagging import TagAction, apply_action


def _ids():
    counter = itertools.count(1)
    return lambda: f"e{next(counter)}"


def _add(t, type_="pass", team="home"):
    return TagAction(action="add", nonce=f"n{t}", type=type_, team=team, t_sec=t)


def _run(actions):
    clip = GoldenClip(clip_id="c", video="v.mp4", fps=25, duration_sec=30)
    history, new_id = [], _ids()
    for a in actions:
        clip, history = apply_action(clip, a, history, new_id)
    return clip, history


def test_add_sorts_by_time_and_rounds():
    clip, history = _run([_add(9.123), _add(2.0, "shot", "away")])
    assert [(e.id, e.t_sec, e.type, e.team) for e in clip.events] == [
        ("e2", 2.0, "shot", "away"), ("e1", 9.12, "pass", "home"),
    ]  # fmt: skip
    assert history == ["e1", "e2"]


def test_undo_removes_last_added():
    clip, history = _run([_add(1.0), _add(2.0), TagAction(action="undo", nonce="u")])
    assert [e.id for e in clip.events] == ["e1"]
    assert history == ["e1"]


def test_undo_on_empty_is_noop():
    clip, history = _run([TagAction(action="undo", nonce="u")])
    assert clip.events == [] and history == []


def test_delete_by_id_and_undo_skips_deleted():
    clip, history = _run([
        _add(1.0), _add(2.0),
        TagAction(action="delete", nonce="d", id="e2"),
        TagAction(action="undo", nonce="u"),
    ])  # fmt: skip
    assert clip.events == [] and history == []


def test_add_requires_fields():
    with pytest.raises(ValueError):
        _run([TagAction(action="add", nonce="x", type="pass", t_sec=1.0)])
