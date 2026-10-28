from match_tagger.metrics.config import ReliabilityConfig
from match_tagger.metrics.sample import Sample
from match_tagger.report.agent import AgentSettings, run_agent
from match_tagger.report.backends import OllamaBackend, ollama_timings_ms
from match_tagger.report.config import OllamaConfig
from match_tagger.report.log import RunLog
from match_tagger.report.sample_note import duration_hu, sample_note
from match_tagger.report.schema import Message


def _sample(seconds, n, per_type, reliability):
    return Sample(
        duration_sec=seconds,
        duration_min=round(seconds / 60, 1),
        n_events=n,
        events_per_type=per_type,
        reliability=reliability,
        thresholds=ReliabilityConfig(),
    )


def test_sample_note_for_a_short_clip():
    # a 30 s clip: a model once swapped these two counts in free text; code cannot
    note = sample_note(_sample(30.0, 18, {"ball_loss": 12, "pass": 6}, "very_low"))
    assert note.startswith("30 másodperc, 18 esemény (12 labdavesztés, 6 passz). ")
    assert "Megbízhatóság: nagyon alacsony – rövid részlet" in note


def test_duration_switches_to_minutes():
    assert duration_hu(119.6) == "120 másodperc"
    assert duration_hu(5703.8) == "95,1 perc"
    assert sample_note(_sample(0.0, 0, {}, "very_low")).startswith("0 másodperc, 0 esemény. ")


def test_ollama_timings_are_converted_to_ms():
    resp = {"load_duration": 6_743_466_800, "eval_duration": 205_973_000, "eval_count": 8}
    assert ollama_timings_ms(resp) == {"load_ms": 6743.5, "eval_ms": 206.0}


def test_ollama_reply_carries_timings_and_gpu_share():
    def post(url, body, timeout):
        assert timeout == 300.0  # configured request timeout
        return {"message": {"content": "szia"}, "total_duration": 2_000_000, "eval_count": 1}

    def get(url, timeout):
        assert url.endswith("/api/ps")
        return {
            "models": [{"name": OllamaConfig().model, "size": 7639704533, "size_vram": 6304101826}]
        }

    reply = OllamaBackend(OllamaConfig(), post=post, get=get).chat(
        "s", [Message(role="user", content="u")], []
    )
    assert reply.backend_timings_ms == {"total_ms": 2.0}
    assert reply.gpu_share == 0.825  # the measured partial offload: 6.30 of 7.64 GB


def test_gpu_share_unknown_when_ps_fails_or_model_not_loaded():
    def failing(url, timeout):
        raise OSError("connection refused")

    assert OllamaBackend(OllamaConfig(), get=failing).gpu_share() is None
    assert OllamaBackend(OllamaConfig(), get=lambda u, t: {"models": []}).gpu_share() is None


class _TimingOut:
    name, model = "ollama", "m"

    def chat(self, system, messages, tools):
        raise TimeoutError("timed out")


def test_backend_timeout_ends_the_run_with_error(box, tmp_path):
    log = RunLog(tmp_path / "runs.jsonl", "r1", {})
    result = run_agent(
        _TimingOut(), "s", "u", box, AgentSettings(match_id="m", has_video=True), log
    )
    assert (result.status, result.steps, result.report) == ("error", 1, None)
    assert "TimeoutError" in result.errors[0]
    assert '"status": "error"' in (tmp_path / "runs.jsonl").read_text(encoding="utf-8")
