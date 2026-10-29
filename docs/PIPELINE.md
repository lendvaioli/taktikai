# Pipeline in detail

How each stage works and how to run it. The overview is in the [README](../README.md).

```
video -> detection -> pitch -> events -> clips -> metrics -> report
```

Each stage reads and writes files under `data/processed/<match>/` and can run on its own.

| stage     | output                |
|-----------|-----------------------|
| detection | `detections.parquet`  |
| pitch     | `positions.parquet`   |
| events    | `events.json`         |
| clips     | `clips/`              |
| metrics   | metrics files         |
| report    | coaching report       |

Code lives in `src/match_tagger/`, stage parameters in `configs/`, LLM prompts in `prompts/`,
and model and dataset licences in [docs/MODELS.md](MODELS.md).

## Detection stage

```sh
make run STAGE=detection VIDEO=data/raw/my_clip.mp4                      # whole video
make run STAGE=detection VIDEO=data/raw/my_clip.mp4 ARGS="--duration 30" # first 30 s
```

Runs detection and ByteTrack tracking at 12.5 fps (stable track ids, see
[docs/EXPERIMENTS.md](EXPERIMENTS.md)), writes records at the same 12.5 fps, and splits
players into two teams without supervision. Classes: player, goalkeeper, referee, ball. Writes
`detections.parquet` (`frame, t_sec, track_id, cls, team, x1, y1, x2, y2, conf`),
`detection_summary.json` and an annotated `debug_detection.mp4`. Weights are downloaded into
`models/` on first run. Runs on CPU; for full matches use
[notebooks/run_detection.ipynb](../notebooks/run_detection.ipynb) on a Kaggle or Colab GPU.

The detector weights are for non-commercial use only, pending licence review; see
[docs/MODELS.md](MODELS.md).

## Pitch stage

```sh
make run STAGE=pitch VIDEO=data/raw/my_clip.mp4 ARGS="--duration 30" # needs detections.parquet
```

Detects 32 pitch landmarks per output frame, fits an image-to-pitch homography with RANSAC
(only with at least 4 confident keypoints), smooths it over a centred 0.4 s window and reuses the
last valid one for up to 2 s; beyond that positions are missing, never extrapolated. Projects
the foot point of every person and the centre of the ball onto a pitch in metres (default
105 x 68, origin at the bottom-left corner as seen from the main camera; override the size per
match with `match_sizes` in `configs/pitch.yaml`). Writes `positions.parquet`
(`frame, t_sec, track_id, cls, team, x_m, y_m, h_quality, out_of_pitch`), `pitch_summary.json`
and `minimap.mp4`: footage with keypoints and projected pitch lines on the left, an mplsoccer
minimap on the right (bar on top: green ok, orange reused, red missing homography; hollow dot =
reused, cross = more than 5 m outside the pitch). The ball is projected as if it were on the
ground, so a ball in the air is drawn too far from the camera.

## Events stage (v1)

```sh
make run STAGE=events VIDEO=data/raw/my_clip.mp4   # needs detections, positions, homographies
```

Rule-based **pass** and **ball_loss** detection from possession changes (shot and set piece are out
of scope for now). Team clusters are mapped to home/away by kit colour, using the kit names in
`configs/matches.yaml`; likely-airborne ball positions are ignored for possession; short ball gaps
are interpolated; the nearest player within a radius who holds the ball long enough possesses
it. A possession change within a team is a pass, to the opponent a ball loss. Every event gets a
rule-based confidence. Writes `events.json` (validated `Event` records) and
`events_summary.json`. Thresholds and the tuning grid: [configs/events.yaml](../configs/events.yaml);
`make tune` runs leave-one-clip-out tuning on the `dfl` golden set, `make team-sheets` renders
crops per team cluster to check the kit-colour mapping.

## Clips and review

```sh
make run STAGE=clips VIDEO=data/raw/my_clip.mp4   # one clip per event (needs events.json)
make review                                            # review app
```

The clips stage cuts one clip per predicted event (5 s before, 3 s after; `configs/clips.yaml`)
with FFmpeg: stream copy when a keyframe lies at most 0.5 s before the clip start, otherwise a
fast-seek re-encode; already cut clips are skipped. The review app (Hungarian UI) plays each
event's window from the source video, filters by type, team, 18 absolute pitch zones and, when
a golden clip exists, TP / FP / FN, and corrects with one click per row (accept, delete, change
type, change team) or adds a missing event at the current time. Corrections are an append-only
log in `evals/corrections/<match>.jsonl`, replayed over `events.json` (which is never modified);
the export writes a golden candidate to `evals/candidates/<match>.json`. Videos are streamed by a
local media server on 127.0.0.1 with HTTP range requests, so full-length recordings work too.

## Metrics and coaching report

```sh
make run STAGE=metrics MATCH=my_clip                  # metrics.json from reviewed events
make run STAGE=report MATCH=my_clip                   # report/report.md via the local LLM
make statsbomb COMP=43 SEASON=3 MATCH=8658            # import a StatsBomb Open Data match
make run STAGE=metrics MATCH=sb_8658                  # same metrics on reference data
```

The metrics stage replays the review corrections over `events.json` and computes, per team:
event counts and pass completion, ball losses per zone, a pass network (zone to zone for own
footage, player to player for StatsBomb), a PPDA-style pressing proxy (opponent passes per
opponent ball loss in the opponent's own 60 %) and a shot map (StatsBomb only for now), plus a
sample block (duration, event counts, reliability `very_low` / `low` / `ok`). Zones are
attack-relative (thirds x lanes) when `home_attacks` is set in `configs/matches.yaml`. A
StatsBomb adapter turns StatsBomb events into the same `Event` schema (downloaded with the
standard library; research use only, see [docs/MODELS.md](MODELS.md)).

The report stage is a plain Python agent loop (at most 8 LLM calls) with three read-only tools,
`get_metric`, `get_events` and `get_clips`, and a `submit_report` tool whose schema is the
pydantic `Report` (2-3 findings with claim, numbers, clip ids and a training focus). The
validator rejects a report (with the reasons and the rules; one resubmission allowed) unless:
every number in the text appears in a tool result of the same run (rounding tolerance, decimal
comma and percentages accepted); every supporting number names its source (tool call and field
path) and equals that field exactly, and the rendered report shows the source next to it;
comparisons ("több", "kevesebb", "magasabb", "alacsonyabb" between two numbers) point the right
way; the text contains no English words, metric keys or untranslated team names (Hungarian
names from `configs/team_names_hu.yaml`); and every clip id was returned by `get_clips`.
Accepted reports are scored by hand with the rubric in
[docs/REPORT_RUBRIC.md](REPORT_RUBRIC.md) (`make score-reports`, scores in
`evals/results/report_scores.jsonl`).
The default backend is a local open model (Qwen3-8B Q4_K_M from the Hugging Face Hub, served by
Ollama); `TAKTIKAI_LLM_BACKEND=anthropic` switches to the Anthropic API. The system prompt is
[prompts/report_system.md](../prompts/report_system.md); every run is logged to
`report/runs.jsonl` (tool calls, tokens, latency, validator verdicts, unchecked comparisons) and
kept as `report/runs/<run_id>.json` (+ `.md` if accepted). Keep the PC awake while reports run
locally: a suspended machine pauses the run and the request timeout
(`ollama.timeout_sec` in `configs/report.yaml`) does not fire during sleep.

## Evaluation and tagging

Event quality is measured against hand-tagged golden sets, one per footage source (for example
`dfl` for the five DFL sample clips). `make tag` opens the Streamlit tagger (Hungarian UI: video
player, Passz / Labdavesztés / Lövés / Pontrúgás buttons, keyboard shortcuts, undo, event list);
`make eval [SET=dfl]` matches predicted events to golden ones of the same type and team within
2 s and prints precision, recall and F1 per type and per golden set, writing a result file to
`evals/results/`; `make eval-compare A=... B=...` shows the difference between two runs. Format
and tagging guidelines: [docs/GOLDEN.md](GOLDEN.md).
