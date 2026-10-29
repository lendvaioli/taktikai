# TODO

Planned work that is out of scope for the task that found it. Each item is a separate task.

## Track stability on fragmented clips (own footage)

- **Why:** on the DFL golden set the clips with the most fragmented tracking (many short player
  tracks) were also the weakest for passes (docs/EXPERIMENTS.md; per-clip numbers in the private
  log). Several tracks also sit in the wrong team cluster on the contact sheets.
- **Scope:** measure fragmentation on own footage first; then try re-linking broken tracks (same
  team, end/start close in time and pitch position) and track-level team re-checks.
- **Done when:** player tracks and median track length on own footage are reported, and pass F1
  is compared with and without re-linking.

## Separate hold time before a ball loss (own footage)

- **Why:** with `min_hold_sec` 0 (best for passes), 13 of 20 false ball losses on DFL are
  "ping-pong" (the ball returns within 2 s) and 12 have an opponent possession under 0.4 s.
- **Scope:** add a minimum opponent hold time that only applies to ball_loss; tune it once enough
  golden ball losses exist (DFL has 4).
- **Done when:** ball_loss precision is reported held-out on own footage with and without it.

## Re-check intercepted passes and pass timing on own footage

- **Why:** `intercept_min_dist_m` ended at the edge of the extended grid (12 m) and
  `pass_time_fraction` chose its highest value (0.4) in most folds; on 43 DFL passes further
  tuning would mostly fit noise.
- **Scope:** on own footage, include "no intercepted passes" (intercept rule off) and
  `pass_time_fraction` up to 0.6 in the grid.
- **Done when:** held-out results on own footage decide both values.

## Attack-relative pitch zones (own footage)

- **Why:** the review app's 18 zones are absolute (6 along x 3 across, numbered from the left
  goal line; `src/match_tagger/events/zones.py`). Analysts think in zones relative to the acting
  team's direction of play ("final third", "own box").
- **Needs:** a reliable direction of play per team and half, e.g. from both goalkeepers'
  positions or the teams' mean x over time; on the DFL clips only one keeper is ever visible.
- **Done when:** on own footage the direction of play per team and half is detected and checked
  against the video, and zones can be shown relative to the acting team.

## StatsBomb logo before publishing StatsBomb-based analysis

- **Why:** the StatsBomb Public Data User Agreement requires the StatsBomb logo on "any
  publication of analysis formed from StatsBomb Data" (docs/MODELS.md). Generated reports only
  carry a text credit, and they stay under `data/` (never committed).
- **Scope:** before a StatsBomb-based report or metric appears in a README, portfolio page or
  talk, add the logo from the StatsBomb media pack next to it.
- **Done when:** every published StatsBomb-based artefact shows the logo.

## Pressing proxy and direction of play on own full matches

- **Why:** the PPDA-style proxy divides by the opponent's ball losses, not by the pressing
  team's defensive actions (tackles, interceptions, fouls), which the events stage does not
  detect, so it is not comparable to published PPDA (StatsBomb reference values: see the
  Hugging Face Space). On 30 s clips it is 0, undefined or based on a handful of events. `home_attacks` is entered by hand per clip;
  a full match needs one value per half.
- **Scope:** enter `home_attacks` per half for the first own match (or detect it from goalkeeper
  positions, see "Attack-relative pitch zones"); compare the proxy with a hand count of defensive
  actions over one half.
- **Done when:** proxy and hand-counted PPDA are reported side by side for one own match.

## Live comparison with Claude Haiku 5.5

- **Why:** all report runs so far used the local Qwen3-8B; the Anthropic backend (default model
  `claude-haiku-5-5`) is only tested with a mocked client. Prompt iteration stopped at v3.1
  with the local model accepting few reports (docs/EXPERIMENTS.md), so it is open whether the
  strict checks or the 8B model limit the acceptance rate.
- **Scope:** once there is a budget, run report_system v3.1 with `TAKTIKAI_LLM_BACKEND=anthropic`
  on the same three matches (two DFL clips, one StatsBomb match), 3 runs each for both backends;
  compare acceptance, steps, tokens, cost and latency from runs.jsonl, and rubric scores.
- **Done when:** the comparison table is in docs/EXPERIMENTS.md with the cost of the Haiku runs.

## Score reports with the rubric

- **Status:** rubric (docs/REPORT_RUBRIC.md) and score collection (`make score-reports`) exist;
  rule-based checks cover numbers, number sources, comparison direction and language.
- **Open:** the checks cannot judge relevance or invented links between events ("passes led to
  ball losses") when no number is involved; only human scores show that.
- **Done when:** accepted reports of at least 3 matches are scored per backend and prompt
  version, and the summary is recorded in docs/EXPERIMENTS.md.

## Validate pitch thresholds on panning tactical footage

- **Why:** the pitch thresholds (`kp_conf 0.5`, `reproj_threshold_m 1.0`,
  `max_reproj_error_m 1.0`) and the centred `smooth_window_sec 0.4` were only checked on the
  30 s fixed broadcast clip, which shows 7-13 confident keypoints in every frame and never
  triggers `reused` or `missing`. It cannot discriminate between settings (docs/EXPERIMENTS.md).
- **Input:** own panning tactical-camera recording (coming soon).
- **Scope:** repeat the cached-keypoint comparison from docs/EXPERIMENTS.md on that footage,
  including `max_reuse_sec`; check in the minimap that smoothing does not lag or blur fast pans.
- **Done when:** the thresholds and smoothing window are confirmed or changed with a results
  table in docs/EXPERIMENTS.md, and the `h_quality` shares on that footage are reported.

## Ball-in-flight handling on lofted balls (own footage)

- **Status:** the airborne check exists in three modes (`off`, `fixed`, `calibrated`) and was
  compared on the DFL clips (docs/EXPERIMENTS.md, 2026-10-28): `fixed` is harmful, `calibrated`
  equals `off`, so v1 runs with `off`.
- **Open:** the DFL clips contain few lofted balls. Re-run the comparison on own footage with long
  balls and crosses; switch to `calibrated` if it then helps.

## Use penalty-area time as goalkeeper evidence (after the pitch stage)

- **Why:** the detector splits the visible keeper about evenly between `goalkeeper` and `player`,
  so the class vote needs the provisional `class_vote.goalkeeper_min_share: 0.3` rule (see
  docs/EXPERIMENTS.md). Position on the pitch is independent evidence: a keeper spends most of
  the match in or near his own penalty area. Image coordinates cannot tell this with a panning
  camera; pitch coordinates from `positions.parquet` can.
- **Scope:** combine the class vote with the share of a track's time inside a penalty area, and
  also use pitch coordinates for the goalkeeper's team (own half / own goal) instead of the
  nearest image-space team centroid.
- **Done when:** on a full match with both keepers visible at times, each keeper resolves to
  `goalkeeper` and the two keepers end up in different teams; the 0.3 threshold is either
  validated on that footage or replaced.

## Migrate ByteTrack to the `trackers` package

- **Why:** `sv.ByteTrack` was deprecated in supervision 0.28.0 and removed in 0.31.0. The
  changelog recommends `ByteTrackTracker` from Roboflow's `trackers` package. Until then
  `pyproject.toml` pins `supervision>=0.25,<0.31` and every run prints a `FutureWarning`.
- **Scope:** `src/match_tagger/tracking/tracker.py` (`build_tracker`, `track`) and
  `configs/tracking.yaml`; lift the supervision pin.
- **Before starting:** verify the `trackers` licence and record it in `docs/MODELS.md`; check how
  its lost-track buffer and confirmation parameters map onto `ByteTrackConfig`.
- **Done when:** track ids on the development clip (first 30 s) are at least as stable as with
  `sv.ByteTrack` (compare `n_tracks` in `detection_summary.json`) and the debug video looks
  equivalent.

## Resolved

### Footage-derived data in the public git history (resolved 2026-10-29)

- **What happened:** until 2026-10-28 the GitHub repository was public and its history contained
  data derived from the DFL development clips (correction logs with pitch positions, the golden
  set, per-clip results) and StatsBomb-derived numbers without the logo.
- **Done:** repository made private; the files moved to `evals/private/` (gitignored); history
  rewritten with `git filter-branch` and force-pushed; `tests/test_publication.py` guards against
  a repeat. A backup bundle of the old history is kept outside the repository.
- **Decision:** GitHub still serves the old commits by hash to authenticated users until its own
  clean-up, but GitHub Insights showed zero views and no forks while the repository was public,
  so GitHub Support is not contacted and the repository is not replaced.
