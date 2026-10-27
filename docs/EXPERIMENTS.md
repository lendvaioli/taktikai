# Experiments

Decisions backed by measurements. Each entry: question, setup, results, decision, caveats.

**Publication rule.** The development footage (DFL Bundesliga clips from roboflow/sports) must
not be published, nor anything derived from it per clip: frames, minimaps, clips, positions,
tagged events, per-clip tables, clip ids or quotes from reports about those clips. This file
therefore keeps only aggregates over the whole golden set; the per-clip measurements behind each
entry are kept in a private log (`docs/EXPERIMENTS_private.md`, gitignored). StatsBomb-derived
numbers appear only in the Hugging Face Space, next to the StatsBomb logo.

## 2026-10-27: Tracking rate and ByteTrack settings for stable track ids

### Question

At a 5 fps processing rate a 30 s broadcast clip produced several times more player tracks than
players on the pitch. Which change makes track ids stable at an acceptable cost?

### Setup

One 30 s DFL clip (fixed broadcast camera, 1080p, 25 fps), CPU only. Four configurations:
processing rate 5 vs 12.5 fps, crossed with ByteTrack defaults vs a looser association
(`minimum_consecutive_frames 1`, `minimum_matching_threshold 0.95`). Metrics: player tracks,
median track length, labelled players per frame, runtime (no ground-truth ids).

### Decision

`detect_fps: 12.5` is the default; ByteTrack stays at its defaults. Processing more frames is
what makes the ids stable (at 5 fps players move far relative to their boxes between frames, so
IoU matching fails); at 12.5 fps the number of player tracks dropped to close to the number of
players and the median track length to close to the clip length, at about 2.4x CPU runtime. The
looser association added only a marginal gain and makes false matches more likely.

### Caveats

One fixed-camera clip; panning footage may favour the looser association. No ground-truth ids:
the metrics measure fragmentation, not correctness. Per-clip numbers: private log.

## 2026-10-27: One class per track, and the goalkeeper coin flip

The detector's per-frame class flickers within a track. Fix: sum confidence per class over every
processed frame of a track and write the winner to all its rows. That exposed a coin flip for the
visible goalkeeper (about half of his votes were `player`), so a track now becomes `goalkeeper`
if goalkeeper votes reach `class_vote.goalkeeper_min_share` (0.3) of its summed confidence.
Goalkeepers stay out of the team fit and get the team of the nearest player centroid. Provisional:
chosen on one clip with one visible keeper; see docs/TODO.md for using pitch position instead.

## 2026-10-27: Pitch stage thresholds (keypoints, RANSAC, smoothing)

### Question

Do the starting values `kp_conf 0.5`, `reproj_threshold_m 1.0`, `max_reproj_error_m 1.0` and
`smooth_window_sec 0.4` hold up, and how sensitive are the outputs to them?

### Setup

Keypoints of one 30 s clip cached once; each threshold varied alone on the cache; metrics: share
of frames per `h_quality`, inlier reprojection error, players inside the pitch, implausible player
steps (> 12 m/s); plus a simulated 4 s keypoint dropout.

### Decision

Keep the starting values: on this clip no threshold changed the frame coverage, the stricter
RANSAC threshold roughly halved the inlier error but slightly raised implausible steps, and
smoothing reduced them. The dropout run confirmed the reuse logic (2 s reused, then missing).

### Caveats

The fixed camera shows enough landmarks in every frame, so the clip cannot discriminate between
thresholds; re-check on panning tactical footage (docs/TODO.md). Per-clip numbers: private log.

## 2026-10-28: Output rate 12.5 fps for event detection (decision)

Detection and pitch now write records at 12.5 fps instead of 5 fps: events v1 derives passes from
possession changes, and a short pass lasts 0.5-1 s, only 2-5 frames at 5 fps; a 5 fps smoke run
found a possessor in too few frames to detect most passes. Tracking quality is unchanged, the
number of processed frames per clip drops slightly, outputs grow about 2.5x.

## 2026-10-28: Events v1 on the DFL golden set (leave-one-clip-out)

### Question

How well does the rule-based possession model find passes on clips it was not tuned on, and which
airborne-ball check (off / fixed / calibrated) helps?

### Setup

- **Data:** golden set `dfl`: 5 clips x 30 s, 43 golden passes, 4 golden ball losses, tagged by
  hand with the tagger app (docs/GOLDEN.md).
- **Protocol:** grid of 972 threshold combinations (`configs/events.yaml`). For each clip, the
  combination with the best pass F1 on the other four clips is evaluated on the held-out clip;
  counts are summed over the five folds (micro average). Matching tolerance 2.0 s.

### Results (held-out, all five folds together)

| | P | R | F1 | TP / FP / FN |
|---|---|---|---|---|
| passes | 0.721 | 0.721 | 0.721 | 31 / 12 / 12 |

| airborne mode | pass P | pass R | pass F1 | ball observations flagged |
|---|---|---|---|---|
| off | 0.727 | 0.744 | 0.736 | 0.0 % |
| fixed (size ratio > 1.6) | 0.778 | 0.488 | 0.600 | 46.4 % |
| calibrated (ratio / clip baseline > 1.6) | 0.721 | 0.721 | 0.721 | 4.0 % |

Diagnostics over all folds: team mapping correct on every clip; all 43 golden passes have a
detected ball within +-0.5 s; of 12 missed passes, 6 had no player near a known ball, 5 were
predicted 2-5 s off, 1 other; ball loss (anecdotal, 4 golden events): P 0.13, R 0.75, most false
losses are brief opponent touches. The clips with the most fragmented tracking were the weakest
for passes.

### Decision

Keep the tuned thresholds and `airborne_mode: off`: `fixed` discards almost half of the ball
observations and loses a third of the recall; `calibrated` is not better than `off`. Track
stability is the most promising lever for passes; ball loss needs a longer opponent hold.

### Caveats

43 passes from fixed broadcast cameras: wide uncertainty, one pass moves a fold by about 0.1.
Several tuned values sit at the edge of the grid. In-sample numbers (tuned on all five clips)
overstate quality; quote the held-out estimate.

## 2026-10-28: Pass timing and grid edges (last tuning on DFL)

Two checks, same protocol; afterwards tuning on the DFL clips stops.

- **Timing:** on the 31 matched passes our timestamp is early (median -0.27 s): the ball leaves
  the foot after the passer's last possessed frame. Tested `pass_time_fraction` in {0, 0.2, 0.4}.
- **Grid edges:** extended `max_transition_sec` to 5.0, `intercept_min_dist_m` to 12.0, probed
  `min_hold_sec` 0.16.

| variant | combinations | pass P | pass R | pass F1 | TP / FP / FN |
|---|---|---|---|---|---|
| reference (same grid, anchored at tuned values) | 972 | 0.727 | 0.744 | 0.736 | 32 / 12 / 11 |
| + timing | 2,916 | 0.767 | 0.767 | 0.767 | 33 / 10 / 10 |
| + grid edges | 2,304 | **0.889** | 0.744 | **0.810** | 32 / 4 / 11 |
| + timing + grid edges | 6,912 | 0.889 | 0.744 | 0.810 | 32 / 4 / 11 |

Decision: keep `intercept_min_dist_m` 12.0 and `pass_time_fraction` 0 (no held-out gain on top of
the grid edges). Held-out pass F1 0.810 (P 0.889, R 0.744) on the golden set; the gain is in
precision (fewer opponent changes taken for intercepted passes). Both values sit at a grid edge
again and are noted in docs/TODO.md for own footage.

## 2026-10-28: Report agent, prompt versions v1 to v3.1 (local Qwen3-8B)

### Question

How often does the local model deliver a report that passes the validator, and which errors does
each stricter check remove?

### Setup

Ollama 0.40.12, `hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M`, non-thinking mode, temperature 0.7, top_p 0.8,
top_k 20, presence_penalty 1.5, seed 0, context 16,384 tokens; RTX 3070 Ti (8 GB). Three matches
(two 30 s DFL clips with video, one full StatsBomb reference match without video), one run per
match and version, at most 8 steps and one retry after a rejection.

| version | change |
|---|---|
| v1 | numbers in the text must appear in some tool result; clip ids must come from get_clips |
| v2 | the rejection message restates the rules |
| v3 | every supporting number names its source (tool call + field path) and must equal it; comparison direction, English words / metric keys and untranslated team names are checked; tools return Hungarian labels and per-third totals |
| v3.1 | the sample note (duration, event counts, reliability) is written by code, not by the model; a rejected source lists where the cited value actually is |

### Results (all three matches)

| version | accepted | errors found by reading the accepted reports |
|---|---|---|
| v1 | 2 of 3 | wrong comparison direction, English words, metric keys |
| v2 | 2 of 3 | real tool numbers with the wrong meaning (summed zones), English keys, invented links between events |
| v3 | 1 of 3 | swapped counts in the model-written sample note; reliability one level off |
| v3.1 | 2 of 3 | no number, source, comparison or language errors; a title contradicting its claim, a lane statement not supported by the counts |

A retro check of the accepted v1/v2 reports with the v3 rules would have rejected 3 of 4. Rejected
runs use all 8 steps (up to about 54k input tokens, 6.5 minutes locally); accepted runs took 2-6
steps.

### The 67-minute call

One v3 call took 4,032 s. The Windows system log shows the PC sleeping for 65.5 minutes during it
(Kernel-Power 42 / 107); the socket timeout cannot fire while the process is suspended. Ollama's
own timings (logged since v3.1): the model with a 16k context is 82.5 % in GPU memory, stays loaded
between steps, and generation takes 97-99 % of every call (about 20-28 tokens/s). The request
timeout is now 300 s; keep the PC awake during local runs.

### Trade-off and decision

Each check removed a class of errors from accepted reports; since v3.1 no accepted report has a
number with the wrong source, a wrong comparison or English, and the sample facts are exact
because code writes them. The cost is acceptance with an 8B model (1 of 3 at v3, back to 2 of 3
only after code took over the sample note) and long rejected runs. What remains is wording no
number check sees. Prompt iteration stops at v3.1; next are human rubric scores
(docs/REPORT_RUBRIC.md) and a live comparison with Claude Haiku 5.5 (docs/TODO.md).

### Caveats

One run per match and version, one seed: 1 vs 2 of 3 is not a measured difference.
