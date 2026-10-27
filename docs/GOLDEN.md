# Golden sets: format and tagging guidelines

Golden sets are hand-tagged match events used to measure the events stage. They are tagged
with the tagger app (`make tag`, Hungarian UI) and evaluated with `make eval`.

**Golden sets of footage that must not be published** (the DFL development clips) live in
`evals/private/` with their corrections and results; `configs/evals.private.yaml` points the
tools there (see CLAUDE.md, publication rule).

**Only the tagger app writes to `evals/golden/`.** Do not edit these files by hand or from
scripts; evaluation code only reads them.

## Layout

```
evals/golden/<set_id>.json     one golden set per source (committed only with the footage
                               owner's consent to publish)
data/raw/<set_id>/<clip_id>.mp4  the clips of that set (never committed)
evals/results/<UTC timestamp>.json  one file per evaluation run
data/processed/<clip_id>/events.json  predictions of the events stage for one clip
```

A golden set groups the short clips of one source, so results can be reported per source:
`dfl` holds the five 30 s DFL sample clips from roboflow/sports; an own recording becomes its own
set later. The clip id is the video file name without extension and must match the match id the
pipeline uses for that clip.

## File format

```json
{
  "set_id": "dfl",
  "source": "roboflow/sports examples/soccer/setup.sh (DFL Bundesliga Data Shootout)",
  "description": "",
  "clips": [
    {
      "clip_id": "clip_a",
      "video": "data/raw/dfl/clip_a.mp4",
      "fps": 25.0,
      "duration_sec": 30.0,
      "kits": {"home": "fehér", "away": "piros"},
      "complete": true,
      "tagged_by": "Oli",
      "updated_at": "2026-10-27T18:00:00Z",
      "events": [
        {"id": "clip_a-1a2b3c4d", "t_sec": 3.12, "type": "pass", "team": "home",
         "x_m": null, "y_m": null, "confidence": null}
      ]
    }
  ]
}
```

- `t_sec` is relative to the start of the clip, rounded to 0.01 s.
- `type` is one of `pass`, `ball_loss`, `shot`, `set_piece`; `team` is `home` or `away`.
- `kits` describes which kit is home and which is away for this clip. The pipeline's team labels
  are arbitrary clusters, so the events stage must map them to home/away; evaluation is strict
  and never swaps teams to improve a score.
- `x_m` / `y_m` are optional (the tagger leaves them empty); `confidence` is for predictions only.
- **Only clips with `complete: true` are evaluated.** Mark a clip complete only after every event
  in it has been tagged; a half-tagged clip would count real events as false positives.
- Event ids are unique within a set; clip ids are unique within a set.

Predictions use the same `Event` model (`src/match_tagger/events/schema.py`), wrapped in
`{"clip_id": ..., "events": [...]}`.

## Scope of the current golden sets

The `dfl` set (5 clips, 48 events: 43 pass, 4 ball_loss, 1 shot, 0 set_piece) is only large
enough to measure passes. Events v1 therefore detects **pass** (primary metric) and
**ball_loss** (reported as anecdotal). **Shot and set piece are out of scope** until an own,
longer recording is tagged; their rules and metrics will be added then.

## Evaluation

- A prediction matches a golden event of the **same clip, type and team** if their times differ by
  at most `tolerance_sec` (2.0 s, `configs/evals.yaml`).
- Matching is one-to-one and greedy by time difference: the closest pair is matched first; every
  event is used at most once, so duplicate predictions produce one true positive and false
  positives for the rest.
- Precision, recall and F1 are reported per event type and overall (micro-averaged: counts summed
  over types), **per golden set**. Precision is undefined ("-") without predictions, recall without
  golden events; F1 is 0 when golden events exist and none was found.
- A clip without an `events.json` counts as "no predictions" and is listed in the report.

```sh
make eval                 # all golden sets
make eval SET=dfl         # one golden set
make eval-compare A=evals/results/<a>.json B=evals/results/<b>.json   # B - A per metric
```

## Tagging guidelines

Every event is a single instant. Tag the moment described below, not the start of the move.

### Pass (Passz)

- **What counts:** any intentional ball played by a player to a teammate in open play, with foot,
  head or chest: short and long passes, crosses, through balls, back passes to the goalkeeper, a
  goalkeeper's throw or kick to a teammate.
- **What does not count:** shots, clearances without a target, deflections, a dribble touch, the
  restart kick of a set piece (tag `set_piece` instead).
- **Successful or not:** both count as a pass. If an opponent intercepts it, also tag a
  `ball_loss` for the passing team.
- **Moment:** when the ball leaves the passer.
- **Team:** the passer's team.

### Ball loss (Labdavesztés)

- **What counts:** a team loses possession in open play: an intercepted or misplaced pass, a lost
  duel or dribble, a tackle, a bad touch that the opponent collects, a ball played out of play
  that gives the opponent a restart.
- **When it starts:** the moment the opponent gains control (their first controlled touch) or the
  moment the ball crosses the line. Not the moment of the mistake.
- **What does not count:** a shot that ends the attack (tag `shot` only); a short contested ball
  that the same team wins back immediately without the opponent ever controlling it; a foul
  against the team in possession (they keep the ball via a free kick).
- **Team:** the team that lost the ball.

### Shot (Lövés)

- **What counts:** any attempt to score with foot or head, on or off target, including blocked
  shots and a direct free kick or penalty.
- **Moment:** the contact with the ball.
- **Team:** the shooter's team.

### Set piece (Pontrúgás)

- **What counts:** the restart of play after a stoppage: kick-off, free kick, corner, throw-in,
  goal kick, penalty.
- **Moment:** when the ball is played (the restart kick or throw), not when the referee whistles.
- **Team:** the team taking it.
- **Combinations:** tag the restart as `set_piece` and not also as `pass`. A direct free kick or
  penalty at goal gets both `set_piece` and `shot` at the same moment.

### Edge cases

- If the moment is not visible (camera elsewhere, replay, graphics), do not tag the event.
- If you are unsure about the type, choose the closest definition above and keep it consistent;
  note recurring doubts here so the guidelines can be extended.
- Tag the whole clip before ticking "Klip kész". Events that start before the clip or end after
  it are tagged only if their moment lies inside the clip.

## Tagger app

`make tag` opens the app. Choose or create a golden set (its clips must be in
`data/raw/<set_id>/`), pick a clip, enter the kit descriptions, then play the video and press:

| key | action |
|---|---|
| `1` / `2` / `3` / `4` | Passz / Labdavesztés / Lövés / Pontrúgás at the current video time |
| `H` / `V` | Hazai / Vendég team for the next events |
| `Z` | undo the last event added in this session |
| `Space` | play / pause |
| `←` / `→` | back / forward 0.2 s |
| `,` / `.` | back / forward one frame |
| `[` / `]` | slower / faster playback |

Click into the tagger area first so the keys reach it. The event list below the player shows
time, type and team; click a row to jump to it, or delete a single event with "Törlés". Every
change is saved immediately.

## From review to golden set

The review app (`make review`) never writes to `evals/golden/`. Every correction of predicted
events is appended to `evals/corrections/<match>.jsonl` (one JSON line: `seq`, `timestamp`,
`action` = accept / delete / change_type / change_team / add / undo, `event_id`, `before`,
`after`, `target_seq` for undo). The merged result can be exported as
`evals/candidates/<match>.json`, a golden clip with `complete: false` and no confidences. To
promote it, review it, set the kits and `complete: true`, and add it to a golden set with the
tagger (or paste it into the set file, then open the set once in the tagger to validate it).
A reviewed candidate starts from the model's predictions, so check it as carefully as a fresh
tagging session: events the model never predicted only appear if you added them.
