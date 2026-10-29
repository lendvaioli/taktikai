---
# DRAFT (phase 7b): not published yet. Fill in licence and consent before uploading.
license: other           # TODO: agree with the club (e.g. cc-by-nc-4.0); see "Licence" below
language:
  - hu
pretty_name: TaktikAI golden match events
task_categories:
  - video-classification
tags:
  - football
  - sports-analytics
  - event-detection
---

# TaktikAI golden match events (draft)

Hand-tagged match events (passes, ball losses, shots, set pieces) for evaluating automatic event
detection from football video, as used by [TaktikAI](https://github.com/lendvaioli/taktikai).

**Status: draft for phase 7b.** It will contain annotations of footage from a club that has given
written consent to publish. The development clips used so far (DFL broadcast samples) are not
part of this dataset and will not be published.

## Contents

One JSON file per golden set, following `golden_set.schema.json` (exported from the pydantic
model `GoldenSet` in `src/match_tagger/evals/golden.py`):

- a golden set: `set_id`, `source`, `description`, `clips`;
- a clip: `clip_id`, `video` (path in the project, the video itself is not included), `fps`,
  `duration_sec`, `kits` (which kit is home and away), `complete`, `tagged_by`, `updated_at`,
  `events`;
- an event: `id`, `t_sec` (seconds from the clip start), `type` (`pass`, `ball_loss`, `shot`,
  `set_piece`), `team` (`home` / `away`), optional pitch position and outcome fields.

Only clips marked `complete: true` (every event tagged) are used for evaluation.

## Collection method

Events are tagged by a video analyst with the TaktikAI tagger app (keyboard shortcuts at the
moment of the event, playback controls, undo), following the written guidelines in
`docs/GOLDEN.md` (what counts as a pass, a ball loss, a shot and a set piece, and which instant
to tag). Predicted events corrected in the review app can be exported as golden candidates, which
are checked like a fresh tagging session before they are added.

## Consent and personal data

- Footage: published only with the club's written consent (TODO: date, scope, who signed).
- The annotations contain no names: teams are `home` / `away`, events have no player identity.
  Players can still be identified from the video, which is why the consent covers the footage.
- TODO before publishing: whether the consent also covers the video, or only the annotations.

## Licence

TODO: to be agreed with the club; the code that produces and evaluates the data is
AGPL-3.0-or-later.

## Limitations

Single annotator, no inter-annotator agreement measured yet; timestamps are at video-frame
precision but the tagged instant is a judgement (e.g. when the ball leaves the passer's foot).
