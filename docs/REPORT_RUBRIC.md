# Report rubric

Human scores for generated coaching reports. The validator already guarantees that every number
comes from the tools, that supporting numbers match their source field, that comparisons point
the right way, that the text is in Hungarian and that clips exist. This rubric covers what code
cannot judge.

## How to score

- Score each **accepted** report once, as a whole (not per finding), after reading it and
  watching at least the clips of one finding.
- Three criteria, 1-5 each. Use the anchors; 2 and 4 lie between them.
- Add a short comment when a score is 3 or lower: what would have to change.
- Score against what the data allows: a 30 s clip cannot support a pattern, and a report that
  says so is not penalised for being modest.

```sh
make score-reports                          # list accepted reports not scored yet
make score-reports ARGS="add <run_id>"      # asks for the three scores and a comment
make score-reports ARGS="summary"           # acceptance and mean scores per backend / prompt
```

Scores go to `evals/results/report_scores.jsonl` (one line per report; scorer defaults to "Oli";
a new score of the same report by the same scorer needs `--replace` and supersedes the old one).

## Criteria

### 1. Usefulness for a coach (Hasznosság edzőnek)

Would a coach change something in the next training week because of this report?

| score | anchor |
|---|---|
| 1 | Nothing to act on: restates counts, findings are generic or contradict the clips. |
| 3 | One finding points at a real situation in the clips, but the training focus is vague ("javítsák a passzokat"). |
| 5 | Every finding names a concrete situation (where, which team, what happened) that the clips show, and a training focus a coach could plan tomorrow. |

### 2. Clarity (Érthetőség)

Can a coach read it in two minutes and repeat the main points?

| score | anchor |
|---|---|
| 1 | Hard to follow: long or broken sentences, numbers without context, findings overlap. |
| 3 | Understandable, but one finding needs re-reading, or the sample note is unclear about what the data allows. |
| 5 | Short, one idea per finding, numbers explained in words, the sample note says plainly how far the findings can be trusted. |

### 3. Correct football terms (Szakszerű futballkifejezések)

Does it use the vocabulary of a Hungarian coaching staff correctly?

| score | anchor |
|---|---|
| 1 | Wrong or invented terms, literal translations, terms used for the wrong thing (e.g. "lane" vs "third"). |
| 3 | Mostly correct, a few clumsy or non-standard phrases. |
| 5 | Natural coaching Hungarian: passz, labdavesztés, letámadás, harmadok and sávok used precisely. |
