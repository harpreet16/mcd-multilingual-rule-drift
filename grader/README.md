# grader

Turns a model's replies into graded rows: `responses.jsonl` -> `grades.jsonl`.

Owner: Brijesh. `judge.py` was Shashwat's in the work division; it is written here so grading is not
blocked, against the interface the contracts already fixed. If he writes his own, it drops in and
every test in `tests/test_judge.py` still applies.

```
python3 -m grader --responses runs/qwen3_4b/responses.jsonl --out runs/qwen3_4b/grades.jsonl
python3 -m grader --responses runs/mock/responses.jsonl --out runs/mock/grades.jsonl --fake-judge
```

Grading is resumable: a conversation already in the output file is skipped, so a stopped run
continues with the same command. `--redo` starts the file again. `--fake-judge` answers YES to
everything and calls nothing; it is for trying the pipeline, and the summary says the verdicts are
not results.

| File | What it does |
|---|---|
| `hybrid.py` | The dispatch: decides, for each rule at each turn, whether code or the judge gives the verdict, and builds the `Grade` rows |
| `static_checks.py` | `run_static_check(rule, reply, reply_lang) -> "PASS" or "FAIL"` for the check types in the dataset |
| `reply_language.py` | `detect_reply_language(reply) -> "en", "hi", "mixed" or "unknown"` |
| `fake_judge.py` | A stand-in judge for tests and dry runs |
| `judge.py` | The LLM judge: Claude Haiku on Bedrock, with caching and retries |
| `__main__.py` | The command line above |

## How a verdict is decided

The full table is at the top of `hybrid.py`. In short:

1. No reply, an empty reply or a cut-off reply is `UNGRADED`.
2. An `english_only` rule on a Hindi or mixed reply is `NOT_APPLICABLE`.
3. A static check that passes is final. A static check that fails is final **only** when the reply
   is in English.
4. A static failure on a Hindi or mixed reply goes to the judge, and the judge's answer is final.
   A correct Hindi reply can fail an English word check: in the earlier audit, 16 of 18 static
   disagreements on Hindi replies were false fails.
5. `both` rules get the static check first and the judge after; the judge decides.
6. `judge` rules go straight to the judge. All asks for one response go in a single call.

## Errors never count as fails

This is the rule the earlier pipeline broke: it stored 38 judge API errors as `pass=False`, which
reads an outage as drift. Here a judge `ERROR`, a missing or repeated answer, a judge that raises,
and a broken check setting all come back as `UNGRADED`, which the analysis never counts as a fail.
`tests/test_hybrid.py` has a test for each of those paths.

The one thing that is **not** swallowed is a judge server that cannot be reached at all. That raises
`ServerUnreachable` and stops the run, so nobody ends up with a file of silently ungraded turns.

## Two decisions worth knowing

**Forbidden words match on word boundaries.** `not_contains "heal"` does not fail on "healthy",
because IFEval's own checker — where these rules come from — matches `\bheal\b`. Rules whose judge
question says "in any form" are `both` rules, so the judge covers the wider reading.

**Romanised Hindi counts as mixed, not English.** A reply written in Hindi using Latin letters is a
language-leakage case, not the model forgetting the rule, and the analysis splits those two apart.

## Known limits

- A long English reply quoting one Hindi phrase is called "mixed", which sends a static failure to
  the judge. That costs a judge call and never costs a false fail.
- The four `starts_with: "\""` rules mean "the whole reply is wrapped in quotes", which a check on
  the start alone cannot confirm. They are all `both` rules, so the judge confirms the ending. If
  the team wants it checked by code, the contract needs a `wrapped_in` check type.
- Hindi written in Latin letters with none of the function words in `reply_language.py` reads as
  English.

## The judge

Claude Haiku on Bedrock, through Harsh's client. It is shown the rule, the yes/no question and the
conversation so far — and never the model name or the language type, so it cannot favour one arm of
the study.

All the turns asked for one rule go in a single call. On the real dataset that turns 376 asks into
112 calls, about 3.4 turns per call, which matters when judging two models is roughly 7,000 asks.
The cost is that a verdict on turn 4 is given while turns 5 onwards are visible. Pass
`asks_per_call=1` for one call per turn with the transcript cut at that turn, and compare the two on
a sample — PLAN.md asks for that comparison before the full run.

Answers are cached in `judge_cache.jsonl`, keyed on the judge model, the rubric version, the rule's
question and text, the exact transcript and the turns asked. A stopped job does not pay twice.
Set `MCD_JUDGE_CACHE` to put the file somewhere else. The default is `runs/judge_cache.jsonl`,
which is the committed cache; it is relative to the working directory, so run the grader from the
repository root or set the variable.

An answer that cannot be read is re-asked twice and then becomes `ERROR`, never `NO`.

Tests: `python3 -m unittest tests.test_static_checks tests.test_hybrid tests.test_judge`
