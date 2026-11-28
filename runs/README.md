# runs

One folder per model: the replies the runner collected, and what produced them.

| Folder | Model | Backend | Ran by | Conversations | jobs.jsonl |
|---|---|---|---|---|---|
| `qwen3_4b/` | Qwen3-4B-Instruct-2507 | Sol | Harpreet | 400 | yes |
| `llama31_8b/` | Llama-3.1-8B-Instruct | Sol | Harpreet | 400 | yes |
| `llama32_3b/` | Llama-3.2-3B-Instruct | Sol | Harpreet | 400 | yes |
| `llama33_70b/` | Llama-3.3-70B-Instruct | Bedrock | Harsh | 400 | **no** |
| `qwen3_32b/` | Qwen3-32B | Bedrock | Brijesh | **378 of 400** | no (but see below) |
| `qwen3_4b_colab_mine/` | Qwen3-4B-Instruct-2507 | Colab A100 | Brijesh | 181 (partial) | yes |

Every folder holds `responses.jsonl` (one finished conversation per line) and `partial.json`
(`{}` when the run finished). The grader reads `responses.jsonl`:

```
python3 -m grader --responses runs/<model>/responses.jsonl --out runs/<model>/grades.jsonl
```

## Checks run on these files

All four completed runs: 400 conversations, 100 complete sets of four language types, no empty
replies, no `<think>` blocks, and settings matching `models/settings.py` exactly. Cut-off replies:
135 (qwen3_4b), 33 (llama31_8b), 33 (llama32_3b), 30 (llama33_70b) — those turns are graded
UNGRADED, never counted as failures.

## qwen3_32b: 378 of 400, and why

The run stopped with `status: needs_attention`, `reason: Remaining variants exceeded the model
context limit`. All 22 failures are the same Bedrock error: *this model's maximum context length is
32768 tokens*. No throttling, no empty replies, no other error type.

**The 22 are not spread evenly.** By language type: 10 `EN_HI_EN`, 10 `HI_ALL`, 2 `HI_EN_HI`,
**0 `EN`**. They come from 13 conversations, and those 13 are the long ones: a mean of 12.2 turns
against 8.3 for the other 87, and 2.4x the user text in `HI_ALL` (2,988 characters against 1,233).
Devanagari costs more tokens per character, so the Hindi-bearing types hit 32K first. The missing
data is missing *because* of the condition under test, so it cannot be treated as missing at random.

**87 conversations have all four language types.** That is 3,404 of the 3,768 rule-turns in the
design (90%). Use those 87 for every paired test; the other 13 are reported as a limitation, not
patched. `analysis/` takes the complete-sets-only path for this model.

Why not use the partial replies in `partial.json`? They hold 221 answered turns from the 22 failed
variants, but only 97 rule-turns (2.6% of the design), and splicing them in would grade `EN` over 15
turns and `HI_ALL` over 11 *within the same conversation* — which breaks the paired comparison the
study rests on. Not worth 2.6%.

Four of the 22 are a client bug rather than a hard wall: `S069/HI_ALL`, `S070/EN_HI_EN`,
`S070/HI_EN_HI` and `S072/EN_HI_EN` had already returned a truncated reply at `max_tokens` 4096, and
failed only when the retry doubled the request to 8192 without checking the remaining context. The
other 18 report at least 28,673 input tokens against a 4,096-token output request, so the history
alone is near the wall. Lowering `max_tokens` would change `models/settings.py` and make the run
non-comparable with the other four, so the 22 cannot be rescued without re-running all 400.

## The provenance gap

`jobs.jsonl` records the dataset SHA-256, the git commit, the GPU and the settings. It is written by
`models/sol/preflight.py`, which the Sol job runs — so the three Sol runs have it and all three
record `variants_sha256 4eb96b1d406d...`, the same dataset.

**The Bedrock runs have no `jobs.jsonl`**, because the Bedrock path calls the runner directly and the
preflight includes a GPU check it cannot pass. For `llama33_70b` the next best evidence is that its
400 (conversation, language type) pairs match `dataset/data/variants.jsonl` exactly — good evidence
it ran on our dataset, but not proof of the turn texts.

`qwen3_32b` closes that gap from the other side: its `job_status.json` records
`variants_sha256 4eb96b1d406d...`, the same dataset hash the three Sol runs record. So four of the
five models are tied to the dataset by hash; only `llama33_70b` rests on the weaker pair-match
argument.

## qwen3_4b_colab_mine

The same model and settings as `qwen3_4b`, run on a Colab A100 rather than Sol (vLLM 0.34.0 against
0.33.0, 40 GB card against 80 GB). Superseded by Harpreet's complete run, kept because comparing the
two is a free reproducibility check: at temperature 0 the replies should be near identical.

**It is partial: 181 responses over 48 of the 100 conversations, and it has no grades.** It is not a
sixth model. See `qwen3_4b_colab_mine/DO_NOT_ANALYZE.md`. Analysis code should name the five model
folders explicitly rather than globbing `runs/*/`, which would now sweep this one in.

## What is committed here, as of 2026-11-27

Everything a run leaves behind, not just the replies and the grades. The team agreed to keep the
whole record in the repo so the analysis can be redone without re-running a model or re-paying the
judge. Only lock files are left out: they hold a dead process id and nothing else.

| file | what it is | safe to analyse? |
|------|------------|------------------|
| `<model>/responses.jsonl` | the model's replies | **yes** |
| `<model>/grades.jsonl` | the verdicts the analysis reads | **yes** |
| `<model>/job_status.json`, `jobs.jsonl` | provenance: dataset hash, settings, GPU | yes, as evidence |
| `<model>/partial.json` | turns from conversations that never finished | no, see the qwen3_32b section |
| `<model>/grades_before_static_fix.jsonl` | grades from **before** commit `6d1033d` | **no, superseded** |
| `<model>/logs/`, `run.log`, `grading.log` | what the run printed: throttles, retries, errors | yes, as evidence |
| `judge_cache.jsonl` | the judge's answers, keyed by SHA-256 of the ask | not data, but keep it |
| `qwen3_4b_colab_mine/` | a partial 48-conversation rerun | **no, see above** |

Two of these need care.

**`grades_before_static_fix.jsonl`** is the output of the grader as it stood before a passing
`not_contains` check was allowed to settle a single-requirement rule. On `qwen3_4b` the fix changed
10 verdicts out of 3,768. Small, but these files are the *wrong* numbers sitting next to the right
ones, under a name a careless glob will match. They are kept only so the fix can be audited.

**`judge_cache.jsonl`** is worth the space: it holds all 4,146 judge answers across the five
models, so re-running the grader costs nothing and returns exactly the same verdicts. Deleting it
does not lose data, it just means paying Bedrock again. It is append-only, so expect it to show up
in diffs after any grading run.

An earlier version of this file claimed 3,333 answers and that re-grading was free for every model.
That was wrong for `qwen3_32b`, and the reason is worth recording.

## The qwen3_32b rubric_version correction

`qwen3_32b/grades.jsonl` originally recorded a blank `rubric_version` on all 3,609 rows, where the
other four models record `v1`. Two separate mistakes, one harmless and one not:

**Harmless: the rubric was never different.** `rubric_version` is a label passed on the command
line; it is not part of the judge's prompt. `build_prompt(rule, conversation_text, turns)` does not
take it, and the rubric itself is `HOUSE_RULES`, a constant in `grader/judge.py`. That file was
byte-identical between the commit that graded the four models and the one that graded qwen3_32b, and
has not changed since. The judge saw the same instructions for all five models.

**Not harmless: the run wrote its cache to the wrong place.** `CACHE_PATH` defaulted to
`judge_cache.jsonl` relative to the working directory, and `.env` does not set `MCD_JUDGE_CACHE`, so
the qwen3_32b grading wrote 813 answers to a second cache at the repository root that was never
committed. Four models had 99.8-100% of their judge calls in the committed cache; qwen3_32b had
0.2%. The default is now `runs/judge_cache.jsonl` so this cannot happen again.

**The fix, and why it is not a hand-edit.** The 813 answers were re-keyed from `rubric_version=""`
to `"v1"` and merged into the committed cache. The cache key is a lookup address — a SHA-256 over
the judge model, rubric version, rule, transcript and turns — so re-keying changes where an answer
is filed, never what it says. The grader was then re-run with `--rubric-version v1`, which made
**zero new judge calls** because every ask hit the cache. So `grades.jsonl` records `v1` because the
grader wrote it, not because anyone edited the file.

The re-run is its own check: comparing the 3,609 rows before and after, the only field that changed
was `rubric_version` (3,235 rows), and **no verdict changed**. The 374 rows still blank are the ones
the judge was never asked about, which is also true of `judge_model` on the same 374 rows and
matches the other four models.
