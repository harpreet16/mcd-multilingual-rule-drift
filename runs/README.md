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
