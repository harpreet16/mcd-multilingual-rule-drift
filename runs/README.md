# runs

One folder per model: the replies the runner collected, and what produced them.

| Folder | Model | Backend | Ran by | Conversations | jobs.jsonl |
|---|---|---|---|---|---|
| `qwen3_4b/` | Qwen3-4B-Instruct-2507 | Sol | Harpreet | 400 | yes |
| `llama31_8b/` | Llama-3.1-8B-Instruct | Sol | Harpreet | 400 | yes |
| `llama32_3b/` | Llama-3.2-3B-Instruct | Sol | Harpreet | 400 | yes |
| `llama33_70b/` | Llama-3.3-70B-Instruct | Bedrock | Harsh | 400 | **no** |
| `qwen3_32b/` | Qwen3-32B | Bedrock | — | not yet run | — |
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

## The provenance gap

`jobs.jsonl` records the dataset SHA-256, the git commit, the GPU and the settings. It is written by
`models/sol/preflight.py`, which the Sol job runs — so the three Sol runs have it and all three
record `variants_sha256 4eb96b1d406d...`, the same dataset.

**The Bedrock runs have no such record**, because the Bedrock path calls the runner directly and the
preflight includes a GPU check it cannot pass. For `llama33_70b` the next best evidence is that its
400 (conversation, language type) pairs match `dataset/data/variants.jsonl` exactly — good evidence
it ran on our dataset, but not proof of the turn texts.

Worth fixing before the remaining Bedrock run: a provenance step for the Bedrock path, so all five
models can be shown to have graded the same data.

## qwen3_4b_colab_mine

The same model and settings as `qwen3_4b`, run on a Colab A100 rather than Sol (vLLM 0.34.0 against
0.33.0, 40 GB card against 80 GB). Superseded by Harpreet's complete run, kept because comparing the
two is a free reproducibility check: at temperature 0 the replies should be near identical.
