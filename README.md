# MCD: Multilingual Rule Drift

Do language models keep following a user's rules after a conversation switches language? This repository measures that for English and Hindi.

## The question

A user gives a model a rule early in a conversation, for example "start every answer with 'Let's talk about that.'". The conversation goes on, and part of it switches to Hindi and back. Does the model still follow the rule?

## Design

- **100 conversations.** Each has rules stated early, in the middle and late, with unrelated side-questions mixed in
- **4 language types per conversation:** `EN`, `EN_HI_EN`, `HI_ALL`, `HI_EN_HI`
- **5 models:** Qwen3-4B, Llama 3.2-3B and Llama 3.1-8B (served on Sol with vLLM), Qwen3-32B and Llama 3.3-70B (Amazon Bedrock)
- **Live conversations.** Only the user's turns are stored. The model's replies are made one turn at a time, and each reply is added to the history before the next turn is sent
- **Per-turn grading.** Every reply after a rule was stated is graded against that rule. Rules with a clear answer are checked by code. Rules that need judgement are checked by an LLM judge (Claude Haiku 3.5 on Bedrock). A code fail on a Hindi reply goes to the judge, since a correct Hindi reply can fail a word check
- **Labels instead of pass or fail.** Each rule is Kept, Recovered or Lost. Each conversation is Clean, Recovered or Lost
- **Two headline numbers:** `slip_rate` (did the model ever forget a rule?) and `loss_rate` (did a rule end up broken?). Each conversation is compared with itself across language types, with an exact McNemar test per model

`PLAN.md` explains the design and the measures in full.

## Pipeline

```
dataset -> runner -> grader -> analysis -> dashboard
              |
           models/  (sol, bedrock)
```

Each stage reads and writes JSON lines files in formats fixed in `contracts/`:

`conversations.jsonl` -> `variants.jsonl` -> `responses.jsonl` -> `grades.jsonl` -> `results.json`

## Repository layout

| Folder | What it holds |
|---|---|
| `contracts/` | The file formats as small dataclasses, one example of each, and a checker |
| `dataset/` | `compile.py` merges and checks the conversations. `remix.py` builds the four language types |
| `runner/` | Plays a conversation against a model turn by turn, with resume and a fake model for tests |
| `models/` | `settings.py` (the five models), `sol/` (vLLM client and Sol job), `bedrock/` (Bedrock client) |
| `grader/` | `static_checks.py`, `hybrid.py` (decides code check or judge), `judge.py` |
| `analysis/` | `metrics.py` (labels, rates, tests), `graphs.py` (figures and tables) |
| `dashboard/` | A small web view of the results and transcripts |
| `tests/` | Unit tests. They use a fake model, so no GPU or network is needed |

## Requirements

Python 3.10 or newer. The runner, the contracts and the tests use only the standard library.

| Part | Package |
|---|---|
| Bedrock client and judge | `boto3` |
| Metrics | `numpy`, `scipy` |
| Graphs | `matplotlib` |
| Sol jobs | `vllm`, in an environment called `mcd-vllm` |

## Usage

Run the tests:

```
python3 -m unittest discover -s tests -t .
```

Run a model on a file of variants (`mock` is a fake model, so it needs no GPU):

```
python3 -m runner --model mock --variants variants.jsonl --out runs/mock --limit 3
python3 -m runner --model llama31_8b --variants variants.jsonl --out runs/llama31_8b
python3 -m runner --model llama31_8b --variants variants.jsonl --out runs/llama31_8b --status
python3 -m runner --model llama31_8b --variants variants.jsonl --out runs/llama31_8b --conditions EN EN_HI_EN --conv-ids S01 S02
```

`--conditions` and `--conv-ids` run only the language types and conversations named. `--limit N` keeps the first N of what is left.

A stopped run continues where it stopped: run the same command again. Finished conversations are skipped. If a setting in `models/settings.py` changes, everything is asked again, including conversations that stopped half way.

On Sol, set your Hugging Face token in your terminal and submit the job for a model from `models/settings.py`:

```
export HF_TOKEN=...
sbatch models/sol/submit.sbatch llama31_8b
```

Set the account line at the top of `models/sol/submit.sbatch` to your own Sol account.

## Limits

The study does not measure reply quality or factual accuracy, and it does not check the Hindi translations unless a native speaker reviewed them. The full list is at the end of `PLAN.md`.
