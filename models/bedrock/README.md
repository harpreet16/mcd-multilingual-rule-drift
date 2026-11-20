# Bedrock backend

Runs Llama 3.3-70B and Qwen3-32B via the Amazon Bedrock Converse API.  
The LLM judge (Claude Haiku 3.5, owned by Shashwat) reuses the same client.

---

## Prerequisites

### 1. Install boto3

```bash
pip install boto3
```

The rest of the project uses only the standard library. `boto3` is only needed for Bedrock models.

### 2. Create AWS credentials

1. Log in to [AWS Console](https://console.aws.amazon.com)
2. Go to **IAM → Users → your user → Security credentials**
3. Click **Create access key** → choose "Other" → Create
4. Copy the **Access Key ID** and **Secret Access Key**

### 3. Enable model access in Bedrock

1. In AWS Console, search for **Amazon Bedrock**
2. Go to **Model access** (left sidebar)
3. Enable these models:

| Model | Used for |
|---|---|
| Llama 3.3 70B Instruct (Meta) | Inference |
| Qwen3 32B (Alibaba) | Inference |
| Claude 3.5 Haiku (Anthropic) | LLM judge (Shashwat's code) |

> Model access can take a few minutes to activate. Check the status before running.

### 4. Set environment variables

Copy `.env.example` to `.env` at the repo root and fill in your credentials:

```bash
cp .env.example .env
# edit .env with your AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
```

Load them before running:

```bash
export $(grep -v '^#' .env | xargs)
```

Or add them to your shell profile / Sol environment.

---

## Verifying the setup

A quick connectivity check before a full run:

```python
from models.bedrock.client import make_chat
chat = make_chat("us.meta.llama3-3-70b-instruct-v1:0")
text, cut_off = chat([{"role": "user", "content": "Reply with one word: Ready"}])
print(text, cut_off)
```

If you see `Ready False` the client is working.

For the judge model:

```python
chat = make_chat("us.anthropic.claude-haiku-4-5-20251001-v1:0")
text, cut_off = chat([{"role": "user", "content": "Reply with one word: Ready"}])
print(text, cut_off)
```

---

## Running a model

Use the runner from the repo root. The runner picks up `AWS_DEFAULT_REGION` from the environment automatically:

```bash
# dry run with the mock model — no credentials needed
python3 -m runner --model mock --variants pilots/variants.jsonl --out pilots/out --limit 2

# real run with Llama 3.3-70B on Bedrock
python3 -m runner --model llama33_70b --variants pilots/variants.jsonl --out pilots/out

# check progress on a stopped run
python3 -m runner --model llama33_70b --variants pilots/variants.jsonl --out pilots/out --status

# continue a stopped run (just run the same command again — finished conversations are skipped)
python3 -m runner --model llama33_70b --variants pilots/variants.jsonl --out pilots/out
```

---

## Model IDs

Model IDs and settings live in `models/settings.py` — change them there, not here.

| Key in settings.py | Bedrock model ID |
|---|---|
| `llama33_70b` | `us.meta.llama3-3-70b-instruct-v1:0` |
| `qwen3_32b` | `qwen.qwen3-32b-v1:0` |

The judge model ID is pinned in Shashwat's `grader/judge.py`.

---

## Cost estimate

Every turn sends the full conversation history again (Bedrock is stateless).  
Rough estimate for 100 conversations × 4 conditions × 10 turns × 5 models on Bedrock (2 models):

- Llama 3.3-70B: ~$0.00072/1K input + $0.00099/1K output tokens
- Qwen3-32B: ~$0.15/1M input + $0.62/1M output tokens (check Bedrock pricing page)

Run the pilot first to get real numbers before committing to 2,000 runs.

---

## Qwen3 thinking mode

Qwen3-32B generates `<think>...</think>` reasoning blocks by default. The system message
`"You are a helpful assistant. /no_think"` switches this off. It is set in `models/settings.py`
and passed automatically by the runner. The client also strips any thinking blocks that slip
through and emits a warning so you can catch problems early.
