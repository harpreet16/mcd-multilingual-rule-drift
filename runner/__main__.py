"""Command line for the runner.

  python3 -m runner --model llama31_8b --variants variants.jsonl --out runs/llama31_8b
  python3 -m runner --model mock --variants variants.jsonl --out runs/mock --limit 3
  python3 -m runner --model llama31_8b --variants variants.jsonl --out runs/llama31_8b --status
"""
import argparse
import os
import sys

from contracts.io import load_variants
from models.settings import MODELS, SETTINGS
from runner.mock import make_mock_chat
from runner.run import run_all, status


def settings_for(key):
    """What is saved with every response. If any of it changes, finished runs are redone."""
    entry = MODELS[key]
    saved = {"backend": entry["backend"], "model_id": entry["model_id"], **SETTINGS}
    for name in ("context_window", "system_message"):
        if name in entry:
            saved[name] = entry[name]
    return saved


def make_chat_for(key, base_url):
    entry = MODELS[key]
    if entry["backend"] == "sol":
        from models.sol.client import make_chat
        return make_chat(base_url, entry["model_id"], SETTINGS["max_tokens"], SETTINGS["max_tokens_cap"],
                         SETTINGS["temperature"], entry.get("extra_body"))
    from models.bedrock.client import make_chat
    region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
    return make_chat(entry["model_id"], region, SETTINGS["max_tokens"], SETTINGS["max_tokens_cap"],
                     SETTINGS["temperature"], entry.get("system_message"))


def main():
    parser = argparse.ArgumentParser(prog="runner")
    parser.add_argument("--model", required=True, help="a key from models/settings.py, or mock")
    parser.add_argument("--variants", required=True)
    parser.add_argument("--out", required=True, help="folder for responses.jsonl and partial.json")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="the vLLM server (Sol models)")
    parser.add_argument("--workers", type=int, help="conversations at once. Default comes from models/settings.py")
    parser.add_argument("--limit", type=int, help="only the first N variants, for a quick check")
    parser.add_argument("--status", action="store_true", help="show progress and exit")
    args = parser.parse_args()

    variants = load_variants(args.variants)
    if args.limit:
        variants = variants[:args.limit]

    if args.model == "mock":
        settings = {"backend": "mock", "model_id": "mock", "temperature": 0, "max_tokens": 4096}
        workers, chat = args.workers or 2, make_mock_chat()
    else:
        settings = settings_for(args.model)
        workers = args.workers or MODELS[args.model]["workers"]
        chat = None if args.status else make_chat_for(args.model, args.base_url)

    if args.status:
        print(status(variants, args.model, settings, args.out))
        return
    summary = run_all(variants, chat, args.model, settings, args.out, workers)
    print(summary)
    if summary["failed"]:
        print("Some conversations failed. Run the same command again: finished turns are kept.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
