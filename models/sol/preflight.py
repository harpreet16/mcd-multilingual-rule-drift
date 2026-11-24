"""Checks before a Sol job starts the GPU, and a record of what the job ran.

  python3 -m models.sol.preflight --model llama31_8b --variants dataset/data/variants.jsonl --out /scratch/$USER/mcd_runs/llama31_8b

What it does:
  1. checks the model is a Sol model in models/settings.py, and that the GPU Slurm gave can run it:
     bfloat16 needs compute capability 8.0 or newer (A100, H100; not V100), and the weights plus the history of
     long conversations need at least MIN_GPU_MEMORY_MB (an A30 with 24 GB cannot hold a 65,536 token window)
  2. checks the variants file against the contract: every variant valid, no (conv_id, condition) twice,
     and every turn the same text as dataset/data/conversations.jsonl (so the remixer output is what runs)
  3. adds one line to <out>/jobs.jsonl: job id, node, GPU, git commit, the variants file and its sha256,
     the vLLM version and the model settings. One line per job, so a resubmitted run keeps its history
  4. prints "MODEL_ID CONTEXT_WINDOW DTYPE" as its last line, for submit.sbatch to start vLLM with

Exits with an error, before any GPU time is spent, if anything is wrong.
"""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from collections import Counter
from importlib import metadata

from contracts.check import check_variant
from contracts.io import load_conversations, load_variants
from models.settings import MODELS
from runner.__main__ import settings_for

CONVERSATIONS = "dataset/data/conversations.jsonl"
MIN_GPU_MEMORY_MB = 40000     # an 8B model is about 16 GB in bfloat16; the rest holds 16 conversations' history
MIN_COMPUTE_CAPABILITY = 8.0  # bfloat16


def check_gpu(nvidia_smi_line):
    """Return (gpu name, problems) from one line of
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits"""
    parts = [p.strip() for p in nvidia_smi_line.split(",")]
    if len(parts) < 3:
        return "", ["no GPU visible (nvidia-smi gave nothing). Did the job ask for --gres=gpu:1?"]
    name, memory, capability = parts[0], parts[1], parts[2]
    problems = []
    try:
        if float(capability) < MIN_COMPUTE_CAPABILITY:
            problems.append(f"{name} has compute capability {capability}; bfloat16 needs {MIN_COMPUTE_CAPABILITY} "
                            f"or newer (A100, H100)")
        if float(memory) < MIN_GPU_MEMORY_MB:
            problems.append(f"{name} has {memory} MB; this run needs at least {MIN_GPU_MEMORY_MB} MB (A100, H100)")
    except ValueError:
        problems.append(f"could not read the GPU details: {nvidia_smi_line!r}")
    return f"{name}, {memory} MB, compute {capability}", problems


def check_variants(variants, conversations):
    """Return the problems with a list of variants, in plain words."""
    by_id = {c.id: c for c in conversations}
    problems = []
    for pair, count in Counter((v.conv_id, v.condition) for v in variants).items():
        if count > 1:
            problems.append(f"{pair[0]} {pair[1]} appears {count} times")
    for variant in variants:
        if variant.conv_id not in by_id:
            problems.append(f"{variant.conv_id}: not in {CONVERSATIONS}")
        problems += check_variant(variant, by_id.get(variant.conv_id))
    return problems


def _run(command):
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _version(package):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return ""


def main():
    parser = argparse.ArgumentParser(prog="models.sol.preflight")
    parser.add_argument("--model", required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    entry = MODELS.get(args.model)
    if entry is None or entry["backend"] != "sol":
        sol = [key for key, m in MODELS.items() if m["backend"] == "sol"]
        sys.exit(f"{args.model} is not a Sol model. Sol models: {', '.join(sol)}")
    gpu, gpu_problems = check_gpu((_run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                                         "--format=csv,noheader,nounits"]).splitlines() or [""])[0])
    if gpu_problems:
        print("this GPU cannot run the job:", *gpu_problems, "Submit again to get another GPU.",
              sep="\n  ", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.variants):
        sys.exit(f"variants file not found: {args.variants} (from {os.getcwd()})")

    variants = load_variants(args.variants)
    problems = check_variants(variants, load_conversations(CONVERSATIONS))
    if not variants:
        problems.append("the variants file is empty")
    if problems:
        print(f"{len(problems)} problem(s) in {args.variants}:", *problems[:20], sep="\n  ", file=sys.stderr)
        sys.exit(1)

    with open(args.variants, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    by_condition = Counter(v.condition for v in variants)
    record = {
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "node": socket.gethostname(),
        "gpu": gpu,
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_uncommitted_changes": bool(_run(["git", "status", "--porcelain", "--untracked-files=no"])),
        "variants_file": args.variants,
        "variants_sha256": sha256,
        "variants": len(variants),
        "variants_by_condition": dict(sorted(by_condition.items())),
        "turns": sum(len(v.turns) for v in variants),
        "model": args.model,
        "settings": settings_for(args.model),
        "vllm_version": _version("vllm"),
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "jobs.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"gpu ok: {gpu}")
    print(f"variants ok: {len(variants)} ({', '.join(f'{k} {n}' for k, n in sorted(by_condition.items()))}), "
          f"{record['turns']} turns, sha256 {sha256[:12]}")
    print(f"git commit {record['git_commit'][:10]}"
          + ("  WARNING: uncommitted changes, results may not match the commit" if record["git_uncommitted_changes"] else ""))
    print(entry["model_id"], entry["context_window"], entry["dtype"])


if __name__ == "__main__":
    main()
