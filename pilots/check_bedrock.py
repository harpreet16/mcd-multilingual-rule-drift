"""Quick connectivity check for the Bedrock backend.

Run this before the pilot to confirm credentials, region and model access are all working.

    python3 pilots/check_bedrock.py

Checks:
  1. boto3 is installed
  2. AWS credentials are set in the environment
  3. Llama 3.3-70B responds (inference model — Harsh runs this)
  4. Claude Haiku 3.5 responds (judge model — Shashwat runs this)
  5. Qwen3-32B responds and thinking is off (inference model)
"""
import os
import sys

ROOT = __file__.rsplit("/pilots/", 1)[0]
sys.path.insert(0, ROOT)


def check(label, fn):
    try:
        result = fn()
        print(f"  ✓  {label}: {result}")
        return True
    except Exception as exc:
        print(f"  ✗  {label}: {exc}")
        return False


def main():
    print("Bedrock connectivity check")
    print()

    # 1. boto3
    ok = check("boto3 installed", lambda: __import__("boto3").__version__)
    if not ok:
        print("\nFix: pip install boto3")
        sys.exit(1)

    # 2. credentials
    key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
    if not key or not secret:
        print("  ✗  AWS credentials: AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY not set")
        print("\nFix: copy .env.example to .env, fill in values, then: export $(grep -v '^#' .env | xargs)")
        sys.exit(1)
    print(f"  ✓  AWS credentials: key={key[:8]}{'*' * (len(key)-8)}  region={region}")

    # 3-5. model calls
    from models.bedrock.client import make_chat
    from models.settings import JUDGE_MODEL_ID, MODELS

    prompt = [{"role": "user", "content": "Reply with one word: Ready"}]
    all_ok = True

    all_ok &= check(
        "Llama 3.3-70B (inference)",
        lambda: make_chat(MODELS["llama33_70b"]["model_id"])(prompt)[0].strip()[:40],
    )
    all_ok &= check(
        f"Claude Haiku 3.5 (judge, {JUDGE_MODEL_ID})",
        lambda: make_chat(JUDGE_MODEL_ID)(prompt)[0].strip()[:40],
    )
    all_ok &= check(
        "Qwen3-32B with /no_think (inference)",
        lambda: (
            lambda text, _: (
                "FAIL — thinking block present" if "<think>" in text else text.strip()[:40]
            )
        )(*make_chat(MODELS["qwen3_32b"]["model_id"],
                     system_message=MODELS["qwen3_32b"]["system_message"])(prompt)),
    )

    print()
    if all_ok:
        print("All checks passed. Ready to run the pilot.")
        print("  python3 -m runner --model llama33_70b --variants pilots/variants.jsonl --out pilots/out")
    else:
        print("Some checks failed. Fix the issues above before running the pilot.")
        sys.exit(1)


if __name__ == "__main__":
    main()
