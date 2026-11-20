"""Build a small variants.jsonl for the Bedrock pilot from the example conversation.

Does not depend on dataset/remix.py (which is not built yet).
Run from the repo root:

    python3 pilots/make_pilot_variants.py

Writes pilots/variants.jsonl with 2 variants of SAMPLE-001:
  EN        — every turn in English (the baseline)
  EN_HI_EN  — rule turns and the last turn English, every other turn Hindi

Usage with the runner:
    python3 -m runner --model llama33_70b --variants pilots/variants.jsonl --out pilots/out
    python3 -m runner --model mock        --variants pilots/variants.jsonl --out pilots/out --limit 2
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "contracts" / "examples" / "conversation.json"
OUTPUT  = Path(__file__).resolve().parent / "variants.jsonl"


def remix_en(conv):
    """Every turn in English."""
    return {
        "conv_id":   conv["id"],
        "condition": "EN",
        "turns":     conv["turns_en"],
    }


def remix_en_hi_en(conv):
    """Rule turns and the last turn in English; every other turn in Hindi.

    Rule turns are turns where a rule is stated (stated_turn across all rules).
    """
    rule_turns = {r["stated_turn"] for r in conv["rules"]}
    last_n     = len(conv["turns_en"])
    turns      = []
    for en, hi in zip(conv["turns_en"], conv["turns_hi"]):
        use_en = (en["n"] in rule_turns) or (en["n"] == last_n)
        turns.append(en if use_en else hi)
    return {
        "conv_id":   conv["id"],
        "condition": "EN_HI_EN",
        "turns":     turns,
    }


def main():
    with open(EXAMPLE, encoding="utf-8") as f:
        conv = json.load(f)

    variants = [remix_en(conv), remix_en_hi_en(conv)]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for v in variants:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    print(f"Wrote {len(variants)} variants to {OUTPUT}")
    for v in variants:
        langs = " ".join(t["lang"] for t in v["turns"])
        print(f"  {v['conv_id']} {v['condition']:12s} turns: {langs}")


if __name__ == "__main__":
    main()
