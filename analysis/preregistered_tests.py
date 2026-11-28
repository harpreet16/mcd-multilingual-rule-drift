"""The comparison PLAN.md step 4 asks for, computed straight from grades.jsonl.

This exists because the headline numbers must be reproducible before analysis/metrics.py lands.
It implements PLAN.md's labelling independently, so it is also an oracle: when metrics.py is
built, its label_rule and its rates must reproduce what this prints. If the two disagree, one of
them is wrong, and that is worth knowing before either goes in the report.

PLAN.md, to the letter:
  rule labels   KEPT (no fail, >=80% of its turns graded), RECOVERED (a fail, last graded turn
                passes), LOST (last graded turn fails), UNKNOWN (no fail, under 80% graded)
  conv labels   LOST if any rule lost, else RECOVERED if any recovered, else UNKNOWN if any
                unknown, else CLEAN
  rates         slip = (RECOVERED+LOST)/(CLEAN+RECOVERED+LOST), loss = LOST/(same),
                over conversations that are not UNKNOWN
  test          each conversation against itself, one model at a time, exact McNemar, Holm over
                the 5 models -- step 4 says "a correction for 5 models (Holm)", so a family is
                5 tests, not 15

Two counting policies are reported, because a NOT_APPLICABLE from an english_only rule on a Hindi
reply is missing data, not a proven failure. See runs/README.md.
  strict      NA never counts as a fail. PLAN.md as written
  inclusive   language-scope NA counts as the rule broken; the judge's own NA never does

Usage:  python3 -m analysis.preregistered_tests [--json runs/preregistered_results.json]
"""
import argparse
import collections
import json
import os

MODELS = ("qwen3_4b", "qwen3_32b", "llama31_8b", "llama32_3b", "llama33_70b")
HINDI = ("EN_HI_EN", "HI_ALL", "HI_EN_HI")

# The 13 conversations Qwen3-32B could not finish inside its context window. Every model is cut to
# the same 87 so the panels are like-for-like. runs/README.md explains why, and shows that dropping
# them from the four complete models flips nothing.
DROPPED = ("S002", "S003", "S005", "S006", "S010", "S046", "S049",
           "S069", "S070", "S071", "S072", "S077", "S079")


def is_language_na(grade):
    """True for a NOT_APPLICABLE that hybrid.py produced because an english_only rule met a Hindi
    reply, as opposed to one the judge gave because the rule did not apply to the turn.

    They are separable by field with no text matching: the language kind never went to the judge.
    """
    return (grade["verdict"] == "NOT_APPLICABLE"
            and grade.get("judge_result") != "NOT_APPLICABLE")


def label_rule(grades, policy="strict"):
    """The label for one rule in one response. grades: its Grade rows, any order."""
    rows = sorted(grades, key=lambda g: g["turn"])
    verdicts = ["FAIL" if (policy == "inclusive" and is_language_na(g)) else g["verdict"]
                for g in rows]
    graded = [v for v in verdicts if v in ("PASS", "FAIL")]
    if not graded:
        return "UNKNOWN"
    if "FAIL" in graded:
        return "LOST" if graded[-1] == "FAIL" else "RECOVERED"
    return "KEPT" if len(graded) / len(verdicts) >= 0.8 else "UNKNOWN"


def label_conversations(grades, policy="strict"):
    """{(model, conv_id, condition): label} for every response in the grades."""
    by_rule = collections.defaultdict(list)
    for g in grades:
        by_rule[(g["model"], g["conv_id"], g["condition"], g["rule_id"])].append(g)
    rules_of = collections.defaultdict(list)
    for key, rows in by_rule.items():
        rules_of[key[:3]].append(label_rule(rows, policy))
    out = {}
    for key, labels in rules_of.items():
        if "LOST" in labels:
            out[key] = "LOST"
        elif "RECOVERED" in labels:
            out[key] = "RECOVERED"
        elif "UNKNOWN" in labels:
            out[key] = "UNKNOWN"
        else:
            out[key] = "CLEAN"
    return out


def rate(conv_labels, model, condition, which):
    """slip or loss over the conversations that are not UNKNOWN. Returns (rate, n)."""
    labels = [v for (m, _, c), v in conv_labels.items()
              if m == model and c == condition and v != "UNKNOWN"]
    if not labels:
        return None, 0
    hit = sum(1 for v in labels
              if (v in ("RECOVERED", "LOST") if which == "slip" else v == "LOST"))
    return hit / len(labels), len(labels)


def discordant(conv_labels, model, condition, which):
    """(en_only, hindi_only): the two discordant counts of the paired table."""
    en_only = hindi_only = 0
    for conv_id in {k[1] for k in conv_labels if k[0] == model}:
        en = conv_labels.get((model, conv_id, "EN"))
        hi = conv_labels.get((model, conv_id, condition))
        if en is None or hi is None or en == "UNKNOWN" or hi == "UNKNOWN":
            continue
        pick = (lambda v: v in ("RECOVERED", "LOST")) if which == "slip" else (lambda v: v == "LOST")
        if pick(hi) and not pick(en):
            hindi_only += 1
        elif pick(en) and not pick(hi):
            en_only += 1
    return en_only, hindi_only


def exact_mcnemar(en_only, hindi_only):
    """Two-sided exact McNemar: a binomial sign test on the discordant pairs."""
    from scipy.stats import binomtest
    n = en_only + hindi_only
    return binomtest(hindi_only, n, 0.5).pvalue if n else 1.0


def holm(pvalues):
    """Holm step-down, returning adjusted p-values in the original order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def load_grades(runs_dir="runs", models=MODELS, dropped=DROPPED):
    dropped = set(dropped)
    out = []
    for model in models:
        path = os.path.join(runs_dir, model, "grades.jsonl")
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                grade = json.loads(line)
                if grade["conv_id"] not in dropped:
                    out.append(grade)
    return out


def compute(grades):
    """Every number this module reports, as a plain dict."""
    results = {"models": list(MODELS), "conditions": ["EN", *HINDI],
               "conversations_per_cell": None, "policies": {}}
    for policy in ("strict", "inclusive"):
        conv = label_conversations(grades, policy)
        block = {"conversation_labels": dict(collections.Counter(conv.values())),
                 "rates": {}, "per_model_tests": {}, "sign_test": {}, "pooled": {}}
        for which in ("slip", "loss"):
            block["rates"][which] = {
                m: {c: rate(conv, m, c, which)[0] for c in ("EN", *HINDI)} for m in MODELS}
            # per model, Holm over the 5 models within each condition
            block["per_model_tests"][which] = {}
            for condition in HINDI:
                tables = [discordant(conv, m, condition, which) for m in MODELS]
                raw = [exact_mcnemar(*t) for t in tables]
                adj = holm(raw)
                block["per_model_tests"][which][condition] = {
                    m: {"en_only": t[0], "hindi_only": t[1], "p": p, "holm": a}
                    for m, t, p, a in zip(MODELS, tables, raw, adj)}
            # direction across all model x condition cells
            up = down = 0
            for m in MODELS:
                base = rate(conv, m, "EN", which)[0]
                for condition in HINDI:
                    here = rate(conv, m, condition, which)[0]
                    if here is None or base is None:
                        continue
                    up += here > base
                    down += here < base
            block["sign_test"][which] = {
                "hindi_worse": up, "of": up + down, "p": exact_mcnemar(down, up)}
            # pooled across models, per condition
            block["pooled"][which] = {}
            for condition in HINDI:
                en_only = hindi_only = 0
                for m in MODELS:
                    a, b = discordant(conv, m, condition, which)
                    en_only += a
                    hindi_only += b
                block["pooled"][which][condition] = {
                    "en_only": en_only, "hindi_only": hindi_only,
                    "p": exact_mcnemar(en_only, hindi_only)}
        results["policies"][policy] = block
    convs = {k[1] for k in label_conversations(grades, "strict")}
    results["conversations_per_cell"] = len(convs)
    return results


def report(results):
    for policy, block in results["policies"].items():
        print("=" * 76)
        print(f"POLICY: {policy}     conversations per cell: {results['conversations_per_cell']}")
        print("=" * 76)
        print("conversation labels:", block["conversation_labels"], "\n")
        for which in ("slip", "loss"):
            print(f"--- {which}_rate ---")
            print("%-13s %8s %9s %8s %9s" % ("model", "EN", *HINDI))
            for m in results["models"]:
                row = block["rates"][which][m]
                print("%-13s %7.1f%% %8.1f%% %7.1f%% %8.1f%%"
                      % (m, *[100 * row[c] for c in results["conditions"]]))
            sign = block["sign_test"][which]
            print(f"\n  direction: Hindi worse than EN in {sign['hindi_worse']} of {sign['of']} "
                  f"model x condition cells, sign test p = {sign['p']:.3g}")
            print("  pooled over the 5 models (exact McNemar on the summed discordant pairs):")
            for condition, cell in block["pooled"][which].items():
                print("    EN vs %-9s EN-only=%3d  Hindi-only=%3d   p = %.3g"
                      % (condition, cell["en_only"], cell["hindi_only"], cell["p"]))
            print("  per model, Holm over the 5 models:")
            for condition, cells in block["per_model_tests"][which].items():
                survived = [m for m, v in cells.items() if v["holm"] < 0.05]
                print("    EN vs %-9s survives Holm: %s"
                      % (condition, ", ".join(survived) if survived else "none"))
            print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--json", help="also write the numbers here")
    args = parser.parse_args()
    results = compute(load_grades(args.runs))
    report(results)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, sort_keys=True)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
