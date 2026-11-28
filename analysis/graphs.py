"""Figures and LaTeX tables made from the results.

Input:  results.json (written by analysis/metrics.py), and grades.jsonl for the figures that need
        turn-by-turn detail
Output: PNG files and .tex tables in one folder

  python3 -m analysis.graphs --results results/results.json --out figures/
  python3 -m analysis.graphs --results results/results.json --grades runs/all/grades.jsonl --out figures/
  python3 -m analysis.graphs --results results/results.json --grades runs/all/grades.jsonl \
                             --out figures/ --grid S021 qwen3_4b EN_HI_EN

Six figures, each making one point:

  fig1_slip_and_loss        the headline: did the model ever forget a rule (slip), and did a rule end
                            up broken (loss), per model and language type
  fig2_switch_cost          the same numbers as a change from the English baseline, which is the only
                            comparison the study makes. Above zero means worse after a switch
  fig3_rule_grid            one conversation, rule by turn, green for PASS and red for FAIL. Needs grades
  fig4_leakage_vs_drift     every FAIL split by the language the model replied in. A model that answers
                            in Hindi to an English turn is being pulled along by the conversation, which
                            is not the same as forgetting the rule. Needs grades
  fig5_first_lapse_turn     when a rule first breaks, by language type. Needs grades
  fig6_coverage             the share of turns that could not be graded, beside the pass rate. This sits
                            next to every other number on purpose: a pass rate over thin coverage is
                            not a result

Two tables: table1_rates.tex (slip and loss per model and language type) and table2_coverage.tex.

Rule labels (KEPT, RECOVERED, LOST) belong to analysis/metrics.py and are not worked out again here.
Where metrics is not built yet, fig3 draws the grid without the label column and says so.
"""
import argparse
import json
import os
import textwrap
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")                      # no display on Sol or in a notebook job
import matplotlib.pyplot as plt            # noqa: E402
from matplotlib.patches import Patch       # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from contracts.constants import CONDITIONS  # noqa: E402

# One colour per language type, kept the same in every figure. Chosen to stay apart in greyscale
# and for the common forms of colour blindness.
CONDITION_COLOUR = {"EN": "#4C72B0", "EN_HI_EN": "#DD8452", "HI_ALL": "#55A868", "HI_EN_HI": "#C44E52"}
VERDICT_COLOUR = {"PASS": "#55A868", "FAIL": "#C44E52", "UNGRADED": "#B0B0B0", "NOT_APPLICABLE": "#FFFFFF"}
HINDI_REPLY = ("hi", "mixed")


def load_results(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_grades(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _models_and_conditions(results):
    models = sorted(results["models"])
    conditions = [c for c in CONDITIONS if any(c in results["models"][m] for m in models)]
    return models, conditions


def _value(results, model, condition, key):
    return results["models"].get(model, {}).get(condition, {}).get(key)


def _finish(fig, path, note=None, note_y=-0.02):
    """Save the figure. A note goes below everything: short figures need a bigger drop."""
    if note:
        # Wrap to the figure's own width: an unwrapped note runs off the side of a wide figure.
        # About 95 characters per 10 inches keeps it readable at fontsize 7.
        width = max(60, int(fig.get_size_inches()[0] * 9.5))
        note = "\n".join(textwrap.wrap(note, width))
        # bbox_inches="tight" grows the saved image to include text placed outside the axes.
        fig.text(0.01, note_y, note, fontsize=7, color="#555555", va="top")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _bar_positions(n_groups, n_bars, width=0.8):
    step = width / n_bars
    return [[g - width / 2 + step * (b + 0.5) for g in range(n_groups)] for b in range(n_bars)]


def fig_slip_and_loss(results, folder):
    """Slip rate and loss rate, per model and language type."""
    models, conditions = _models_and_conditions(results)
    fig, axes = plt.subplots(1, 2, figsize=(max(8, 2.2 * len(models)), 4.2), sharey=True)
    for ax, key, title in ((axes[0], "slip_rate", "Slip rate — ever forgot a rule"),
                           (axes[1], "loss_rate", "Loss rate — a rule ended up broken")):
        positions = _bar_positions(len(models), len(conditions))
        for b, condition in enumerate(conditions):
            values = [_value(results, m, condition, key) or 0 for m in models]
            ax.bar(positions[b], values, width=0.8 / len(conditions),
                   color=CONDITION_COLOUR.get(condition, "#888888"), label=condition)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("share of conversations")
    axes[1].legend(title="language type", fontsize=8, title_fontsize=8)
    fig.suptitle("Does a rule survive a language switch?", fontsize=13)
    return _finish(fig, os.path.join(folder, "fig1_slip_and_loss.png"),
                   "Conversations labelled UNKNOWN are left out of both rates.")


def fig_switch_cost(results, folder, baseline="EN"):
    """The change from the English baseline. Above zero means the switch cost something."""
    models, conditions = _models_and_conditions(results)
    others = [c for c in conditions if c != baseline]
    if not others:
        return None            # only the baseline was run: there is no switch cost to draw
    fig, ax = plt.subplots(figsize=(max(7, 2.0 * len(models)), 4.2))
    positions = _bar_positions(len(models), len(others))
    for b, condition in enumerate(others):
        deltas = []
        for model in models:
            here, base = _value(results, model, condition, "loss_rate"), _value(results, model, baseline, "loss_rate")
            deltas.append((here - base) if here is not None and base is not None else 0)
        ax.bar(positions[b], deltas, width=0.8 / len(others),
               color=CONDITION_COLOUR.get(condition, "#888888"), label=f"{condition} − {baseline}")
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(f"loss rate minus {baseline}")
    ax.set_title("Switch cost: how much worse than the all-English baseline", fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)
    return _finish(fig, os.path.join(folder, "fig2_switch_cost.png"),
                   "Above zero: the rule was lost more often than in all-English. Each conversation is its own control.")


def fig_rule_grid(grades, folder, conv_id=None, model=None, condition=None, labels=None):
    """One conversation as a grid: a row per rule, a column per turn."""
    rows = [g for g in grades
            if (conv_id is None or g["conv_id"] == conv_id)
            and (model is None or g["model"] == model)
            and (condition is None or g["condition"] == condition)]
    if not rows:
        return None
    conv_id, model, condition = rows[0]["conv_id"], rows[0]["model"], rows[0]["condition"]
    rows = [g for g in rows if g["conv_id"] == conv_id and g["model"] == model and g["condition"] == condition]
    rule_ids = sorted({g["rule_id"] for g in rows})
    turns = sorted({g["turn"] for g in rows})
    verdict = {(g["rule_id"], g["turn"]): g["verdict"] for g in rows}

    fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(turns) + 3), max(2.4, 0.5 * len(rule_ids) + 1.4)))
    for r, rule_id in enumerate(rule_ids):
        for c, turn in enumerate(turns):
            v = verdict.get((rule_id, turn))
            colour = VERDICT_COLOUR.get(v, "#FFFFFF") if v else "#F2F2F2"
            ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=colour, edgecolor="white", linewidth=1.5))
    ax.set_xlim(0, len(turns))
    ax.set_ylim(0, len(rule_ids))
    ax.set_xticks([c + 0.5 for c in range(len(turns))])
    ax.set_xticklabels(turns, fontsize=9)
    ax.set_yticks([r + 0.5 for r in range(len(rule_ids))])
    ax.set_yticklabels([f"{rid}  [{labels[rid]}]" if labels and rid in labels else rid
                        for rid in rule_ids], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("turn")
    ax.set_title(f"{conv_id} · {model} · {condition}", fontsize=11)
    ax.legend(handles=[Patch(facecolor=VERDICT_COLOUR[v], edgecolor="#CCCCCC", label=v.lower().replace("_", " "))
                       for v in ("PASS", "FAIL", "UNGRADED", "NOT_APPLICABLE")],
              fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    note = None if labels else "Rule labels (kept / recovered / lost) appear here once analysis/metrics.py is built."
    return _finish(fig, os.path.join(folder, "fig3_rule_grid.png"), note, note_y=-0.30)


def fig_leakage_vs_drift(grades, folder):
    """Every FAIL split by the language the model replied in."""
    fails = defaultdict(Counter)
    for g in grades:
        if g["verdict"] == "FAIL":
            side = "replied in Hindi or mixed" if g["reply_lang"] in HINDI_REPLY else "replied in English"
            fails[(g["model"], g["condition"])][side] += 1
    if not fails:
        return None
    models = sorted({m for m, _ in fails})
    conditions = [c for c in CONDITIONS if any((m, c) in fails for m in models)]
    fig, ax = plt.subplots(figsize=(max(8, 1.25 * len(models) * max(1, len(conditions)) / 2), 4.6))
    labels, english, hindi = [], [], []
    for model in models:
        for condition in conditions:
            counts = fails.get((model, condition), Counter())
            labels.append(condition)           # the model is named once, under its group
            english.append(counts["replied in English"])
            hindi.append(counts["replied in Hindi or mixed"])
    x = range(len(labels))
    ax.bar(x, english, color="#4C72B0", label="replied in English — the rule itself was broken")
    ax.bar(x, hindi, bottom=english, color="#C44E52",
           label="replied in Hindi or mixed — pulled into the other language")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    # one model name per group of language types, and a divider between groups
    for i, model in enumerate(models):
        middle = i * len(conditions) + (len(conditions) - 1) / 2
        ax.text(middle, -0.17, model, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, fontweight="bold")
        if i:
            ax.axvline(i * len(conditions) - 0.5, color="#CCCCCC", linewidth=0.8, zorder=0)
    ax.set_ylabel("failed rule turns")
    ax.margins(y=0.22)                       # headroom so the legend never sits on a bar
    ax.set_title("Two different failures: forgetting the rule, or drifting into the other language", fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)
    return _finish(fig, os.path.join(folder, "fig4_leakage_vs_drift.png"),
                   note="Across all five models, on the 87 conversations every model completed, "
                        "2,536 of 2,945 failures outside the English baseline (86%) happened while "
                        "the model was replying in Hindi or mixed script. That share splits by "
                        "family: 95-98% for the three Llama models, 63-71% for the two Qwen models, "
                        "which do break rules in English.",
                   note_y=-0.19)


def fig_first_lapse_turn(grades, folder):
    """When a rule first breaks, by language type."""
    first = {}
    for g in grades:
        if g["verdict"] != "FAIL":
            continue
        key = (g["model"], g["condition"], g["conv_id"], g["rule_id"])
        if key not in first or g["turn"] < first[key]:
            first[key] = g["turn"]
    if not first:
        return None
    by_condition = defaultdict(list)
    for (_, condition, _, _), turn in first.items():
        by_condition[condition].append(turn)
    conditions = [c for c in CONDITIONS if c in by_condition]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for condition in conditions:
        turns = sorted(by_condition[condition])
        share = [(i + 1) / len(turns) for i in range(len(turns))]
        ax.step(turns, share, where="post", linewidth=2,
                color=CONDITION_COLOUR.get(condition, "#888888"),
                label=f"{condition} (n={len(turns)})")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))   # turns are whole numbers
    ax.set_xlabel("turn where the rule first broke")
    ax.set_ylabel("share of broken rules, cumulative")
    ax.set_ylim(0, 1.02)
    ax.set_title("How soon a rule breaks", fontsize=12)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)
    return _finish(fig, os.path.join(folder, "fig5_first_lapse_turn.png"),
                   "Only rules that broke at least once are counted.")


def fig_coverage(results, folder):
    """How much could not be graded, beside the pass rate."""
    models, conditions = _models_and_conditions(results)
    fig, axes = plt.subplots(1, 2, figsize=(max(8, 2.2 * len(models)), 4.0))
    for ax, key, title, limit in ((axes[0], "ungraded", "Share of rule turns not graded", None),
                                  (axes[1], "pass_rate", "Pass rate per rule turn", 1)):
        positions = _bar_positions(len(models), len(conditions))
        for b, condition in enumerate(conditions):
            values = []
            for model in models:
                numbers = results["models"].get(model, {}).get(condition, {})
                if key == "ungraded":
                    graded = numbers.get("rule_turns_graded", 0)
                    ungraded = numbers.get("rule_turns_ungraded", 0)
                    total = graded + ungraded
                    values.append(ungraded / total if total else 0)
                else:
                    values.append(numbers.get("pass_rate") or 0)
            ax.bar(positions[b], values, width=0.8 / len(conditions),
                   color=CONDITION_COLOUR.get(condition, "#888888"), label=condition)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
        if limit:
            ax.set_ylim(0, limit)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[1].legend(title="language type", fontsize=8, title_fontsize=8)
    fig.suptitle("Read every other figure next to this one", fontsize=12)
    return _finish(fig, os.path.join(folder, "fig6_coverage.png"),
                   "A pass rate measured over few graded turns is not a result. UNGRADED is never a fail.")


def _tex_escape(text):
    return str(text).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def table_rates(results, folder):
    """Slip and loss per model and language type, as a LaTeX table."""
    models, conditions = _models_and_conditions(results)
    lines = [r"\begin{tabular}{ll" + "r" * 5 + "}", r"\toprule",
             r"Model & Language type & Conversations & Clean & Lost & Slip rate & Loss rate \\", r"\midrule"]
    for model in models:
        for condition in conditions:
            n = results["models"].get(model, {}).get(condition, {})
            if not n:
                continue
            lines.append(
                f"{_tex_escape(model)} & {_tex_escape(condition)} & {n.get('conversations', 0)} & "
                f"{n.get('clean_conversations', 0)} & {n.get('lost_conversations', 0)} & "
                f"{n.get('slip_rate', 0):.3f} & {n.get('loss_rate', 0):.3f} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    path = os.path.join(folder, "table1_rates.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def table_coverage(results, folder):
    """Graded and ungraded turns per model and language type."""
    models, conditions = _models_and_conditions(results)
    lines = [r"\begin{tabular}{ll" + "r" * 4 + "}", r"\toprule",
             r"Model & Language type & Turns graded & Turns ungraded & Ungraded share & Pass rate \\", r"\midrule"]
    for model in models:
        for condition in conditions:
            n = results["models"].get(model, {}).get(condition, {})
            if not n:
                continue
            graded, ungraded = n.get("rule_turns_graded", 0), n.get("rule_turns_ungraded", 0)
            total = graded + ungraded
            share = ungraded / total if total else 0
            lines.append(f"{_tex_escape(model)} & {_tex_escape(condition)} & {graded} & {ungraded} & "
                         f"{share:.3f} & {n.get('pass_rate') or 0:.3f} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    path = os.path.join(folder, "table2_coverage.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def rule_labels(grades, conv_id, model, condition):
    """Ask analysis/metrics.py for the rule labels. Returns {} until that module is built."""
    try:
        from analysis.metrics import label_rule
    except ImportError:
        return {}
    rows = defaultdict(list)
    for g in grades:
        if g["conv_id"] == conv_id and g["model"] == model and g["condition"] == condition:
            rows[g["rule_id"]].append(g)
    labels = {}
    for rule_id, group in rows.items():
        try:
            label, _, _ = label_rule(sorted(group, key=lambda g: g["turn"]))
        except (NotImplementedError, Exception):   # noqa: B014 - metrics is someone else's module
            return {}
        labels[rule_id] = label
    return labels


def make_figures(results_path, output_folder, grades_path=None, grid=None):
    """Write every figure and table into the output folder. Returns the paths written."""
    os.makedirs(output_folder, exist_ok=True)
    results = load_results(results_path)
    written = [fig_slip_and_loss(results, output_folder),
               fig_switch_cost(results, output_folder),
               fig_coverage(results, output_folder),
               table_rates(results, output_folder),
               table_coverage(results, output_folder)]
    if grades_path:
        grades = load_grades(grades_path)
        conv_id, model, condition = (grid or (None, None, None))
        labels = rule_labels(grades, conv_id, model, condition) if conv_id else {}
        written += [fig_rule_grid(grades, output_folder, conv_id, model, condition, labels),
                    fig_leakage_vs_drift(grades, output_folder),
                    fig_first_lapse_turn(grades, output_folder)]
    return [p for p in written if p]


def main():
    parser = argparse.ArgumentParser(prog="analysis.graphs")
    parser.add_argument("--results", required=True, help="results.json from analysis/metrics.py")
    parser.add_argument("--out", required=True, help="folder for the figures and tables")
    parser.add_argument("--grades", help="grades.jsonl, for the grid, leakage and first-lapse figures")
    parser.add_argument("--grid", nargs=3, metavar=("CONV_ID", "MODEL", "CONDITION"),
                        help="which conversation the rule grid shows")
    args = parser.parse_args()
    paths = make_figures(args.results, args.out, args.grades, tuple(args.grid) if args.grid else None)
    for path in paths:
        print(path)
    if not args.grades:
        print("(pass --grades to add the rule grid, the leakage split and the first-lapse figure)")


if __name__ == "__main__":
    main()
