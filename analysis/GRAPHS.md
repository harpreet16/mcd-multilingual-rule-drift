# graphs

Figures and LaTeX tables for the report. Owner: Brijesh. `metrics.py` in this folder is Shashwat's.

```
python3 -m analysis.graphs --results results/results.json --out figures/
python3 -m analysis.graphs --results results/results.json --grades runs/all/grades.jsonl \
                           --out figures/ --grid S021 qwen3_4b EN_HI_EN
```

`--results` alone makes the three figures and both tables that only need `results.json`. Add
`--grades` for the three that need turn-by-turn detail.

| Figure | The one point it makes |
|---|---|
| `fig1_slip_and_loss` | The headline: how often a rule was ever forgotten, and how often it ended up broken, per model and language type |
| `fig2_switch_cost` | The same numbers as a change from the all-English baseline. Above zero means the switch cost something |
| `fig3_rule_grid` | One conversation, rule by turn, green for pass and red for fail. This is where a reader sees a rule break and come back |
| `fig4_leakage_vs_drift` | Every failure split by the language the model replied in — forgetting the rule is not the same as being pulled into the other language |
| `fig5_first_lapse_turn` | How soon a rule breaks, by language type |
| `fig6_coverage` | The share of turns that could not be graded, beside the pass rate |

`table1_rates.tex` and `table2_coverage.tex` are `tabular` bodies, ready for `\input{}`.

## Why coverage has a figure of its own

A pass rate measured over thin coverage is not a result. `UNGRADED` is never counted as a fail, so a
flaky judge quietly shrinks the denominator instead of showing up as drift. Read every other figure
next to `fig6`, and keep `table2_coverage.tex` in the report.

## What this module does not do

Rule labels (`KEPT`, `RECOVERED`, `LOST`) belong to `analysis/metrics.py` and are not worked out
here, so there is one definition of them in the repo. `fig3` asks `metrics.label_rule` for them; until
that module is built, the grid is drawn without the label column and says so on the figure.

Colours are fixed per language type in `CONDITION_COLOUR` and stay the same in every figure. They
hold apart in greyscale and for the common forms of colour blindness.

Tests: `python3 -m unittest tests.test_graphs` (renders into a temporary folder, no display needed)
