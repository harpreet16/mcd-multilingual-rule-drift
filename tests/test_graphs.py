"""Tests for analysis/graphs.py. They render into a temporary folder; no display is needed."""
import json
import os
import tempfile
import unittest

from analysis import graphs
from contracts.check import check_results

MODELS = ["qwen3_4b", "llama32_3b"]
CONDITIONS = ["EN", "EN_HI_EN", "HI_ALL", "HI_EN_HI"]


def make_results():
    out = {"models": {}}
    for i, model in enumerate(MODELS):
        out["models"][model] = {}
        for j, condition in enumerate(CONDITIONS):
            out["models"][model][condition] = {
                "conversations": 100, "clean_conversations": 60 - j * 5,
                "recovered_conversations": 10 + j, "lost_conversations": 25 + j * 4,
                "unknown_conversations": 5 - j if j < 5 else 0,
                "slip_rate": 0.30 + 0.05 * j + 0.02 * i, "loss_rate": 0.20 + 0.04 * j,
                "rules_kept": 120, "rules_recovered": 20, "rules_lost": 30,
                "rule_turns_graded": 900 + j, "rule_turns_ungraded": 50 + j,
                "pass_rate": 0.9 - 0.05 * j,
            }
    return out


def make_grades(with_failures=True):
    rows = []
    for model in MODELS:
        for condition in CONDITIONS:
            for conv in range(1, 4):
                for rule_id in ("R1", "R2"):
                    for turn in (2, 4, 6):
                        failed = with_failures and turn >= 4 and rule_id == "R2"
                        rows.append({
                            "conv_id": f"S{conv:03d}", "condition": condition, "model": model,
                            "rule_id": rule_id, "turn": turn,
                            "verdict": "FAIL" if failed else "PASS",
                            "method": "judge",
                            "reply_lang": "hi" if (failed and condition != "EN") else "en",
                            "reason": "", "static_result": "", "judge_result": "NO" if failed else "YES",
                            "judge_model": "fake", "rubric_version": "v1", "escalated": False,
                        })
    return rows


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.results_path = os.path.join(self.folder, "results.json")
        with open(self.results_path, "w", encoding="utf-8") as f:
            json.dump(make_results(), f)
        self.grades_path = os.path.join(self.folder, "grades.jsonl")
        with open(self.grades_path, "w", encoding="utf-8") as f:
            for row in make_grades():
                f.write(json.dumps(row) + "\n")

    def test_the_test_results_match_the_contract(self):
        self.assertEqual(check_results(make_results()), [])

    def test_results_only_makes_the_figures_that_need_no_grades(self):
        out = os.path.join(self.folder, "a")
        written = graphs.make_figures(self.results_path, out)
        names = sorted(os.path.basename(p) for p in written)
        self.assertEqual(names, ["fig1_slip_and_loss.png", "fig2_switch_cost.png", "fig6_coverage.png",
                                 "table1_rates.tex", "table2_coverage.tex"])
        for path in written:
            self.assertGreater(os.path.getsize(path), 0)

    def test_with_grades_every_figure_is_written(self):
        out = os.path.join(self.folder, "b")
        written = graphs.make_figures(self.results_path, out, self.grades_path,
                                      grid=("S001", "qwen3_4b", "EN_HI_EN"))
        names = sorted(os.path.basename(p) for p in written)
        self.assertEqual(names, ["fig1_slip_and_loss.png", "fig2_switch_cost.png", "fig3_rule_grid.png",
                                 "fig4_leakage_vs_drift.png", "fig5_first_lapse_turn.png",
                                 "fig6_coverage.png", "table1_rates.tex", "table2_coverage.tex"])

    def test_a_grid_for_a_conversation_with_no_grades_is_skipped(self):
        out = os.path.join(self.folder, "c")
        os.makedirs(out, exist_ok=True)
        self.assertIsNone(graphs.fig_rule_grid(make_grades(), out, "S999", "qwen3_4b", "EN"))

    def test_no_failures_means_no_leakage_or_lapse_figure(self):
        out = os.path.join(self.folder, "d")
        os.makedirs(out, exist_ok=True)
        clean = make_grades(with_failures=False)
        self.assertIsNone(graphs.fig_leakage_vs_drift(clean, out))
        self.assertIsNone(graphs.fig_first_lapse_turn(clean, out))

    def test_one_model_and_one_condition_still_renders(self):
        small = {"models": {"qwen3_4b": {"EN": make_results()["models"]["qwen3_4b"]["EN"]}}}
        path = os.path.join(self.folder, "small.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(small, f)
        out = os.path.join(self.folder, "e")
        written = [os.path.basename(p) for p in graphs.make_figures(path, out)]
        self.assertIn("fig1_slip_and_loss.png", written)
        # with only the baseline there is nothing to compare it against
        self.assertNotIn("fig2_switch_cost.png", written)


class TableTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()

    def test_rates_table_has_a_row_per_model_and_condition(self):
        path = graphs.table_rates(make_results(), self.folder)
        with open(path, encoding="utf-8") as f:
            body = f.read()
        for model in MODELS:
            for condition in CONDITIONS:
                self.assertIn(condition.replace("_", r"\_"), body)   # LaTeX escapes underscores
            self.assertIn(model.replace("_", r"\_"), body)
        self.assertIn(r"\bottomrule", body)
        self.assertNotIn(r"\midrule" + "\n" + r"\end{tabular}", body)

    def test_underscores_are_escaped_for_latex(self):
        path = graphs.table_coverage(make_results(), self.folder)
        with open(path, encoding="utf-8") as f:
            body = f.read()
        self.assertIn(r"qwen3\_4b", body)
        self.assertIn(r"EN\_HI\_EN", body)

    def test_coverage_table_reports_the_ungraded_share(self):
        path = graphs.table_coverage(make_results(), self.folder)
        with open(path, encoding="utf-8") as f:
            body = f.read()
        self.assertIn("Ungraded share", body)


class MetricsHandoffTests(unittest.TestCase):
    def test_labels_are_empty_while_metrics_is_not_built(self):
        # graphs does not work out rule labels itself: that belongs to analysis/metrics.py.
        # Until it is built the grid is drawn without them rather than guessing.
        self.assertEqual(graphs.rule_labels(make_grades(), "S001", "qwen3_4b", "EN"), {})


if __name__ == "__main__":
    unittest.main()
