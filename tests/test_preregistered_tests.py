"""Tests for analysis/preregistered_tests.py: the labelling PLAN.md specifies, and the corrections."""
import unittest

from analysis.preregistered_tests import (discordant, holm, is_language_na, label_conversations,
                                          label_rule)


def grade(turn, verdict, judge_result=None, **rest):
    g = {"turn": turn, "verdict": verdict, "model": "m", "conv_id": "S001",
         "condition": "EN", "rule_id": "R1"}
    if judge_result is not None:
        g["judge_result"] = judge_result
    g.update(rest)
    return g


class TestLanguageNA(unittest.TestCase):
    def test_hybrid_skip_is_the_language_kind(self):
        self.assertTrue(is_language_na(grade(1, "NOT_APPLICABLE")))

    def test_the_judges_own_na_is_not(self):
        self.assertFalse(is_language_na(grade(1, "NOT_APPLICABLE", judge_result="NOT_APPLICABLE")))

    def test_a_pass_is_never_na(self):
        self.assertFalse(is_language_na(grade(1, "PASS", judge_result="YES")))

    def test_a_judge_yes_on_an_na_row_is_the_language_kind(self):
        # the field that matters is judge_result, not whether a judge was involved at all
        self.assertFalse(is_language_na(grade(1, "FAIL", judge_result="NO")))


class TestLabelRule(unittest.TestCase):
    def test_all_pass_and_full_coverage_is_kept(self):
        self.assertEqual(label_rule([grade(1, "PASS"), grade(2, "PASS")]), "KEPT")

    def test_a_fail_then_a_pass_is_recovered(self):
        self.assertEqual(label_rule([grade(1, "FAIL"), grade(2, "PASS")]), "RECOVERED")

    def test_ending_on_a_fail_is_lost(self):
        self.assertEqual(label_rule([grade(1, "PASS"), grade(2, "FAIL")]), "LOST")

    def test_the_last_GRADED_turn_decides_not_the_last_turn(self):
        rows = [grade(1, "FAIL"), grade(2, "PASS"), grade(3, "UNGRADED")]
        self.assertEqual(label_rule(rows), "RECOVERED")

    def test_order_does_not_matter(self):
        rows = [grade(3, "FAIL"), grade(1, "PASS"), grade(2, "PASS")]
        self.assertEqual(label_rule(rows), "LOST")

    def test_thin_coverage_without_a_fail_is_unknown(self):
        rows = [grade(1, "PASS")] + [grade(n, "UNGRADED") for n in range(2, 6)]
        self.assertEqual(label_rule(rows), "UNKNOWN")

    def test_exactly_eighty_percent_coverage_is_kept(self):
        rows = [grade(n, "PASS") for n in range(1, 5)] + [grade(5, "UNGRADED")]
        self.assertEqual(label_rule(rows), "KEPT")

    def test_a_fail_outranks_thin_coverage(self):
        # PLAN.md gates only KEPT on coverage; a fail is a fail however little was graded
        rows = [grade(1, "FAIL")] + [grade(n, "UNGRADED") for n in range(2, 9)]
        self.assertEqual(label_rule(rows), "LOST")

    def test_nothing_graded_is_unknown(self):
        self.assertEqual(label_rule([grade(1, "UNGRADED"), grade(2, "NOT_APPLICABLE")]), "UNKNOWN")

    def test_inclusive_turns_language_na_into_a_fail(self):
        # four graded of five is 80%, so strict reaches KEPT; inclusive makes turn 5 a FAIL
        rows = [grade(n, "PASS") for n in range(1, 5)] + [grade(5, "NOT_APPLICABLE")]
        self.assertEqual(label_rule(rows, "strict"), "KEPT")
        self.assertEqual(label_rule(rows, "inclusive"), "LOST")

    def test_inclusive_leaves_the_judges_na_alone(self):
        rows = ([grade(n, "PASS") for n in range(1, 5)]
                + [grade(5, "NOT_APPLICABLE", judge_result="NOT_APPLICABLE")])
        self.assertEqual(label_rule(rows, "strict"), "KEPT")
        self.assertEqual(label_rule(rows, "inclusive"), "KEPT")

    def test_language_na_below_the_coverage_bar_is_unknown_under_strict(self):
        # the NA is not graded, so coverage is 50% and KEPT is out of reach
        rows = [grade(1, "PASS"), grade(2, "NOT_APPLICABLE")]
        self.assertEqual(label_rule(rows, "strict"), "UNKNOWN")
        self.assertEqual(label_rule(rows, "inclusive"), "LOST")


class TestLabelConversations(unittest.TestCase):
    def rows(self, by_rule):
        out = []
        for rule_id, verdicts in by_rule.items():
            for n, v in enumerate(verdicts, start=1):
                out.append(grade(n, v, rule_id=rule_id))
        return out

    def test_any_lost_rule_loses_the_conversation(self):
        labels = label_conversations(self.rows({"R1": ["PASS", "PASS"], "R2": ["PASS", "FAIL"]}))
        self.assertEqual(labels[("m", "S001", "EN")], "LOST")

    def test_recovered_beats_unknown(self):
        labels = label_conversations(self.rows({"R1": ["FAIL", "PASS"], "R2": ["UNGRADED"]}))
        self.assertEqual(labels[("m", "S001", "EN")], "RECOVERED")

    def test_all_kept_is_clean(self):
        labels = label_conversations(self.rows({"R1": ["PASS", "PASS"], "R2": ["PASS", "PASS"]}))
        self.assertEqual(labels[("m", "S001", "EN")], "CLEAN")

    def test_one_unknown_rule_makes_it_unknown(self):
        labels = label_conversations(self.rows({"R1": ["PASS", "PASS"], "R2": ["UNGRADED"]}))
        self.assertEqual(labels[("m", "S001", "EN")], "UNKNOWN")


class TestDiscordant(unittest.TestCase):
    def test_counts_only_the_pairs_that_disagree(self):
        conv = {("m", "A", "EN"): "CLEAN",     ("m", "A", "HI_ALL"): "LOST",
                ("m", "B", "EN"): "LOST",      ("m", "B", "HI_ALL"): "CLEAN",
                ("m", "C", "EN"): "LOST",      ("m", "C", "HI_ALL"): "LOST"}
        self.assertEqual(discordant(conv, "m", "HI_ALL", "loss"), (1, 1))

    def test_unknown_on_either_side_drops_the_pair(self):
        conv = {("m", "A", "EN"): "UNKNOWN", ("m", "A", "HI_ALL"): "LOST"}
        self.assertEqual(discordant(conv, "m", "HI_ALL", "loss"), (0, 0))

    def test_slip_counts_recovered_as_a_slip_but_loss_does_not(self):
        conv = {("m", "A", "EN"): "CLEAN", ("m", "A", "HI_ALL"): "RECOVERED"}
        self.assertEqual(discordant(conv, "m", "HI_ALL", "slip"), (0, 1))
        self.assertEqual(discordant(conv, "m", "HI_ALL", "loss"), (0, 0))


class TestHolm(unittest.TestCase):
    def test_smallest_is_multiplied_by_the_family_size(self):
        self.assertAlmostEqual(holm([0.01, 0.5, 0.5, 0.5, 0.5])[0], 0.05)

    def test_adjusted_values_never_decrease_along_the_sorted_order(self):
        adj = holm([0.01, 0.02, 0.03])
        self.assertLessEqual(adj[0], adj[1])
        self.assertLessEqual(adj[1], adj[2])

    def test_capped_at_one(self):
        self.assertTrue(all(p <= 1.0 for p in holm([0.4, 0.5, 0.9])))

    def test_order_is_preserved(self):
        self.assertAlmostEqual(holm([0.5, 0.01])[1], 0.02)


if __name__ == "__main__":
    unittest.main()
