"""Tests for the cut-off retry in the Sol client. No server is needed."""
import unittest

from models.sol.client import reply_with_bigger_budget


class BiggerBudgetTests(unittest.TestCase):
    def test_a_complete_reply_is_returned_at_once(self):
        budgets = []
        def post(messages, budget):
            budgets.append(budget)
            return "done", False
        text, cut_off = reply_with_bigger_budget(post, [], 4096, 8192)
        self.assertEqual(budgets, [4096])
        self.assertFalse(cut_off)

    def test_a_cut_off_reply_is_asked_again_with_double_the_tokens(self):
        budgets = []
        def post(messages, budget):
            budgets.append(budget)
            return ("partial", True) if budget == 4096 else ("finished", False)
        text, cut_off = reply_with_bigger_budget(post, [], 4096, 8192)
        self.assertEqual(budgets, [4096, 8192])
        self.assertEqual(text, "finished")

    def test_still_cut_off_at_the_cap_is_returned_marked_as_cut_off(self):
        budgets = []
        def post(messages, budget):
            budgets.append(budget)
            return "partial", True
        text, cut_off = reply_with_bigger_budget(post, [], 4096, 8192)
        self.assertEqual(budgets, [4096, 8192])   # it never goes above the cap
        self.assertTrue(cut_off)

    def test_an_empty_reply_is_asked_again(self):
        budgets = []
        def post(messages, budget):
            budgets.append(budget)
            return ("", False) if budget == 4096 else ("text", False)
        self.assertEqual(reply_with_bigger_budget(post, [], 4096, 8192)[0], "text")


if __name__ == "__main__":
    unittest.main()
