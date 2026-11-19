"""Final numbers and tests. The exact steps are in PLAN.md under "Analysis steps".

Input:  grades.jsonl and the conversations
Output: results.json (format checked by contracts.check.check_results)

The rule labels (KEPT, RECOVERED, LOST, UNKNOWN) are worked out here and are not saved to their own file.
graphs.py calls label_rule to get them.

Done when the numbers match the earlier S01 to S20 results on the same data.
"""


def label_rule(grades):
    """grades: the Grade rows for one rule in one response, in turn order.

    Return (label, first_lapse_turn, recovery_turn). The label is one of RULE_LABELS.
    """
    raise NotImplementedError("Not built yet")


def compute_results(grades, conversations):
    """Return the results dict described in contracts/check.py (RESULT_KEYS) and PLAN.md."""
    raise NotImplementedError("Not built yet")
