"""Turn a model response into grades. This decides, for each rule and turn, who gives the verdict.

Input:  a Response, its Variant, and its Conversation (for the rules)
Output: a list of Grade rows

For each rule, for each turn from from_turn to to_turn (skip side questions):
  1. Empty or cut-off reply: UNGRADED
  2. Static rule: run the static check.
       PASS is final. FAIL on an English reply is final.
       FAIL on a Hindi or mixed reply goes to the judge, and the judge decides
  3. "both" rule: static check first, then the judge confirms
  4. "judge" rule: goes to the judge
  5. Collect every (rule, turn) that needs the judge into one list (asks) and call judge_response once.
     It returns Grade rows with the judge's answer. Set verdict, method, static_result and escalated on them
  6. A judge ERROR, or no answer at all: UNGRADED, never FAIL

Every Grade records the method, the static result, the judge result, the reply language and a reason.
Build and test against a fake judge until judge.py is ready.
"""


def grade_response(response, variant, conversation):
    """Return a list of Grade rows for one response."""
    raise NotImplementedError("Not built yet")
