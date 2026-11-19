"""The LLM judge: Claude Haiku 3.5 on Bedrock, run from Sol through the client in models/bedrock.

Input:  a Response, its Variant, a list of asks (rule, turn) pairs, and a rubric version
Output: one Grade for every ask, with judge_result (YES, NO, NOT_APPLICABLE or ERROR), a short reason,
        judge_model and rubric_version filled in. hybrid.py fills in the rest (verdict, method, static result)

How it answers is up to you. Both are allowed:
  - one call for each ask (small, simple, easy to cache)
  - one call for the whole conversation that returns a verdict for every ask in a fixed format
Try both on about 50 conversations and compare with a hand-checked sample. Use the one-call version if it
agrees on about 90% of turn verdicts, because it needs fewer calls.

Rules:
  - Never trust a verdict that is missing, repeated or invalid (contracts.check.check_judge_grades finds them).
    Retry, then return ERROR
  - Do not give an overall conversation verdict. The analysis works that out
  - Use the house rules for each rubric group, temperature 0, and never show the model name or language type
  - Retry a failed call twice, then return ERROR
  - Cache answers by question, text, judge model and rubric version
  - Provide a script that judges a whole file of responses in bulk
"""


def judge_response(response, variant, asks, rubric_version):
    """Return one Grade for every (rule, turn) pair in asks."""
    raise NotImplementedError("Not built yet")
