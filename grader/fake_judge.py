"""A stand-in judge for tests and dry runs. It never calls Bedrock.

The real judge is grader/judge.py (Shashwat's module). This one keeps the same shape, so the
dispatch can be built and tested before it lands, and so `python3 -m grader --fake-judge` can grade
a whole file with no credentials.

    judge = make_fake_judge(default="YES", answers={("R1", 4): "NO"}, error_on={("R2", 5)})
    grades = grade_response(response, variant, conversation, judge=judge)
"""
from contracts.models import Grade

JUDGE_MODEL = "fake-judge"


def make_fake_judge(default="YES", answers=None, error_on=(), raises=None, skip=()):
    """Return a judge function.

    default   what to answer when nothing more specific is set
    answers   {(rule_id, turn): "YES"|"NO"|"NOT_APPLICABLE"|"ERROR"}
    error_on  asks that come back as ERROR
    raises    an exception instance to raise instead of answering
    skip      asks to leave out of the answer, to test a judge that loses one
    """
    answers = dict(answers or {})
    error_on, skip = set(error_on), set(skip)

    def judge_response(response, variant, asks, rubric_version=""):
        if raises is not None:
            raise raises
        out = []
        for rule, turn in asks:
            key = (rule.id, turn)
            if key in skip:
                continue
            result = "ERROR" if key in error_on else answers.get(key, default)
            out.append(Grade(conv_id=response.conv_id, condition=response.condition,
                             model=response.model, rule_id=rule.id, turn=turn,
                             verdict="", method="judge", reply_lang="unknown",
                             judge_result=result, judge_model=JUDGE_MODEL,
                             rubric_version=rubric_version, reason=f"fake judge said {result}"))
        return out

    return judge_response
