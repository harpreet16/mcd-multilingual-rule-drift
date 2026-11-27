"""Turn a model response into grades. This decides, for each rule and turn, who gives the verdict.

Input:  a Response, its Variant, and its Conversation (for the rules)
Output: a list of Grade rows, one per rule per graded turn

The decision, in order. The first line that matches wins.

| situation                                            | verdict        | method            | judge asked |
|------------------------------------------------------|----------------|-------------------|-------------|
| no reply saved for that turn                          | UNGRADED       | none              | no          |
| the reply is empty                                    | UNGRADED       | none              | no          |
| the reply was cut off by the token limit              | UNGRADED       | none              | no          |
| english_only rule, reply is Hindi or mixed            | NOT_APPLICABLE | none              | no          |
| static rule, check passes                             | PASS           | static            | no          |
| static rule, check fails, reply is English            | FAIL           | static            | no          |
| static rule, check fails, reply is Hindi/mixed        | the judge says | static_then_judge | yes         |
| "both" rule, check passes and covers the whole rule   | PASS           | static            | no          |
| "both" rule otherwise (the judge confirms)            | the judge says | static_then_judge | yes         |
| "judge" rule                                          | the judge says | judge             | yes         |
| the check itself is broken (bad settings)             | UNGRADED       | none              | no          |

Side questions are never graded. A rule is graded from rule.from_turn to rule.to_turn, or to the
last turn when to_turn is None.

**Why a static FAIL on a Hindi reply is not final.** A correct Hindi reply can fail an English word
check: the model did what was asked, in Hindi, and the English phrase the check looks for is simply
not there. In the earlier audit, 16 of 18 static disagreements on Hindi replies were false fails.
So the judge decides those, and its answer is final.

**Errors never count as fails.** This is the rule the earlier pipeline broke: it stored 38 judge API
errors as pass=False, which turns an outage into evidence of drift. Here a judge ERROR, a missing
or repeated judge answer, a judge that raises, and a broken check all become UNGRADED. UNGRADED and
NOT_APPLICABLE are never counted as a fail by the analysis, and the share of them is reported beside
every number. The one exception is a model server that cannot be reached at all: that is raised, so
a run stops loudly instead of quietly grading nothing.

**When the judge confirms a "both" rule and then errors**, the verdict is UNGRADED rather than the
static result. Keeping the static PASS but dropping the static FAIL would bias the numbers toward
PASS whenever the judge is flaky. Losing the turn to coverage is the honest cost.

The judge is called once per response with every ask, so one conversation is one call where the
judge supports it. Pass judge= to use a different judge, for example a test double.
"""
from collections import Counter

from contracts.check import check_judge_grades
from contracts.models import Grade, ServerUnreachable
from grader.reply_language import detect_reply_language
from grader.static_checks import run_static_check

VERDICT_FOR_JUDGE = {"YES": "PASS", "NO": "FAIL", "NOT_APPLICABLE": "NOT_APPLICABLE", "ERROR": "UNGRADED"}


def rule_turns(rule, variant):
    """The turn numbers where this rule is graded: its window, without side questions."""
    last = variant.turns[-1].n if variant.turns else 0
    end = rule.to_turn if rule.to_turn is not None else last
    return [t.n for t in variant.turns if rule.from_turn <= t.n <= end and not t.side_question]


def _new_grade(response, rule, turn, reply_lang, **rest):
    return Grade(conv_id=response.conv_id, condition=response.condition, model=response.model,
                 rule_id=rule.id, turn=turn, reply_lang=reply_lang, **rest)


def _skip_reason(reply, cut_off):
    """Why this turn cannot be graded at all, or None when it can."""
    if reply is None:
        return "no reply saved for this turn"
    if not reply.strip():
        return "the reply is empty"
    if cut_off:
        return "the reply was cut off by the token limit"
    return None


# Check types where a PASS settles the rule on its own. A not_contains check that passes has
# proved the forbidden text is absent from the whole reply; that is a fact about the string, not a
# judgement, and a judge that disagrees is simply wrong. In the November run the judge claimed
# commas in 63 of 63 comma-free replies. It cannot count.
CONCLUSIVE_ON_PASS = ("not_contains",)


def _static_pass_is_final(rule):
    """True when a passing static check answers the whole rule, so the judge need not be asked.

    Only when the check covers everything the rule asks for. "The reply does not use any commas"
    is fully covered. "Exactly three suggestions are given and none includes the words 'pilates
    studio'" is not: the check says nothing about the count, so a passing check leaves the first
    half open and the judge still decides. Any rule joining two requirements with "and" is treated
    as uncovered, which errs toward asking the judge.
    """
    if rule.check.get("type") not in CONCLUSIVE_ON_PASS:
        return False
    return " and " not in rule.text.lower()


def plan_grades(response, variant, conversation):
    """Work out every grade that can be settled without the judge, and every ask that needs it.

    Returns (settled_grades, asks, pending), where asks is the list of (rule, turn) pairs for the
    judge and pending maps (rule_id, turn) to the part of the Grade already known.
    """
    replies = {t.n: (response.replies[i] if i < len(response.replies) else None)
               for i, t in enumerate(variant.turns)}
    cut = {t.n: (response.cut_off[i] if i < len(response.cut_off) else False)
           for i, t in enumerate(variant.turns)}
    settled, asks, pending = [], [], {}

    for rule in conversation.rules:
        for turn in rule_turns(rule, variant):
            reply = replies.get(turn)
            reason = _skip_reason(reply, cut.get(turn, False))
            if reason:
                settled.append(_new_grade(response, rule, turn, "unknown",
                                          verdict="UNGRADED", method="none", reason=reason))
                continue
            lang = detect_reply_language(reply)
            if rule.language_scope == "english_only" and lang in ("hi", "mixed"):
                settled.append(_new_grade(response, rule, turn, lang, verdict="NOT_APPLICABLE",
                                          method="none",
                                          reason=f"rule is english_only and the reply is {lang}"))
                continue

            static_result = ""
            if rule.method in ("static", "both"):
                try:
                    static_result = run_static_check(rule, reply, lang)
                except ValueError as error:
                    settled.append(_new_grade(response, rule, turn, lang, verdict="UNGRADED",
                                              method="none", reason=f"the check is broken: {error}"))
                    continue

            if rule.method == "static":
                if static_result == "PASS" or lang in ("en", "unknown"):
                    settled.append(_new_grade(response, rule, turn, lang, verdict=static_result,
                                              method="static", static_result=static_result,
                                              reason=f"static check {static_result.lower()}ed"))
                    continue
                # a static FAIL on a Hindi or mixed reply: the judge decides
            elif rule.method == "both" and static_result == "PASS" and _static_pass_is_final(rule):
                settled.append(_new_grade(response, rule, turn, lang, verdict="PASS",
                                          method="static", static_result=static_result,
                                          reason="static check passed and settles the whole rule"))
                continue
            asks.append((rule, turn))
            pending[(rule.id, turn)] = {
                "rule": rule, "turn": turn, "reply_lang": lang, "static_result": static_result,
                "escalated": rule.method == "static" and static_result == "FAIL",
            }
    return settled, asks, pending


def _grade_from_judge(response, answer, waiting):
    """Build the final Grade for one ask, from the judge's answer."""
    rule, turn = waiting["rule"], waiting["turn"]
    verdict = VERDICT_FOR_JUDGE.get(answer.judge_result, "UNGRADED")
    method = "static_then_judge" if waiting["static_result"] else "judge"
    reason = answer.reason or f"judge said {answer.judge_result}"
    if verdict == "UNGRADED":
        reason = f"judge error, not counted as a fail ({reason})"
    return _new_grade(response, rule, turn, waiting["reply_lang"], verdict=verdict, method=method,
                      static_result=waiting["static_result"], judge_result=answer.judge_result,
                      judge_model=answer.judge_model, rubric_version=answer.rubric_version,
                      escalated=waiting["escalated"], reason=reason)


def _ungraded(response, waiting, reason):
    rule, turn = waiting["rule"], waiting["turn"]
    return _new_grade(response, rule, turn, waiting["reply_lang"], verdict="UNGRADED",
                      method="none", static_result=waiting["static_result"],
                      escalated=waiting["escalated"], reason=reason)


def grade_response(response, variant, conversation, judge=None, rubric_version=""):
    """Return a list of Grade rows for one response, in rule and turn order."""
    settled, asks, pending = plan_grades(response, variant, conversation)
    if not asks:
        return sorted(settled, key=lambda g: (g.rule_id, g.turn))

    if judge is None:
        from grader.judge import judge_response as judge      # imported late: it needs boto3
    try:
        answers = list(judge(response, variant, asks, rubric_version))
    except ServerUnreachable:
        raise                                                  # a dead judge stops the run, loudly
    except Exception as error:                                 # noqa: BLE001 - never a FAIL
        settled += [_ungraded(response, w, f"the judge call failed, not counted as a fail: {error}")
                    for w in pending.values()]
        return sorted(settled, key=lambda g: (g.rule_id, g.turn))

    # An answer is only trusted when it was asked for, came back exactly once, and is readable.
    # A repeated answer is dropped on purpose: there is no way to tell which of the two is meant.
    counts = Counter((answer.rule_id, answer.turn) for answer in answers)
    good = {}
    for answer in answers:
        key = (answer.rule_id, answer.turn)
        if key in pending and counts[key] == 1 and answer.judge_result in VERDICT_FOR_JUDGE:
            good[key] = answer
    problems = check_judge_grades(asks, answers)

    for key, waiting in pending.items():
        if key in good:
            settled.append(_grade_from_judge(response, good[key], waiting))
        else:
            why = "the judge gave no usable answer, not counted as a fail"
            settled.append(_ungraded(response, waiting, f"{why} ({len(problems)} problem(s) in its reply)"))
    return sorted(settled, key=lambda g: (g.rule_id, g.turn))
