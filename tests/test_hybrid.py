"""Tests for grader/hybrid.py: who grades what, and what happens when the judge misbehaves."""
import unittest

from contracts.check import check_grade
from contracts.models import Conversation, Response, Rule, ServerUnreachable, Turn, Variant
from grader.fake_judge import make_fake_judge
from grader.hybrid import grade_response, rule_turns

HINDI = "यह एक पूरा हिंदी उत्तर है और इसमें कोई अंग्रेज़ी वाक्य नहीं है।"


def turn(n, side=False, lang="en"):
    return Turn(n=n, lang=lang, text=f"turn {n}", side_question=side)


def make(rules, turns=None, replies=None, cut_off=None, condition="EN"):
    """A conversation, its variant and one response, all lined up."""
    turns = turns or [turn(1), turn(2), turn(3, side=True), turn(4)]
    replies = ["reply one", "reply two", "reply three", "reply four"] if replies is None else replies
    conversation = Conversation(id="S001", source="test", turns_en=turns, turns_hi=turns, rules=rules)
    variant = Variant(conv_id="S001", condition=condition, turns=turns)
    response = Response(conv_id="S001", condition=condition, model="mock", replies=replies,
                        cut_off=cut_off if cut_off is not None else [False] * len(replies),
                        settings={"backend": "mock", "model_id": "mock", "temperature": 0, "max_tokens": 10})
    return conversation, variant, response


def rule(method="judge", check=None, scope="any", from_turn=2, to_turn=None, rule_id="R1"):
    return Rule(id=rule_id, conv_id="S001", text="a rule", stated_turn=1, from_turn=from_turn,
                to_turn=to_turn, method=method, check=check or {}, question="Did it?",
                language_scope=scope)


def by_turn(grades):
    return {g.turn: g for g in grades}


class WindowTests(unittest.TestCase):
    def test_side_questions_are_never_graded(self):
        conversation, variant, response = make([rule()])
        grades = grade_response(response, variant, conversation, judge=make_fake_judge())
        self.assertEqual(sorted(g.turn for g in grades), [2, 4])

    def test_the_window_is_respected(self):
        conversation, variant, response = make([rule(from_turn=2, to_turn=2)])
        grades = grade_response(response, variant, conversation, judge=make_fake_judge())
        self.assertEqual([g.turn for g in grades], [2])

    def test_rule_turns_skips_side_questions(self):
        _, variant, _ = make([])
        self.assertEqual(rule_turns(rule(from_turn=1), variant), [1, 2, 4])


class CannotGradeTests(unittest.TestCase):
    def test_a_missing_reply_is_ungraded(self):
        conversation, variant, response = make([rule()], replies=["only one"])
        grades = by_turn(grade_response(response, variant, conversation, judge=make_fake_judge()))
        self.assertEqual(grades[4].verdict, "UNGRADED")
        self.assertEqual(grades[4].method, "none")

    def test_an_empty_reply_is_ungraded(self):
        conversation, variant, response = make([rule()], replies=["a", "   ", "c", "d"])
        self.assertEqual(by_turn(grade_response(response, variant, conversation,
                                                judge=make_fake_judge()))[2].verdict, "UNGRADED")

    def test_a_cut_off_reply_is_ungraded(self):
        conversation, variant, response = make([rule()], cut_off=[False, True, False, False])
        self.assertEqual(by_turn(grade_response(response, variant, conversation,
                                                judge=make_fake_judge()))[2].verdict, "UNGRADED")

    def test_a_broken_check_is_ungraded_not_failed(self):
        conversation, variant, response = make([rule(method="static", check={"type": "no_such_check", "text": "x"})])
        grades = grade_response(response, variant, conversation, judge=make_fake_judge())
        self.assertTrue(all(g.verdict == "UNGRADED" for g in grades))
        self.assertIn("the check is broken", grades[0].reason)


class EnglishOnlyTests(unittest.TestCase):
    def test_a_hindi_reply_is_not_applicable(self):
        conversation, variant, response = make([rule(scope="english_only")],
                                               replies=["a", HINDI, "c", "d"])
        grades = by_turn(grade_response(response, variant, conversation, judge=make_fake_judge()))
        self.assertEqual(grades[2].verdict, "NOT_APPLICABLE")
        self.assertEqual(grades[2].reply_lang, "hi")

    def test_an_english_reply_is_still_graded(self):
        conversation, variant, response = make([rule(scope="english_only")])
        self.assertEqual(by_turn(grade_response(response, variant, conversation,
                                                judge=make_fake_judge()))[2].verdict, "PASS")


class StaticTests(unittest.TestCase):
    def test_a_passing_check_needs_no_judge(self):
        conversation, variant, response = make([rule(method="static", check={"type": "not_contains", "text": "zebra"})])
        judge = make_fake_judge(default="NO")       # would fail everything if it were called
        grades = grade_response(response, variant, conversation, judge=judge)
        self.assertTrue(all(g.verdict == "PASS" and g.method == "static" for g in grades))

    def test_a_failing_check_on_an_english_reply_is_final(self):
        conversation, variant, response = make([rule(method="static", check={"type": "not_contains", "text": "reply"})])
        grades = grade_response(response, variant, conversation, judge=make_fake_judge(default="YES"))
        self.assertTrue(all(g.verdict == "FAIL" and g.method == "static" for g in grades))
        self.assertFalse(any(g.escalated for g in grades))

    def test_a_failing_check_on_a_hindi_reply_goes_to_the_judge(self):
        conversation, variant, response = make(
            [rule(method="static", check={"type": "starts_with", "text": "Let's talk about that."})],
            replies=["a", HINDI, "c", HINDI])
        grades = by_turn(grade_response(response, variant, conversation, judge=make_fake_judge(default="YES")))
        self.assertEqual(grades[2].verdict, "PASS")                 # the judge overruled the check
        self.assertEqual(grades[2].static_result, "FAIL")
        self.assertEqual(grades[2].method, "static_then_judge")
        self.assertTrue(grades[2].escalated)

    def test_a_both_rule_always_asks_the_judge(self):
        conversation, variant, response = make([rule(method="both", check={"type": "not_contains", "text": "zebra"})])
        grades = grade_response(response, variant, conversation, judge=make_fake_judge(default="NO"))
        # the static check passes, but the judge confirms and its answer is final
        self.assertTrue(all(g.static_result == "PASS" for g in grades))
        self.assertTrue(all(g.verdict == "FAIL" and g.method == "static_then_judge" for g in grades))


class ErrorsAreNeverFailsTests(unittest.TestCase):
    def test_a_judge_error_is_ungraded(self):
        conversation, variant, response = make([rule()])
        judge = make_fake_judge(error_on=[("R1", 2), ("R1", 4)])
        grades = grade_response(response, variant, conversation, judge=judge)
        self.assertTrue(all(g.verdict == "UNGRADED" for g in grades))
        self.assertNotIn("FAIL", [g.verdict for g in grades])

    def test_a_judge_that_raises_leaves_everything_ungraded(self):
        conversation, variant, response = make([rule()])
        judge = make_fake_judge(raises=RuntimeError("bedrock is unhappy"))
        grades = grade_response(response, variant, conversation, judge=judge)
        self.assertTrue(all(g.verdict == "UNGRADED" for g in grades))
        self.assertIn("not counted as a fail", grades[0].reason)

    def test_a_missing_answer_is_ungraded(self):
        conversation, variant, response = make([rule()])
        grades = by_turn(grade_response(response, variant, conversation,
                                        judge=make_fake_judge(skip=[("R1", 4)])))
        self.assertEqual(grades[2].verdict, "PASS")
        self.assertEqual(grades[4].verdict, "UNGRADED")

    def test_a_repeated_answer_is_not_trusted(self):
        conversation, variant, response = make([rule()])
        base = make_fake_judge(default="YES")

        def doubling_judge(resp, var, asks, rubric_version=""):
            answers = base(resp, var, asks, rubric_version)
            return answers + answers[:1]            # the first ask answered twice

        grades = by_turn(grade_response(response, variant, conversation, judge=doubling_judge))
        self.assertEqual(grades[2].verdict, "UNGRADED")
        self.assertEqual(grades[4].verdict, "PASS")

    def test_an_unreachable_server_stops_the_run(self):
        conversation, variant, response = make([rule()])
        judge = make_fake_judge(raises=ServerUnreachable("no route to bedrock"))
        with self.assertRaises(ServerUnreachable):
            grade_response(response, variant, conversation, judge=judge)

    def test_an_unreadable_judge_answer_is_ungraded(self):
        conversation, variant, response = make([rule()])
        judge = make_fake_judge(default="probably?")      # not one of the four allowed answers
        grades = grade_response(response, variant, conversation, judge=judge)
        self.assertTrue(all(g.verdict == "UNGRADED" for g in grades))


class ContractTests(unittest.TestCase):
    def test_every_grade_passes_the_contract_check(self):
        rules = [rule(rule_id="R1", method="judge"),
                 rule(rule_id="R2", method="static", check={"type": "not_contains", "text": "reply"}),
                 rule(rule_id="R3", method="both", check={"type": "starts_with", "text": "reply"})]
        conversation, variant, response = make(rules, replies=["a", HINDI, "c", "reply four"])
        grades = grade_response(response, variant, conversation,
                                judge=make_fake_judge(default="YES"), rubric_version="v1")
        self.assertTrue(grades)
        for grade in grades:
            self.assertEqual(check_grade(grade), [], f"{grade} broke the contract")

    def test_grades_come_back_in_a_fixed_order(self):
        rules = [rule(rule_id="R2"), rule(rule_id="R1")]
        conversation, variant, response = make(rules)
        grades = grade_response(response, variant, conversation, judge=make_fake_judge())
        self.assertEqual([(g.rule_id, g.turn) for g in grades], [("R1", 2), ("R1", 4), ("R2", 2), ("R2", 4)])


if __name__ == "__main__":
    unittest.main()
