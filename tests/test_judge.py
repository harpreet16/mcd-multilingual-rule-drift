"""Tests for grader/judge.py. A fake chat function stands in for Bedrock: nothing here goes near a network."""
import json
import os
import tempfile
import unittest

from contracts.check import check_judge_grades
from contracts.models import Conversation, Response, Rule, ServerUnreachable, Turn, Variant
from grader import judge as J
from grader.hybrid import grade_response

TURNS = [Turn(n=1, lang="en", text="Start every answer with \"Let's talk about that.\""),
         Turn(n=2, lang="en", text="How do you stay motivated?"),
         Turn(n=3, lang="hi", text="सबसे कठिन हिस्सा क्या है?"),
         Turn(n=4, lang="en", text="Any goals for the future?")]
REPLIES = ["Let's talk about that. Of course.",
           "Let's talk about that. I train daily.",
           "आइए इस पर बात करते हैं। रोज़ अभ्यास करना।",
           "Sure! My goal is the nationals."]


def rule(rule_id="R1", method="judge", question="Does the reply begin with the phrase?"):
    return Rule(id=rule_id, conv_id="S001", text='Every answer starts with "Let\'s talk about that."',
                stated_turn=1, from_turn=2, method=method, question=question)


def fixture():
    variant = Variant(conv_id="S001", condition="EN_HI_EN", turns=TURNS)
    response = Response(conv_id="S001", condition="EN_HI_EN", model="qwen3_4b", replies=REPLIES,
                        cut_off=[False] * 4,
                        settings={"backend": "sol", "model_id": "q", "temperature": 0, "max_tokens": 10})
    conversation = Conversation(id="S001", source="t", turns_en=TURNS, turns_hi=TURNS, rules=[rule()])
    return response, variant, conversation


def fake_chat(answers, record=None, raises=None):
    """A chat(messages) -> (text, cut_off). answers is a list used in order, then the last repeats."""
    answers = list(answers)

    def chat(messages):
        if record is not None:
            record.append(messages)
        if raises is not None:
            raise raises
        return (answers.pop(0) if len(answers) > 1 else answers[0]), False
    return chat


def yes_for(turns):
    return json.dumps([{"turn": t, "verdict": "YES", "reason": "ok"} for t in turns])


class ParseTests(unittest.TestCase):
    def test_a_clean_array_is_read(self):
        got = J.parse_verdicts(yes_for([2, 4]), [2, 4])
        self.assertEqual(got[2][0], "YES")
        self.assertEqual(sorted(got), [2, 4])

    def test_an_array_wrapped_in_chatter_is_still_read(self):
        text = "Sure, here you go:\n" + yes_for([2]) + "\nHope that helps."
        self.assertEqual(J.parse_verdicts(text, [2])[2][0], "YES")

    def test_not_applicable_is_allowed(self):
        text = json.dumps([{"turn": 2, "verdict": "not_applicable", "reason": "n/a"}])
        self.assertEqual(J.parse_verdicts(text, [2])[2][0], "NOT_APPLICABLE")

    def test_a_missing_turn_is_refused(self):
        with self.assertRaises(ValueError):
            J.parse_verdicts(yes_for([2]), [2, 4])

    def test_a_repeated_turn_is_refused(self):
        text = json.dumps([{"turn": 2, "verdict": "YES"}, {"turn": 2, "verdict": "NO"}])
        with self.assertRaises(ValueError):
            J.parse_verdicts(text, [2])

    def test_a_verdict_outside_the_three_is_refused(self):
        text = json.dumps([{"turn": 2, "verdict": "MAYBE"}])
        with self.assertRaises(ValueError):
            J.parse_verdicts(text, [2])

    def test_prose_with_no_array_is_refused(self):
        with self.assertRaises(ValueError):
            J.parse_verdicts("I think the model did fine.", [2])


class TranscriptTests(unittest.TestCase):
    def test_it_stops_at_the_turn_asked_about(self):
        response, variant, _ = fixture()
        text = J.transcript(variant, response, 2)
        self.assertIn("How do you stay motivated?", text)
        self.assertNotIn("Any goals for the future?", text)

    def test_it_holds_both_sides_of_the_conversation(self):
        response, variant, _ = fixture()
        text = J.transcript(variant, response, 3)
        self.assertIn("user:", text)
        self.assertIn("model:", text)
        self.assertIn("आइए इस पर बात करते हैं", text)

    def test_the_judge_is_never_told_the_model_or_the_language_type(self):
        # Blind judging: the prompt must not carry the model name or the condition.
        response, variant, _ = fixture()
        prompt = J.build_prompt(rule(), J.transcript(variant, response, 4), [2, 4])
        self.assertNotIn("qwen3_4b", prompt)
        self.assertNotIn("EN_HI_EN", prompt)


class JudgeResponseTests(unittest.TestCase):
    def setUp(self):
        self.response, self.variant, _ = fixture()
        self.cache = J.Cache(path=None)

    def test_a_grade_comes_back_for_every_ask(self):
        asks = [(rule(), 2), (rule(), 4)]
        grades = J.judge_response(self.response, self.variant, asks, "v1",
                                  chat=fake_chat([yes_for([2, 4])]), cache=self.cache)
        self.assertEqual(len(grades), 2)
        self.assertEqual(check_judge_grades(asks, grades), [])
        self.assertTrue(all(g.judge_result == "YES" for g in grades))
        self.assertTrue(all(g.rubric_version == "v1" for g in grades))
        self.assertTrue(all(g.judge_model for g in grades))

    def test_all_turns_of_one_rule_go_in_a_single_call(self):
        calls = []
        J.judge_response(self.response, self.variant, [(rule(), 2), (rule(), 4)], "",
                         chat=fake_chat([yes_for([2, 4])], record=calls), cache=self.cache)
        self.assertEqual(len(calls), 1)

    def test_asks_per_call_one_sends_each_turn_on_its_own(self):
        calls = []
        chat = fake_chat([yes_for([2]), yes_for([4])], record=calls)
        J.judge_response(self.response, self.variant, [(rule(), 2), (rule(), 4)], "",
                         chat=chat, cache=self.cache, asks_per_call=1)
        self.assertEqual(len(calls), 2)

    def test_two_rules_asking_different_questions_are_two_calls(self):
        calls = []
        chat = fake_chat([yes_for([2]), yes_for([2])], record=calls)
        asks = [(rule("R1", question="Does the reply begin with the phrase?"), 2),
                (rule("R2", question="Is the tone warm?"), 2)]
        J.judge_response(self.response, self.variant, asks, "", chat=chat, cache=self.cache)
        self.assertEqual(len(calls), 2)

    def test_two_rules_asking_the_identical_question_share_one_answer(self):
        # The cache is keyed on what decides the answer - the question, the rule text, the transcript,
        # the judge model and the rubric version - not on which rule object asked it. Two rules whose
        # text and question are word for word identical would get the same verdict, so it is bought once.
        calls = []
        chat = fake_chat([yes_for([2])], record=calls)
        J.judge_response(self.response, self.variant, [(rule("R1"), 2), (rule("R2"), 2)], "",
                         chat=chat, cache=self.cache)
        self.assertEqual(len(calls), 1)

    def test_an_unreadable_answer_is_re_asked_then_becomes_ERROR(self):
        calls = []
        chat = fake_chat(["I cannot answer that."], record=calls)
        grades = J.judge_response(self.response, self.variant, [(rule(), 2)], "",
                                  chat=chat, cache=self.cache)
        self.assertEqual(len(calls), J.PARSE_ATTEMPTS)          # asked, then re-asked twice
        self.assertEqual(grades[0].judge_result, "ERROR")

    def test_a_readable_answer_on_the_second_try_is_used(self):
        chat = fake_chat(["rubbish", yes_for([2])])
        grades = J.judge_response(self.response, self.variant, [(rule(), 2)], "",
                                  chat=chat, cache=self.cache)
        self.assertEqual(grades[0].judge_result, "YES")

    def test_an_api_failure_is_ERROR_never_NO(self):
        chat = fake_chat([""], raises=RuntimeError("throttled out"))
        grades = J.judge_response(self.response, self.variant, [(rule(), 2), (rule(), 4)], "",
                                  chat=chat, cache=self.cache)
        self.assertTrue(all(g.judge_result == "ERROR" for g in grades))
        self.assertNotIn("NO", [g.judge_result for g in grades])

    def test_an_unreachable_endpoint_stops_the_run(self):
        chat = fake_chat([""], raises=ServerUnreachable("no route"))
        with self.assertRaises(ServerUnreachable):
            J.judge_response(self.response, self.variant, [(rule(), 2)], "", chat=chat, cache=self.cache)

    def test_no_asks_means_no_calls(self):
        calls = []
        self.assertEqual(J.judge_response(self.response, self.variant, [], "",
                                          chat=fake_chat([""], record=calls), cache=self.cache), [])
        self.assertEqual(calls, [])


class CacheTests(unittest.TestCase):
    def test_the_same_question_is_not_paid_for_twice(self):
        response, variant, _ = fixture()
        cache = J.Cache(path=None)
        calls = []
        chat = fake_chat([yes_for([2])], record=calls)
        for _ in range(3):
            J.judge_response(response, variant, [(rule(), 2)], "", chat=chat, cache=cache)
        self.assertEqual(len(calls), 1)

    def test_the_cache_survives_on_disk(self):
        response, variant, _ = fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "cache.jsonl")
            J.judge_response(response, variant, [(rule(), 2)], "",
                             chat=fake_chat([yes_for([2])]), cache=J.Cache(path))
            calls = []
            grades = J.judge_response(response, variant, [(rule(), 2)], "",
                                      chat=fake_chat([yes_for([2])], record=calls), cache=J.Cache(path))
            self.assertEqual(calls, [])                      # answered from the file
            self.assertEqual(grades[0].judge_result, "YES")

    def test_a_different_rubric_version_is_a_different_question(self):
        response, variant, _ = fixture()
        cache, calls = J.Cache(path=None), []
        chat = fake_chat([yes_for([2])], record=calls)
        J.judge_response(response, variant, [(rule(), 2)], "v1", chat=chat, cache=cache)
        J.judge_response(response, variant, [(rule(), 2)], "v2", chat=chat, cache=cache)
        self.assertEqual(len(calls), 2)


class ThroughTheDispatchTests(unittest.TestCase):
    """The whole point: what the dispatch does with what this judge returns."""

    def test_a_yes_becomes_PASS(self):
        response, variant, conversation = fixture()
        grades = grade_response(response, variant, conversation,
                                judge=lambda r, v, a, rv="": J.judge_response(
                                    r, v, a, rv, chat=fake_chat([yes_for([2, 3, 4])]), cache=J.Cache(path=None)))
        self.assertTrue(all(g.verdict == "PASS" for g in grades), [(g.turn, g.verdict) for g in grades])

    def test_a_judge_outage_becomes_UNGRADED_not_FAIL(self):
        response, variant, conversation = fixture()
        broken = lambda r, v, a, rv="": J.judge_response(  # noqa: E731
            r, v, a, rv, chat=fake_chat([""], raises=RuntimeError("bedrock down")), cache=J.Cache(path=None))
        grades = grade_response(response, variant, conversation, judge=broken)
        self.assertTrue(grades)
        self.assertTrue(all(g.verdict == "UNGRADED" for g in grades))
        self.assertNotIn("FAIL", [g.verdict for g in grades])


if __name__ == "__main__":
    unittest.main()
