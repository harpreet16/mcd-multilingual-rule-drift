"""Tests for the shared file formats. Run: python3 -m unittest discover -s tests -t ."""
import json
import tempfile
import unittest
from pathlib import Path

from contracts.check import (check_conversation, check_grade, check_judge_grades,
                             check_response, check_results, check_variant, warn_conversation)
from contracts.io import append_jsonl, read_jsonl, write_jsonl
from contracts.models import Conversation, Grade, Response, Variant

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "contracts" / "examples"


def load_example(name):
    with open(EXAMPLES / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


class ExampleFilesTests(unittest.TestCase):
    def test_every_example_passes_its_check(self):
        conv = Conversation.from_dict(load_example("conversation"))
        variant = Variant.from_dict(load_example("variant"))
        response = Response.from_dict(load_example("response"))
        grade = Grade.from_dict(load_example("grade"))
        self.assertEqual(check_conversation(conv), [])
        self.assertEqual(check_variant(variant, conv), [])
        self.assertEqual(check_response(response, variant), [])
        self.assertEqual(check_grade(grade), [])
        self.assertEqual(check_results(load_example("results")), [])
        self.assertEqual(warn_conversation(conv), [])


class CheckerCatchesProblemsTests(unittest.TestCase):
    def setUp(self):
        self.data = load_example("conversation")

    def problems(self):
        return check_conversation(Conversation.from_dict(self.data))

    def test_rule_window_outside_the_conversation(self):
        self.data["rules"][0]["from_turn"] = 99
        self.assertTrue(any("after the last turn" in p for p in self.problems()))

    def test_static_rule_without_a_check_type(self):
        self.data["rules"][0]["check"] = {}
        self.assertTrue(any("needs check with a type" in p for p in self.problems()))

    def test_judge_rule_without_a_question(self):
        self.data["rules"][1]["question"] = ""
        self.assertTrue(any("needs a question" in p for p in self.problems()))

    def test_hindi_and_english_turn_counts_differ(self):
        self.data["turns_hi"].pop()
        self.assertTrue(any("Hindi turns" in p for p in self.problems()))

    def test_duplicate_rule_id(self):
        self.data["rules"][1]["id"] = self.data["rules"][0]["id"]
        self.assertTrue(any("used twice" in p for p in self.problems()))

    def test_variant_text_changed(self):
        conv = Conversation.from_dict(load_example("conversation"))
        variant = Variant.from_dict(load_example("variant"))
        variant.turns[1].text = variant.turns[1].text + " extra"
        self.assertTrue(any("does not match" in p for p in check_variant(variant, conv)))

    def test_variant_language_tag_wrong(self):
        conv = Conversation.from_dict(load_example("conversation"))
        variant = Variant.from_dict(load_example("variant"))
        variant.turns[1].lang = "en"   # the text is Hindi but the tag says English
        self.assertTrue(any("does not match" in p for p in check_variant(variant, conv)))

    def test_warns_when_a_rule_has_only_one_testable_turn(self):
        self.data["rules"][1]["to_turn"] = 5   # rule R2 now covers turn 5 only
        warnings = warn_conversation(Conversation.from_dict(self.data))
        self.assertTrue(any("only 1 testable" in w for w in warnings))

    def test_judge_must_answer_every_ask_once(self):
        conv = Conversation.from_dict(load_example("conversation"))
        rule = conv.rules[1]
        asks = [(rule, 5), (rule, 6)]
        def judged(turn, answer):
            return Grade(conv.id, "EN", "mock-model", rule.id, turn, verdict="", method="judge",
                         reply_lang="en", judge_result=answer)
        good = [judged(5, "YES"), judged(6, "NO")]
        self.assertEqual(check_judge_grades(asks, good), [])
        self.assertTrue(any("no answer" in p for p in check_judge_grades(asks, good[:1])))
        self.assertTrue(any("twice" in p for p in check_judge_grades(asks, good + good[:1])))
        bad = [judged(5, "MAYBE"), judged(6, "NO")]
        self.assertTrue(any("MAYBE" in p for p in check_judge_grades(asks, bad)))

    def test_bad_verdict_in_a_grade(self):
        grade = Grade.from_dict(load_example("grade"))
        grade.verdict = "MAYBE"
        self.assertTrue(any("verdict" in p for p in check_grade(grade)))

    def test_bad_judge_result_in_a_grade(self):
        grade = Grade.from_dict(load_example("grade"))
        grade.judge_result = "MAYBE"
        self.assertTrue(any("judge_result" in p for p in check_grade(grade)))

    def test_response_missing_a_setting(self):
        response = Response.from_dict(load_example("response"))
        del response.settings["model_id"]
        self.assertTrue(any("model_id" in p for p in check_response(response)))


class FileReadWriteTests(unittest.TestCase):
    def test_write_append_and_read_back(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            write_jsonl(path, [{"a": 1}, {"b": "हिंदी"}])
            append_jsonl(path, {"c": 3})
            self.assertEqual(read_jsonl(path), [{"a": 1}, {"b": "हिंदी"}, {"c": 3}])


if __name__ == "__main__":
    unittest.main()
