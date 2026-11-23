"""Tests for the dataset tools. Run: python3 -m unittest discover -s tests -t ."""
import json
import tempfile
import unittest
from pathlib import Path

from contracts.check import check_variant
from contracts.constants import CONDITIONS
from contracts.io import load_conversations, read_jsonl, write_jsonl
from contracts.models import Conversation, Rule, Turn
from dataset.compile import compile_dataset, warn_more
from dataset.hindi_checks import check_turn_pair
from dataset.import_xlsx import build_conversations, new_id
from dataset.remix import remix, remix_all

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "contracts" / "examples" / "conversation.json"


def example():
    return Conversation.from_dict(json.loads(EXAMPLE.read_text(encoding="utf-8")))


def six_turn_conversation(stated_turns=(1, 5)):
    turns_en = [Turn(n, "en", f"English {n}") for n in range(1, 7)]
    turns_hi = [Turn(n, "hi", f"हिंदी {n}") for n in range(1, 7)]
    rules = [Rule(id=f"R{t}", conv_id="S001", text="x", stated_turn=t, from_turn=t, method="judge", question="q?")
             for t in stated_turns]
    return Conversation(id="S001", source="test", turns_en=turns_en, turns_hi=turns_hi, rules=rules)


def langs(variant):
    return " ".join(t.lang for t in variant.turns)


class RemixTests(unittest.TestCase):
    def test_the_four_patterns_match_the_documented_example(self):
        conv = six_turn_conversation()
        self.assertEqual(langs(remix(conv, "EN")), "en en en en en en")
        self.assertEqual(langs(remix(conv, "EN_HI_EN")), "en hi hi hi en en")
        self.assertEqual(langs(remix(conv, "HI_ALL")), "hi hi hi hi hi hi")
        self.assertEqual(langs(remix(conv, "HI_EN_HI")), "hi en en en en hi")

    def test_turns_are_copied_unaltered(self):
        conv = six_turn_conversation()
        variant = remix(conv, "EN_HI_EN")
        self.assertEqual([t.text for t in variant.turns], ["English 1", "हिंदी 2", "हिंदी 3", "हिंदी 4", "English 5", "English 6"])
        variant.turns[0].text = "changed"
        self.assertEqual(conv.turns_en[0].text, "English 1")    # the conversation is not touched

    def test_side_question_flag_follows_the_turn(self):
        conv = example()
        for condition in CONDITIONS:
            self.assertEqual(check_variant(remix(conv, condition), conv), [])

    def test_en_hi_en_needs_rules(self):
        conv = six_turn_conversation(stated_turns=())
        with self.assertRaises(ValueError):
            remix(conv, "EN_HI_EN")
        self.assertEqual(langs(remix(conv, "HI_EN_HI")), "hi en en en en hi")

    def test_unknown_condition(self):
        with self.assertRaises(ValueError):
            remix(example(), "EN_ONLY")

    def test_remix_all_gives_every_condition(self):
        variants = remix_all([example()])
        self.assertEqual([v.condition for v in variants], list(CONDITIONS))


def sheet_rows(conv_id="LIS-S01"):
    """The rows of a tiny spreadsheet: 3 turns, the second is an added side question."""
    texts = {"en": ["Rule: start with \"Hello there.\"", "Quick side question: what is a paper clip for?", "Tell me about tea."],
             "hi": ["नियम: \"Hello there.\" से शुरू करें", "बीच में एक छोटा सवाल: पेपर क्लिप किस काम आता है?", "मुझे चाय के बारे में बताइए।"]}
    rows = []
    for version, lang, name in (("All English", "en", "English"), ("All Hindi", "hi", "Hindi"), ("English + Hindi", "en", "English")):
        for n, text in enumerate(texts[lang], start=1):
            rows.append({"Conversation ID": conv_id, "Version": version, "Turn": str(n), "Language": name,
                         "Turn text": text, "Source": "MultiChallenge\nRecord: x"})
    return rows


class ImportTests(unittest.TestCase):
    def test_new_id(self):
        self.assertEqual(new_id("LIS-S01"), "S001")
        self.assertEqual(new_id("LIS-S100"), "S100")
        with self.assertRaises(ValueError):
            new_id("abc")

    def test_build_conversation(self):
        convs, problems = build_conversations(sheet_rows())
        self.assertEqual(problems, [])
        conv = Conversation.from_dict(convs[0])
        self.assertEqual(conv.id, "S001")
        self.assertEqual([t.side_question for t in conv.turns_en], [False, True, False])
        self.assertEqual([t.side_question for t in conv.turns_hi], [False, True, False])
        self.assertEqual(conv.rules, [])
        self.assertIn("LIS-S01", conv.source)

    def test_mixed_version_that_does_not_match_is_reported(self):
        rows = sheet_rows()
        rows[-1]["Turn text"] = "something else"
        _, problems = build_conversations(rows)
        self.assertEqual(len(problems), 1)

    def test_hindi_fix_replaces_one_hindi_turn(self):
        fixes = [{"conv_id": "S001", "turn": 1, "text": "नया हिंदी पाठ"}]
        convs, problems = build_conversations(sheet_rows(), fixes)
        self.assertEqual(problems, [])
        self.assertEqual(convs[0]["turns_hi"][0]["text"], "नया हिंदी पाठ")
        self.assertEqual(convs[0]["turns_en"][0]["text"], 'Rule: start with "Hello there."')

    def test_missing_version_is_reported(self):
        rows = [r for r in sheet_rows() if r["Version"] != "All Hindi"]
        convs, problems = build_conversations(rows)
        self.assertEqual(convs, [])
        self.assertTrue(problems)


class HindiChecksTests(unittest.TestCase):
    def pair(self, en, hi):
        return check_turn_pair(Turn(1, "en", en), Turn(1, "hi", hi))

    def test_good_pair(self):
        self.assertEqual(self.pair('Say "Let\'s talk about that." in 3 words', 'इसे 3 शब्दों में "Let\'s talk about that." कहें'), [])

    def test_digit_change(self):
        self.assertEqual(len(self.pair("Give 3 tips", "३ सुझाव दीजिए")), 2)    # Devanagari digit, and the 3 is missing

    def test_extra_digit_in_hindi_is_fine(self):
        self.assertEqual(self.pair("Sign at noon", "दोपहर 12 बजे हस्ताक्षर करें"), [])

    def test_single_quoted_word_may_be_translated(self):
        self.assertEqual(self.pair('A "classic" film', 'एक "क्लासिक" फ़िल्म'), [])

    def test_quote_not_kept(self):
        self.assertTrue(any("quoted" in p for p in self.pair('Say "Hello there."', "नमस्ते कहें")))

    def test_long_quote_may_be_translated(self):
        quote = " ".join(["word"] * 12)
        self.assertEqual(self.pair(f'He wrote "{quote}"', "उसने कुछ लिखा"), [])

    def test_zero_width_and_no_devanagari(self):
        problems = self.pair("Hello", "Hello​")
        self.assertTrue(any("zero-width" in p for p in problems))
        self.assertTrue(any("no Devanagari" in p for p in problems))
        self.assertFalse(any("copy" in p for p in problems))    # it differs by the zero-width character


class CompileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.out = self.dir / "out.jsonl"
        data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        data["id"] = "S001"
        for rule in data["rules"]:
            rule["conv_id"] = "S001"
        self.data = data

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, rows):
        path = self.dir / name
        write_jsonl(path, rows)
        return path

    def compile(self, *files, **kwargs):
        return compile_dataset([str(f) for f in files], str(self.out), log=lambda *_: None, **kwargs)

    def test_good_file_is_written(self):
        self.assertEqual(self.compile(self.write("a.jsonl", [self.data])), [])
        self.assertEqual(len(read_jsonl(self.out)), 1)

    def test_duplicate_id_across_files_blocks_the_output(self):
        a, b = self.write("a.jsonl", [self.data]), self.write("b.jsonl", [self.data])
        problems = self.compile(a, b)
        self.assertTrue(any("is in both" in p for p in problems))
        self.assertFalse(self.out.exists())

    def test_bad_id_format(self):
        self.data["id"] = "LIS-S01"
        self.assertTrue(any("not in the form" in p for p in self.compile(self.write("a.jsonl", [self.data]))))

    def test_conversation_without_rules_blocks_the_output(self):
        self.data["rules"] = []
        self.assertTrue(any("no rules" in p for p in self.compile(self.write("a.jsonl", [self.data]))))
        self.assertFalse(self.out.exists())

    def test_rules_file_is_merged(self):
        rules = self.data["rules"]
        self.data["rules"] = []
        problems = self.compile(self.write("a.jsonl", [self.data]), rules_path=str(self.write("rules.jsonl", rules)))
        self.assertEqual(problems, [])
        self.assertEqual(len(read_jsonl(self.out)[0]["rules"]), len(rules))

    def test_rule_for_unknown_conversation(self):
        rule = dict(self.data["rules"][0], conv_id="S099")
        problems = self.compile(self.write("a.jsonl", [self.data]), rules_path=str(self.write("rules.jsonl", [rule])))
        self.assertTrue(any("unknown conversation" in p for p in problems))

    def test_output_is_sorted_by_id(self):
        other = json.loads(json.dumps(self.data))
        other["id"] = "S000"
        for rule in other["rules"]:
            rule["conv_id"] = "S000"
        self.compile(self.write("a.jsonl", [self.data, other]))
        self.assertEqual([c["id"] for c in read_jsonl(self.out)], ["S000", "S001"])


class WarningTests(unittest.TestCase):
    def rule(self, rid, text, stated=1, from_turn=2, to_turn=None):
        return Rule(id=rid, conv_id="S001", text="x", stated_turn=stated, from_turn=from_turn, to_turn=to_turn,
                    method="static", check={"type": "starts_with", "text": text})

    def test_conflicting_rules_are_flagged(self):
        conv = six_turn_conversation(stated_turns=())
        conv.rules = [self.rule("A", "Hi."), self.rule("B", "Hello.", stated=2, from_turn=3)]
        self.assertTrue(any("conflict" in w for w in warn_more(conv)))

    def test_cancelled_rule_does_not_conflict(self):
        conv = six_turn_conversation(stated_turns=())
        conv.rules = [self.rule("A", "Hi.", to_turn=2), self.rule("B", "Hello.", stated=3, from_turn=4)]
        self.assertFalse(any("conflict" in w for w in warn_more(conv)))

    def test_rule_with_no_hindi_turn_in_en_hi_en(self):
        conv = six_turn_conversation(stated_turns=())
        conv.rules = [self.rule("A", "Hi.", stated=5, from_turn=5)]    # turns 5 and 6 are English in EN_HI_EN
        self.assertTrue(any("no Hindi turn" in w for w in warn_more(conv)))

    def test_hindi_turn_between_statement_and_check_is_a_switch(self):
        conv = six_turn_conversation(stated_turns=())
        conv.turns_en[1].side_question = conv.turns_hi[1].side_question = True    # turn 2: a Hindi side question
        conv.rules = [self.rule("A", "Hi.", stated=1, from_turn=1, to_turn=3)]    # checked at 1 and 3, after turn 2
        self.assertFalse(any("no Hindi turn" in w for w in warn_more(conv)))

    def test_checked_word_missing_from_the_hindi_turn(self):
        conv = six_turn_conversation(stated_turns=())
        conv.turns_en[0].text = "Do not use the words: experience, volunteer."
        conv.turns_hi[0].text = "इन शब्दों का प्रयोग न करें: अनुभव, स्वयंसेवक।"
        conv.rules = [self.rule("A", "experience", stated=1, from_turn=2)]
        self.assertTrue(any("not in the Hindi turn" in w for w in warn_more(conv)))
        conv.turns_hi[0].text = "इन शब्दों का प्रयोग न करें: experience, volunteer।"
        self.assertFalse(any("not in the Hindi turn" in w for w in warn_more(conv)))

    def test_one_shot_rule_is_counted_not_warned(self):
        conv = six_turn_conversation(stated_turns=())
        conv.rules = [self.rule("A", "Hi.", stated=5, from_turn=5, to_turn=5)]    # ends before the last turn
        from dataset.compile import _runs_to_end
        self.assertEqual(_runs_to_end(conv), set())
        conv.rules[0].to_turn = None
        self.assertEqual(_runs_to_end(conv), {"A"})

    def test_generic_reference_to_the_previous_reply_is_not_flagged(self):
        conv = six_turn_conversation()
        conv.turns_en[2].text = "Please summarise your previous response."
        self.assertFalse(any("turn 3" in w for w in warn_more(conv)))

    def test_pointing_at_assistant(self):
        conv = six_turn_conversation()
        conv.turns_en[2].text = "As you mentioned, tea is good."
        self.assertTrue(any("turn 3" in w for w in warn_more(conv)))


FROZEN = ROOT / "dataset" / "data" / "conversations.jsonl"


@unittest.skipUnless(FROZEN.exists(), "the compiled dataset is not in the repo")
class CompiledDatasetTests(unittest.TestCase):
    def test_every_conversation_passes_the_contract_and_remixes(self):
        from contracts.check import check_conversation
        convs = load_conversations(FROZEN)
        self.assertEqual(len(convs), 100)
        for conv in convs:
            self.assertEqual(check_conversation(conv), [], conv.id)
        self.assertEqual(len(remix_all(convs)), 400)


class FollowUpTests(unittest.TestCase):
    def conv(self, stated, to_turn=None, n=5):
        turns_en = [Turn(i, "en", f"English {i}") for i in range(1, n + 1)]
        turns_hi = [Turn(i, "hi", f"हिंदी {i}") for i in range(1, n + 1)]
        rule = Rule(id="R1", conv_id="S001", text="x", stated_turn=stated, from_turn=stated, to_turn=to_turn,
                    method="judge", question="q?")
        return Conversation(id="S001", source="t", turns_en=turns_en, turns_hi=turns_hi, rules=[rule])

    def test_rule_stated_at_the_last_turn_needs_follow_ups(self):
        from dataset.follow_ups import needs_follow_ups
        self.assertTrue(needs_follow_ups(self.conv(stated=5)))
        self.assertFalse(needs_follow_ups(self.conv(stated=1)))

    def test_cancelled_rule_is_left_alone(self):
        from dataset.follow_ups import needs_follow_ups
        self.assertFalse(needs_follow_ups(self.conv(stated=4, to_turn=4)))

    def test_follow_ups_make_the_rule_testable_after_a_switch(self):
        from dataset.follow_ups import add_follow_ups, needs_follow_ups, untested_rules
        conv = self.conv(stated=5, to_turn=5)
        self.assertEqual(untested_rules(conv), ["R1"])
        add_follow_ups(conv)
        self.assertEqual(len(conv.turns_en), 7)
        self.assertEqual(len(conv.turns_hi), 7)
        self.assertIsNone(conv.rules[0].to_turn)
        self.assertEqual(untested_rules(conv), [])
        self.assertFalse(needs_follow_ups(conv))
        self.assertEqual(langs(remix(conv, "EN_HI_EN")), "hi hi hi hi en hi en")
        for condition in CONDITIONS:
            self.assertEqual(check_variant(remix(conv, condition), conv), [])
