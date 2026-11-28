"""Tests for grader/static_checks.py and grader/reply_language.py."""
import unittest

from contracts.models import Rule
from grader.reply_language import detect_reply_language, devanagari_share, romanised_hindi_words
from grader.static_checks import run_static_check


def rule(check_type, text, method="static", scope="any"):
    return Rule(id="R1", conv_id="S001", text="a rule", stated_turn=1, from_turn=2,
                method=method, check={"type": check_type, "text": text}, language_scope=scope)


class NotContainsTests(unittest.TestCase):
    def test_the_word_itself_fails(self):
        self.assertEqual(run_static_check(rule("not_contains", "sleep"), "You need sleep tonight", "en"), "FAIL")

    def test_case_does_not_matter(self):
        self.assertEqual(run_static_check(rule("not_contains", "sleep"), "SLEEP is important", "en"), "FAIL")

    def test_a_longer_word_containing_it_passes(self):
        # IFEval's own forbidden-word check matches on word boundaries, so "sleeping" does not
        # break a rule against "sleep". The rules here came from IFEval, so they keep its meaning.
        self.assertEqual(run_static_check(rule("not_contains", "sleep"), "Asleep and sleeping", "en"), "PASS")
        self.assertEqual(run_static_check(rule("not_contains", "heal"), "Healthy and healing", "en"), "PASS")

    def test_a_phrase_is_matched_as_a_phrase(self):
        r = rule("not_contains", "pilates studio")
        self.assertEqual(run_static_check(r, "Try the Pilates Studio downtown", "en"), "FAIL")
        self.assertEqual(run_static_check(r, "Try pilates at a studio", "en"), "PASS")

    def test_punctuation_has_no_word_boundary(self):
        r = rule("not_contains", ",")
        self.assertEqual(run_static_check(r, "No commas here", "en"), "PASS")
        self.assertEqual(run_static_check(r, "One, two", "en"), "FAIL")

    def test_commas_in_a_hindi_reply(self):
        r = rule("not_contains", ",")
        self.assertEqual(run_static_check(r, "यह एक वाक्य है। कोई चिह्न नहीं।", "hi"), "PASS")
        self.assertEqual(run_static_check(r, "पहला, दूसरा और तीसरा", "hi"), "FAIL")

    def test_the_hindi_full_stop_is_not_a_comma(self):
        # The danda ends a sentence. A rule against commas must not fail on it.
        self.assertEqual(run_static_check(rule("not_contains", ","), "आज मौसम अच्छा है।", "hi"), "PASS")

    def test_an_english_word_is_absent_from_a_devanagari_reply(self):
        self.assertEqual(run_static_check(rule("not_contains", "sleep"), "आपको आराम करना चाहिए।", "hi"), "PASS")

    def test_zero_width_characters_do_not_hide_a_word(self):
        # A zero-width space inside the word must not get a reply past the check.
        self.assertEqual(run_static_check(rule("not_contains", "sleep"), "You need sle​ep", "en"), "FAIL")


class StartsWithTests(unittest.TestCase):
    def test_exact_start_passes(self):
        r = rule("starts_with", "Let's talk about that.")
        self.assertEqual(run_static_check(r, "Let's talk about that. Here is more.", "en"), "PASS")

    def test_leading_blank_space_is_ignored(self):
        r = rule("starts_with", "Let's talk about that.")
        self.assertEqual(run_static_check(r, "\n  Let's talk about that. Yes.", "en"), "PASS")

    def test_the_phrase_later_in_the_reply_fails(self):
        r = rule("starts_with", "Let's talk about that.")
        self.assertEqual(run_static_check(r, "Sure! Let's talk about that.", "en"), "FAIL")

    def test_curly_quotes_count_as_straight_ones(self):
        self.assertEqual(run_static_check(rule("starts_with", '"'), "“wrapped”", "en"), "PASS")

    def test_a_hindi_reply_that_keeps_the_english_phrase_passes(self):
        r = rule("starts_with", "Let's talk about that.")
        self.assertEqual(run_static_check(r, "Let's talk about that. यह एक अच्छा सवाल है।", "hi"), "PASS")

    def test_a_translated_phrase_fails_here_and_goes_to_the_judge(self):
        # hybrid.py sends this FAIL to the judge, because a correct Hindi reply can fail a word check.
        r = rule("starts_with", "Let's talk about that.")
        self.assertEqual(run_static_check(r, "आइए इस पर बात करते हैं। ...", "hi"), "FAIL")


class BadSettingsTests(unittest.TestCase):
    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            run_static_check(rule("ends_with", "x"), "anything", "en")

    def test_missing_type_raises(self):
        r = Rule(id="R1", conv_id="S001", text="t", stated_turn=1, from_turn=2, method="static", check={})
        with self.assertRaises(ValueError):
            run_static_check(r, "anything", "en")

    def test_empty_text_raises(self):
        with self.assertRaises(ValueError):
            run_static_check(rule("not_contains", ""), "anything", "en")


class ReplyLanguageTests(unittest.TestCase):
    def test_plain_english(self):
        self.assertEqual(detect_reply_language("Here are three ideas for you."), "en")

    def test_plain_hindi(self):
        self.assertEqual(detect_reply_language("यह एक पूरा हिंदी उत्तर है।"), "hi")

    def test_mixed_reply(self):
        self.assertEqual(detect_reply_language("Sure — यह एक अच्छा सवाल है and here is more text."), "mixed")

    def test_romanised_hindi_is_not_called_english(self):
        # A reply in Hindi written in Latin letters is a language-leakage case, not plain English.
        self.assertEqual(detect_reply_language("aap kaise hain, mera naam test hai aur yeh theek hai"), "mixed")

    def test_one_romanised_word_is_still_english(self):
        self.assertEqual(detect_reply_language("The word hai appears once in this English sentence."), "en")

    def test_no_letters_is_unknown(self):
        self.assertEqual(detect_reply_language("12345 !!! ???"), "unknown")
        self.assertEqual(detect_reply_language(""), "unknown")

    def test_digits_and_punctuation_do_not_count(self):
        self.assertIsNone(devanagari_share("123 !!!"))
        self.assertEqual(devanagari_share("अ a"), 0.5)

    def test_romanised_word_list_finds_distinct_words(self):
        self.assertEqual(romanised_hindi_words("hai hai nahi kya"), {"hai", "nahi", "kya"})


if __name__ == "__main__":
    unittest.main()
