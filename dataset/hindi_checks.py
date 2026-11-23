"""Automatic checks on the Hindi turns, to help the human Hindi reviewer (they do not replace one).

Each function takes the English turn and the Hindi turn of the same number and returns a list of
problems in plain words. An empty list means nothing was found.

What is checked:
  - the Hindi text is in NFC form, with no zero-width characters and no Devanagari digits
    (graders compare digits and words, so these would cause false fails)
  - the Hindi turn is written in Devanagari, and the English turn has none
  - every digit in the English turn is in the Hindi turn
  - every short quoted phrase that ends like a sentence ("Let's talk about that.") is kept in the Hindi turn
  - the Hindi turn is not just a copy of the English turn
"""
import re
import unicodedata
from collections import Counter

_ZERO_WIDTH = re.compile("[​‌‍⁠﻿]")
_DEVANAGARI = re.compile("[ऀ-ॿ]")
_DEVANAGARI_DIGIT = re.compile("[०-९]")
_QUOTED = re.compile(r'["“]([^"”]*)["”]')
_PHRASE_END = re.compile(r"[.!?]$")


def check_turn_pair(en, hi):
    problems = []
    if unicodedata.normalize("NFC", hi.text) != hi.text:
        problems.append("Hindi text is not in NFC form")
    if _ZERO_WIDTH.search(hi.text):
        problems.append("Hindi text has a zero-width character")
    if _DEVANAGARI_DIGIT.search(hi.text):
        problems.append("Hindi text has Devanagari digits (use 0-9)")
    if not _DEVANAGARI.search(hi.text):
        problems.append("Hindi text has no Devanagari letters")
    if _DEVANAGARI.search(en.text):
        problems.append("English text has Devanagari letters")
    # Every digit of the English text must be in the Hindi text. Extra digits are fine ("noon" may become "12").
    if Counter(re.findall(r"\d", en.text)) - Counter(re.findall(r"\d", hi.text)):
        problems.append("a digit of the English text is missing from the Hindi text")
    for quoted in _QUOTED.findall(en.text):
        # Only a short quoted phrase that ends like a sentence ("Let's talk about that.", "Disaster!") is a phrase to
        # repeat. Single quoted words ("classic") and long passages are naturally translated. compile also checks the
        # words each rule checks.
        if re.search("[A-Za-z]", quoted) and len(quoted.split()) <= 8 and _PHRASE_END.search(quoted) \
                and quoted not in hi.text:
            problems.append(f'quoted text "{quoted}" is not kept in the Hindi text')
    if en.text.strip() == hi.text.strip():
        problems.append("Hindi text is a copy of the English text")
    return problems


def check_conversation_hindi(conv):
    """Problems for every turn of one conversation, each starting with the conversation id and turn."""
    problems = []
    for en, hi in zip(conv.turns_en, conv.turns_hi):
        problems += [f"{conv.id} turn {en.n}: {p}" for p in check_turn_pair(en, hi)]
    return problems
