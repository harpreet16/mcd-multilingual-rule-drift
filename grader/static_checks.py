"""Code checks for rules that have a clear answer.

Input:  a Rule, a reply, and the reply language ("en", "hi", "mixed" or "unknown")
Output: "PASS" or "FAIL"

The settings for a check are in rule.check: a "type" and whatever that type needs. The dataset
uses two types today. Add a function and an entry in CHECKS to add another, and test it in both
languages.

| type         | settings | passes when                                           |
|--------------|----------|-------------------------------------------------------|
| not_contains | text     | the reply does not contain that word or phrase        |
| starts_with  | text     | the reply begins with that text                       |

What the checks are careful about:

**Whole words, not fragments.** `not_contains "heal"` must not fail on "healthy" or "healing".
Where the forbidden text starts and ends with a letter or digit, it is matched on word boundaries,
which is also what IFEval's own forbidden-word check does. A forbidden text that is punctuation,
for example a comma, has no word boundary and is matched as plain text.

**Hindi.** Replies are compared after NFC normalisation, with zero-width joiners removed and curly
quotes folded to straight ones, so a Devanagari reply is not failed over invisible characters or
the quotation marks a model happens to choose. Devanagari has no upper and lower case, so folding
case changes nothing there and is safe. The Hindi full stop (danda, "।") is not a comma and is
never counted as one.

**This function does not decide what happens next.** It only answers the narrow question. Whether
a FAIL is final or goes to the judge is hybrid.py's decision, and on a Hindi or mixed reply a FAIL
is always sent to the judge, because a correct Hindi reply can fail an English word check.

An unknown check type raises ValueError. hybrid.py turns that into UNGRADED, never a FAIL.
"""
import re
import unicodedata

ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"), None)
CURLY_QUOTES = {ord("“"): '"', ord("”"): '"', ord("„"): '"',
                ord("‘"): "'", ord("’"): "'"}


def normalise(text):
    """NFC, no zero-width characters, straight quotes. Used on both the reply and the rule's text."""
    text = unicodedata.normalize("NFC", text or "")
    return text.translate(ZERO_WIDTH).translate(CURLY_QUOTES)


def _pattern(text):
    """A regex for this text, on word boundaries where the text has letters or digits at its edges."""
    body = re.escape(text)
    start = r"\b" if text[:1].isalnum() else ""
    end = r"\b" if text[-1:].isalnum() else ""
    return re.compile(start + body + end, re.IGNORECASE | re.UNICODE)


def not_contains(reply, text):
    """PASS when the reply does not hold this word or phrase."""
    if not text:
        raise ValueError("not_contains needs a non-empty text")
    return "FAIL" if _pattern(normalise(text)).search(normalise(reply)) else "PASS"


def starts_with(reply, text):
    """PASS when the reply begins with this text, ignoring leading blank space."""
    if not text:
        raise ValueError("starts_with needs a non-empty text")
    return "PASS" if normalise(reply).lstrip().lower().startswith(normalise(text).lower()) else "FAIL"


CHECKS = {"not_contains": not_contains, "starts_with": starts_with}


def run_static_check(rule, reply, reply_lang):
    """Return "PASS" or "FAIL". The settings for the check are in rule.check.

    reply_lang is accepted so a future check can behave differently per language. The two checks
    here do not need it: hybrid.py already decides what a FAIL on a Hindi reply means.
    """
    check = dict(rule.check or {})
    kind = check.pop("type", "")
    if not kind:
        raise ValueError(f"rule {rule.id}: a static rule needs check.type")
    if kind not in CHECKS:
        raise ValueError(f"rule {rule.id}: check type {kind!r} is not one of {sorted(CHECKS)}")
    return CHECKS[kind](reply, check.get("text", ""))
