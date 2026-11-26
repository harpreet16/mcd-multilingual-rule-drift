"""Work out what language a model reply is written in.

Input:  the reply text
Output: one of REPLY_LANGS — "en", "hi", "mixed" or "unknown"

Every Grade records this, and hybrid.py uses it twice:
  - an english_only rule is NOT_APPLICABLE on a Hindi or mixed reply
  - a static FAIL on a Hindi or mixed reply goes to the judge instead of being final

How it decides. Letters are counted by script: Devanagari against Latin. Digits, punctuation and
spaces are ignored, so "5 ways to do X" and "X करने के 5 तरीके" are judged on their letters alone.

  share of Devanagari letters >= HI_FROM   -> "hi"
  share <= EN_UPTO                         -> "en"   (unless it reads as romanised Hindi, see below)
  anything between                         -> "mixed"
  no letters at all                        -> "unknown"

The English band is deliberately narrow (0.10). A reply with even a little Devanagari is called
"mixed", which sends a static FAIL to the judge rather than failing it outright. That is the cheap
direction to be wrong in: in the earlier audit 16 of 18 static disagreements on Hindi replies were
false fails, and there are only 19 static rules in the whole dataset, so extra judge calls are few.

Romanised Hindi ("aap kaise hain") has no Devanagari at all and would otherwise count as English.
A reply with ROMANISED_MIN or more distinct common Hindi function words written in Latin letters is
called "mixed" instead. This matters because a reply that answered in Hindi is a language-leakage
case, not a case of the model forgetting the rule, and the analysis splits the two.

Known limits: a long English reply quoting one Hindi phrase is "mixed"; a Hindi reply written
entirely in Latin letters with no function word from the list below is "en"; other scripts
(Urdu, Bengali) count as neither and are ignored.
"""
import re
import unicodedata

HI_FROM = 0.60          # at or above this share of Devanagari letters, the reply is Hindi
EN_UPTO = 0.10          # at or below this share, the reply is English
ROMANISED_MIN = 3       # distinct romanised Hindi function words that make a reply "mixed"

# Common Hindi function words as people type them in Latin letters. Function words only: they appear
# in almost any Hindi sentence and are rare as whole English words.
ROMANISED_HINDI = {
    "hai", "hain", "nahi", "nahin", "kya", "aap", "aapka", "aapke", "aapki", "mera", "meri", "mere",
    "karein", "kare", "karna", "karta", "karte", "karti", "kiya", "kijiye", "liye", "lekin", "aur",
    "yeh", "yah", "woh", "vah", "hoga", "hogi", "hona", "tha", "thi", "the", "bhi", "kuch", "kuchh",
    "sakta", "sakte", "sakti", "chahiye", "jaise", "abhi", "phir", "bahut", "thoda", "sabhi",
}

_WORD = re.compile(r"[a-z]+")


def devanagari_share(text):
    """The share of letters that are Devanagari, counting only Devanagari and Latin letters.

    Returns None when the text holds no letters of either script.
    """
    devanagari = latin = 0
    for ch in unicodedata.normalize("NFC", text or ""):
        if not ch.isalpha():
            continue
        if "ऀ" <= ch <= "ॿ":
            devanagari += 1
        elif "a" <= ch.lower() <= "z":
            latin += 1
    total = devanagari + latin
    if total == 0:
        return None
    return devanagari / total


def romanised_hindi_words(text):
    """The distinct romanised Hindi function words found in the text."""
    return {w for w in _WORD.findall((text or "").lower()) if w in ROMANISED_HINDI}


def detect_reply_language(reply):
    """Return "en", "hi", "mixed" or "unknown" for one model reply."""
    share = devanagari_share(reply)
    if share is None:
        return "unknown"
    if share >= HI_FROM:
        return "hi"
    if share <= EN_UPTO:
        if len(romanised_hindi_words(reply)) >= ROMANISED_MIN:
            return "mixed"
        return "en"
    return "mixed"
