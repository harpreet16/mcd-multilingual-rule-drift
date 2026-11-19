"""Code checks for rules that have a clear answer.

Input:  a Rule, a reply, and the reply language ("en", "hi", "mixed" or "unknown")
Output: "PASS" or "FAIL"

Write one small function per check type. The examples use starts_with and not_contains. List the types
and their settings here as you add them. Be Hindi aware: accept Devanagari spellings and the Hindi full stop.
Add tests for every type, in English and in Hindi.
"""


def run_static_check(rule, reply, reply_lang):
    """Return "PASS" or "FAIL". The settings for the check are in rule.check."""
    raise NotImplementedError("Not built yet")
