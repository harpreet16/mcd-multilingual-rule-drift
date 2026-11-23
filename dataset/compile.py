"""Merge the contributors' conversation files into one checked dataset.

Input:  several conversations.jsonl files (format in contracts/models.py), and optionally a rules file
Output: one conversations.jsonl with unique ids

What it does:
  1. reads each file and builds the conversations
  2. merges a rules file: a .jsonl with one Rule per line (the fields of contracts.models.Rule, with conv_id).
     conv_id may be S003 or the spreadsheet form LIS-S03. The conversations sheet has no rules, so they come from here
     Rules replace any rules already in the conversation, and a rule for an unknown conversation is a problem
  3. runs contracts.check.check_conversation on every conversation, and checks the ids:
       - the id looks like S001, and no id is used twice (even across files)
         (ids are not renumbered, because rules and results refer to them)
  4. warnings, which never block the file (worth fixing before the freeze):
       - contracts.check.warn_conversation (rules testable at too few turns), only for rules that run to the last
         turn. A rule whose window ends earlier is a one-shot check and is only counted
       - the Hindi checks in dataset/hindi_checks.py
       - a rule stated at a side question
       - two rules that tell the model to start (or end) every answer with different text in the same turns
       - EN_HI_EN has no Hindi turn between a rule's statement and the turns it is checked at, so the rule is
         never tested after a switch (a Hindi side question in between counts)
       - a user turn that seems to point at what the assistant said earlier ("as you mentioned").
         Replies are made live, so such a turn may not make sense. Rewrite it or mark it as a special case
  5. with --follow-ups, adds two follow-up turns to the conversations whose late rules cannot be tested (dataset/follow_ups.py)
  6. writes the file only when there are no problems

  python3 -m dataset.compile draft.jsonl --rules dataset/data/rules_grading.jsonl \
      --follow-ups -o dataset/data/conversations.jsonl
"""
import argparse
import re
import sys
from collections import defaultdict

from contracts.check import check_conversation, warn_conversation
from contracts.io import read_jsonl, write_jsonl
from contracts.models import Conversation, Rule
from dataset.hindi_checks import check_conversation_hindi
from dataset.import_xlsx import new_id
from dataset.follow_ups import add_follow_ups, needs_follow_ups, untested_rules

_ID = re.compile(r"S\d{3}")
# Phrases that point at something the assistant said earlier in the chat.
# Only claims about specific content count. "your previous response" works with a live reply, which exists.
_POINTS_AT_ASSISTANT = re.compile(
    r"\b(as you (mentioned|said|suggested|noted|wrote)|you (mentioned|said|suggested|wrote|included|gave|came up with)"
    r"|you put in|typo in your|your (earlier|previous|last) (instructions|message))\b", re.I)
# Check types that fix the start or the end of every answer.
_EDGE_CHECKS = ("starts_with", "ends_with")


def _window(rule, turn_count):
    return range(rule.from_turn, (rule.to_turn if rule.to_turn is not None else turn_count) + 1)


def _runs_to_end(conv):
    """Ids of the rules whose window runs to the last turn. A rule that ends earlier ended on purpose (a single request
    such as "exactly five suggestions", or one the user replaced), so it is a one-shot check, not a problem."""
    last = len(conv.turns_en)
    return {rule.id for rule in conv.rules if rule.to_turn is None or rule.to_turn >= last}


def warn_more(conv):
    """The warnings that are specific to this project's design. See the module docstring."""
    warnings = []
    count = len(conv.turns_en)
    side = {t.n for t in conv.turns_en if t.side_question}
    for rule in conv.rules:
        if rule.stated_turn in side:
            warnings.append(f"{conv.id} rule {rule.id}: stated at turn {rule.stated_turn}, which is a side question")
    for i, a in enumerate(conv.rules):
        for b in conv.rules[i + 1:]:
            kind = a.check.get("type")
            if kind in _EDGE_CHECKS and kind == b.check.get("type") and a.check.get("text") != b.check.get("text") \
                    and set(_window(a, count)) & set(_window(b, count)):
                warnings.append(f"{conv.id} rules {a.id} and {b.id}: both fix the {kind.replace('_', ' ')} text "
                                f"with different words at the same turns, so they conflict")
    for rule in conv.rules:
        # A word the rule checks (static check text, or a quoted phrase in the rule) must survive in the Hindi turn.
        tokens = {t for t in re.findall(r'"([^"]{2,80})"', rule.text) if t in conv.turns_en[rule.stated_turn - 1].text}
        word = rule.check.get("text", "")
        if word.isalpha():
            tokens.add(word)
        hindi = conv.turns_hi[rule.stated_turn - 1].text
        for token in sorted(tokens):
            if token.lower() not in hindi.lower():
                warnings.append(f"{conv.id} rule {rule.id}: the checked text {token!r} is not in the Hindi turn "
                                f"{rule.stated_turn}, so Hindi variants ask for something else")
    for rule_id in sorted(set(untested_rules(conv)) & _runs_to_end(conv)):
        warnings.append(f"{conv.id} rule {rule_id}: no Hindi turn between the rule and the turns it is checked at "
                        f"in EN_HI_EN, so it is never tested after a switch")
    for turn in conv.turns_en:
        if turn.n > 1 and _POINTS_AT_ASSISTANT.search(turn.text):
            warnings.append(f"{conv.id} turn {turn.n}: seems to point at an earlier assistant reply")
    return warnings


def merge_rules(conversations, rules_path):
    """Replace each conversation's rules with those in the rules file. Return the problems found."""
    problems = []
    by_conv = defaultdict(list)
    known = {c.id for c in conversations}
    for row in read_jsonl(rules_path):
        rule = Rule.from_dict(row)
        rule.conv_id = new_id(rule.conv_id)    # the rules file may use the spreadsheet ids (LIS-S03)
        if rule.conv_id not in known:
            problems.append(f"rules file: rule {rule.id} is for unknown conversation {rule.conv_id}")
        by_conv[rule.conv_id].append(rule)
    for conv in conversations:
        if conv.id in by_conv:
            conv.rules = by_conv[conv.id]
    return problems


def compile_dataset(input_files, output_path, rules_path=None, log=print, follow_ups=False):
    """Return the list of problems found. Write the output only when there are none."""
    problems, conversations, seen = [], [], {}
    for path in input_files:
        for row in read_jsonl(path):
            conv = Conversation.from_dict(row)
            if not _ID.fullmatch(conv.id):
                problems.append(f"{path}: conversation id {conv.id!r} is not in the form S001")
            if conv.id in seen:
                problems.append(f"{conv.id} is in both {seen[conv.id]} and {path}")
            seen[conv.id] = path
            conversations.append(conv)
    if rules_path:
        problems += merge_rules(conversations, rules_path)
    if follow_ups:
        extended = [c.id for c in conversations if needs_follow_ups(c)]
        for conv in conversations:
            if conv.id in extended:
                add_follow_ups(conv)
        log(f"added follow-up turns to {len(extended)} conversations: {' '.join(extended)}")
    warnings, one_shot = [], 0
    for conv in conversations:
        problems += check_conversation(conv)
        # warn_conversation words each message as "<conv> rule <id>: ..."
        warnings += [w for w in warn_conversation(conv) if w.split(" rule ")[1].split(":")[0] in _runs_to_end(conv)]
        one_shot += len(conv.rules) - len(_runs_to_end(conv))
        warnings += check_conversation_hindi(conv)
        warnings += warn_more(conv)
    for warning in warnings:
        log(f"warning: {warning}")
    log(f"{one_shot} rules are one-shot checks (their window ends before the last turn on purpose)")
    log(f"{len(conversations)} conversations, {len(problems)} problems, {len(warnings)} warnings")
    if not problems:
        conversations.sort(key=lambda c: c.id)
        write_jsonl(output_path, conversations)
        log(f"wrote {output_path}")
    return problems


def main():
    parser = argparse.ArgumentParser(prog="dataset.compile")
    parser.add_argument("inputs", nargs="+", help="conversations.jsonl files from the contributors")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("--rules", help="a .jsonl file with one Rule per line")
    parser.add_argument("--follow-ups", action="store_true",
                        help="add two follow-up turns where a rule is stated too late to be tested (dataset/follow_ups.py)")
    args = parser.parse_args()
    problems = compile_dataset(args.inputs, args.out, args.rules, follow_ups=args.follow_ups)
    for problem in problems:
        print("problem:", problem, file=sys.stderr)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
