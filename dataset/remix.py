"""Build the four language types from one conversation.

Input:  a Conversation
Output: one Variant per language type: EN, EN_HI_EN, HI_ALL, HI_EN_HI

For each turn, take either the English or the Hindi version of that same turn, never altered.
A "rule turn" is any turn where a rule is stated (Rule.stated_turn). The four patterns:

  EN        every turn English. The baseline.
  EN_HI_EN  rule turns and the last turn English, every other turn Hindi.
            The model always hears the rule in English, so a later slip cannot be blamed on not
            understanding the rule. It then has to keep the rule through the longest possible stretch
            of Hindi, and the conversation ends back in English.
  HI_ALL    every turn Hindi. The control.
  HI_EN_HI  first and last turn Hindi, every turn in between English. The reverse direction.
            Rule turns are not kept in English here, so a rule stated at turn 1 is in Hindi.

Example, 6 turns with rules stated at turns 1 and 5:
  EN        en en en en en en
  EN_HI_EN  en hi hi hi en en
  HI_ALL    hi hi hi hi hi hi
  HI_EN_HI  hi en en en en hi

Side questions follow the same pattern as any other turn.
Check the result with contracts.check.check_variant, which confirms every turn matches the conversation.

Command line (writes one variants.jsonl with every language type for every conversation):
  python3 -m dataset.remix dataset/data/conversations.jsonl -o variants.jsonl
"""
import argparse
import sys

from contracts.check import check_variant
from contracts.constants import CONDITIONS
from contracts.io import load_conversations, write_jsonl
from contracts.models import Turn, Variant


def pattern(conversation, condition):
    """Return the list of languages, one per turn, for this language type."""
    last = len(conversation.turns_en)
    if condition == "EN":
        return ["en"] * last
    if condition == "HI_ALL":
        return ["hi"] * last
    if condition == "EN_HI_EN":
        if not conversation.rules:
            raise ValueError(f"{conversation.id}: EN_HI_EN needs the rules, to know which turns state one")
        rule_turns = {rule.stated_turn for rule in conversation.rules}
        return ["en" if n in rule_turns or n == last else "hi" for n in range(1, last + 1)]
    if condition == "HI_EN_HI":
        return ["hi" if n in (1, last) else "en" for n in range(1, last + 1)]
    raise ValueError(f"condition {condition} is not one of {CONDITIONS}")


def remix(conversation, condition):
    """Return a Variant for this conversation and language type."""
    turns = []
    for lang, en, hi in zip(pattern(conversation, condition), conversation.turns_en, conversation.turns_hi):
        source = en if lang == "en" else hi
        turns.append(Turn(n=source.n, lang=source.lang, text=source.text, side_question=source.side_question))
    return Variant(conv_id=conversation.id, condition=condition, turns=turns)


def remix_all(conversations, conditions=CONDITIONS):
    """Return every variant of every conversation, checked against its conversation.

    Raises ValueError listing every problem, so a bad variant never reaches the runner.
    """
    variants, problems = [], []
    for conversation in conversations:
        for condition in conditions:
            variant = remix(conversation, condition)
            problems += check_variant(variant, conversation)
            variants.append(variant)
    if problems:
        raise ValueError("variants do not match their conversations:\n  " + "\n  ".join(problems))
    return variants


def main():
    parser = argparse.ArgumentParser(prog="dataset.remix")
    parser.add_argument("conversations", help="the compiled conversations.jsonl")
    parser.add_argument("-o", "--out", required=True, help="where to write variants.jsonl")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    args = parser.parse_args()
    try:
        variants = remix_all(load_conversations(args.conversations), args.conditions)
    except ValueError as error:
        print(error, file=sys.stderr)
        sys.exit(1)
    write_jsonl(args.out, variants)
    print(f"wrote {len(variants)} variants to {args.out}")


if __name__ == "__main__":
    main()
