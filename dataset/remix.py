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
"""


def remix(conversation, condition):
    """Return a Variant for this conversation and language type."""
    raise NotImplementedError("Not built yet")
