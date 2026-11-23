"""Fix rules that are stated too late to be tested after a language switch.

The problem: a rule stated at (or near) the last turn has no later turn where it can slip, and in EN_HI_EN the last
turns are English, so it is never tested after a switch. The Multi-IF, IFEval, AdvancedIF and M2Lingual
conversations end with a new request at the last turn, so this affects most of them. Editing a rule cannot fix it,
because the turns do not exist.

The fix: add two follow-up user turns after the last turn. Each asks for another version of the last answer and says
the earlier requirements still apply, so the rules stay in force:

  turn n+1   "Thanks. Please give me a second version of that answer, in different words, and keep following
              everything I asked for earlier."
  turn n+2   "Great. One more time: please give me a third version of that answer, again following everything
              I asked for earlier."

In EN_HI_EN the last turn is English and the turn before it is Hindi (it is neither a rule turn nor the last turn),
so a rule stated at turn n is now checked at a Hindi turn and then at an English one, which is a switch and a switch
back. Rules whose window ran to the old last turn (to_turn None, or equal to n) now run to the new last turn
(to_turn None). A rule the user cancelled earlier (to_turn below n) is left alone.

Only conversations that need it get the turns (see needs_follow_ups). The Hindi text is a draft for the Hindi reviewer.

  python3 -m dataset.compile draft.jsonl --rules dataset/data/rules_grading.jsonl \
      --follow-ups -o dataset/data/conversations.jsonl
"""
from contracts.check import warn_conversation
from contracts.models import Turn
from dataset.remix import pattern

FOLLOW_UPS = (
    ("Thanks. Please give me a second version of that answer, in different words, and keep following "
     "everything I asked for earlier.",
     "धन्यवाद। कृपया उसी उत्तर का दूसरा संस्करण अलग शब्दों में दीजिए, और मैंने पहले जो कुछ कहा था उसका पालन करते रहिए।"),
    ("Great. One more time: please give me a third version of that answer, again following everything "
     "I asked for earlier.",
     "बढ़िया। एक बार और: कृपया उसी उत्तर का तीसरा संस्करण दीजिए, और फिर से मेरी सभी पिछली बातों का पालन कीजिए।"),
)


def _window(rule, turn_count):
    return range(rule.from_turn, (rule.to_turn if rule.to_turn is not None else turn_count) + 1)


def untested_rules(conv):
    """Rules never checked after a language switch in EN_HI_EN. Returns a list of rule ids."""
    if not conv.rules:
        return []
    count = len(conv.turns_en)
    side = {t.n for t in conv.turns_en if t.side_question}
    try:
        langs = pattern(conv, "EN_HI_EN")
    except ValueError:
        return []
    found = []
    for rule in conv.rules:
        # A graded turn comes after (or is) a Hindi turn that follows the rule's statement. A Hindi side question
        # in between counts: the model has to come back from it.
        window = [n for n in _window(rule, count) if n not in side and 1 <= n <= count]
        if window and not any(langs[u - 1] == "hi" for n in window for u in range(rule.stated_turn + 1, n + 1)):
            found.append(rule.id)
    return found


def needs_follow_ups(conv):
    """True when a rule that runs to the last turn is never tested after a switch, or at fewer than 2 turns."""
    count = len(conv.turns_en)
    late = set(untested_rules(conv))
    # warn_conversation words its message as "<conv> rule <id>: only ..."
    for message in warn_conversation(conv):
        late.add(message.split(" rule ")[1].split(":")[0])
    return any(rule.id in late and (rule.to_turn is None or rule.to_turn >= count) for rule in conv.rules)


def add_follow_ups(conv):
    """Append the follow-up turns to this conversation and let the rules that ran to the old last turn run on."""
    old_last = len(conv.turns_en)
    for offset, (en, hi) in enumerate(FOLLOW_UPS, start=1):
        conv.turns_en.append(Turn(n=old_last + offset, lang="en", text=en))
        conv.turns_hi.append(Turn(n=old_last + offset, lang="hi", text=hi))
    for rule in conv.rules:
        if rule.to_turn is not None and rule.to_turn >= old_last:
            rule.to_turn = None
    return conv
