"""Check a file's contents against the contract.

Every function returns a list of problems in plain words.
An empty list means everything is fine.
"""
from .constants import (CONDITIONS, GRADE_METHODS, JUDGE_VERDICTS, LANGUAGE_SCOPES,
                        METHODS, MIN_TEST_TURNS, REPLY_LANGS,
                        SETTINGS_KEYS, TURN_LANGS, VERDICTS)


def _check_turn_numbers(label, turns):
    # Turns must be numbered 1, 2, 3 and so on.
    problems = []
    for position, turn in enumerate(turns, start=1):
        if turn.n != position:
            problems.append(f"{label}: turn at position {position} is numbered {turn.n}")
    return problems


def _check_rule(rule, conv_id, turn_count):
    problems = []
    name = f"{conv_id} rule {rule.id}"
    if rule.conv_id != conv_id:
        problems.append(f"{name}: conv_id is {rule.conv_id}")
    if rule.method not in METHODS:
        problems.append(f"{name}: method {rule.method} is not one of {METHODS}")
    if rule.method in ("static", "both") and "type" not in rule.check:
        problems.append(f"{name}: a static rule needs check with a type")
    if rule.method in ("judge", "both") and not rule.question:
        problems.append(f"{name}: a judge rule needs a question")
    if rule.language_scope not in LANGUAGE_SCOPES:
        problems.append(f"{name}: language_scope {rule.language_scope} is not allowed")
    if not 1 <= rule.stated_turn <= turn_count:
        problems.append(f"{name}: stated_turn {rule.stated_turn} is outside the conversation")
    if rule.from_turn < rule.stated_turn:
        problems.append(f"{name}: from_turn is before stated_turn")
    if rule.from_turn > turn_count:
        problems.append(f"{name}: from_turn is after the last turn")
    if rule.to_turn is not None:
        if rule.to_turn < rule.from_turn:
            problems.append(f"{name}: to_turn is before from_turn")
        if rule.to_turn > turn_count:
            problems.append(f"{name}: to_turn is after the last turn")
    return problems


def check_conversation(conv):
    problems = []
    if not conv.id:
        problems.append("a conversation has no id")
    count = len(conv.turns_en)
    if count == 0:
        problems.append(f"{conv.id}: no English turns")
    problems += _check_turn_numbers(f"{conv.id} English", conv.turns_en)
    problems += _check_turn_numbers(f"{conv.id} Hindi", conv.turns_hi)
    if len(conv.turns_hi) != count:
        problems.append(f"{conv.id}: {count} English turns but {len(conv.turns_hi)} Hindi turns")
    else:
        for en, hi in zip(conv.turns_en, conv.turns_hi):
            if en.side_question != hi.side_question:
                problems.append(f"{conv.id}: turn {en.n} is a side question in only one language")
    for turn in conv.turns_en:
        if turn.lang != "en":
            problems.append(f"{conv.id}: English turn {turn.n} has lang {turn.lang}")
    for turn in conv.turns_hi:
        if turn.lang != "hi":
            problems.append(f"{conv.id}: Hindi turn {turn.n} has lang {turn.lang}")
    if not conv.rules:
        problems.append(f"{conv.id}: no rules")
    seen = set()
    for rule in conv.rules:
        if rule.id in seen:
            problems.append(f"{conv.id}: rule id {rule.id} is used twice")
        seen.add(rule.id)
        problems += _check_rule(rule, conv.id, count)
    return problems


def check_variant(variant, conversation=None):
    problems = []
    name = f"{variant.conv_id} {variant.condition}"
    if variant.condition not in CONDITIONS:
        problems.append(f"{name}: condition is not one of {CONDITIONS}")
    problems += _check_turn_numbers(name, variant.turns)
    for turn in variant.turns:
        if turn.lang not in TURN_LANGS:
            problems.append(f"{name}: turn {turn.n} has lang {turn.lang}")
    if conversation is not None:
        if len(variant.turns) != len(conversation.turns_en):
            problems.append(f"{name}: {len(variant.turns)} turns but the conversation has {len(conversation.turns_en)}")
        else:
            problems += _check_variant_text(name, variant, conversation)
    return problems


def _check_variant_text(name, variant, conversation):
    # Each turn must be the English or the Hindi text of the same turn, never altered.
    problems = []
    for turn, en, hi in zip(variant.turns, conversation.turns_en, conversation.turns_hi):
        source = en if turn.lang == "en" else hi
        if turn.text != source.text:
            problems.append(f"{name}: turn {turn.n} text does not match the conversation's {turn.lang} text")
        if turn.side_question != source.side_question:
            problems.append(f"{name}: turn {turn.n} side_question does not match the conversation")
    return problems


def check_response(response, variant=None):
    problems = []
    name = f"{response.conv_id} {response.condition} {response.model}"
    if len(response.cut_off) != len(response.replies):
        problems.append(f"{name}: {len(response.replies)} replies but {len(response.cut_off)} cut_off flags")
    for key in SETTINGS_KEYS:
        if key not in response.settings:
            problems.append(f"{name}: settings is missing {key}")
    if variant is not None and len(response.replies) != len(variant.turns):
        problems.append(f"{name}: {len(response.replies)} replies but the variant has {len(variant.turns)} turns")
    return problems


def check_grade(grade):
    problems = []
    name = f"{grade.conv_id} {grade.condition} {grade.model} rule {grade.rule_id} turn {grade.turn}"
    if grade.verdict not in VERDICTS:
        problems.append(f"{name}: verdict {grade.verdict} is not one of {VERDICTS}")
    if grade.method not in GRADE_METHODS:
        problems.append(f"{name}: method {grade.method} is not one of {GRADE_METHODS}")
    if grade.reply_lang not in REPLY_LANGS:
        problems.append(f"{name}: reply_lang {grade.reply_lang} is not one of {REPLY_LANGS}")
    if grade.static_result not in ("", "PASS", "FAIL"):
        problems.append(f"{name}: static_result {grade.static_result} is not PASS, FAIL or empty")
    if grade.judge_result and grade.judge_result not in JUDGE_VERDICTS:
        problems.append(f"{name}: judge_result {grade.judge_result} is not one of {JUDGE_VERDICTS}")
    return problems


def check_judge_grades(asks, grades):
    """asks is a list of (rule, turn) pairs. The judge must return one Grade for each, once, with a judge_result."""
    problems = []
    wanted = {(rule.id, turn) for rule, turn in asks}
    got = set()
    for grade in grades:
        key = (grade.rule_id, grade.turn)
        if key in got:
            problems.append(f"rule {key[0]} turn {key[1]} was answered twice")
        got.add(key)
        if grade.judge_result not in JUDGE_VERDICTS:
            problems.append(f"rule {key[0]} turn {key[1]}: judge_result {grade.judge_result} is not one of {JUDGE_VERDICTS}")
        if key not in wanted:
            problems.append(f"rule {key[0]} turn {key[1]} was not asked for")
    for rule_id, turn in sorted(wanted - got):
        problems.append(f"rule {rule_id} turn {turn} has no answer")
    return problems


def warn_conversation(conv):
    """Softer advice than check_conversation. These are worth fixing but do not block the file."""
    warnings = []
    last = len(conv.turns_en)
    for rule in conv.rules:
        end = rule.to_turn if rule.to_turn is not None else last
        testable = [t for t in conv.turns_en if rule.from_turn <= t.n <= end and not t.side_question]
        if len(testable) < MIN_TEST_TURNS:
            warnings.append(f"{conv.id} rule {rule.id}: only {len(testable)} testable turn(s), "
                            f"so a later slip would not show (aim for {MIN_TEST_TURNS} or more)")
    return warnings


# The numbers the results file must hold for every model and condition.
# slip_rate = (recovered + lost) / (clean + recovered + lost) conversations
# loss_rate = lost / (clean + recovered + lost) conversations
RESULT_KEYS = ("conversations", "clean_conversations", "recovered_conversations",
               "lost_conversations", "unknown_conversations", "slip_rate", "loss_rate",
               "rules_kept", "rules_recovered", "rules_lost",
               "rule_turns_graded", "rule_turns_ungraded", "pass_rate")


def check_results(results):
    """results is a dict like {"models": {model: {condition: {numbers}}}}."""
    problems = []
    if "models" not in results:
        return ["results has no models key"]
    for model, by_condition in results["models"].items():
        for condition, numbers in by_condition.items():
            if condition not in CONDITIONS:
                problems.append(f"{model}: condition {condition} is not one of {CONDITIONS}")
            for key in RESULT_KEYS:
                if key not in numbers:
                    problems.append(f"{model} {condition}: missing {key}")
    return problems
