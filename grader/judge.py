"""The LLM judge: Claude Haiku on Bedrock, called through the client in models/bedrock.

Input:  a Response, its Variant, a list of asks (rule, turn) pairs, and a rubric version
Output: one Grade for every ask, with judge_result (YES, NO, NOT_APPLICABLE or ERROR), a short reason,
        judge_model and rubric_version filled in. hybrid.py fills in the rest

    judge_response(response, variant, asks, rubric_version) -> list[Grade]

**What the judge is shown.** The rule, the yes/no question, and the conversation so far: the user's
turns and the model's own replies, in order, with turn numbers. It is never shown which model wrote
the replies or which language type the conversation is, so it cannot favour one arm of the study.
It answers one turn at a time and never gives a verdict for the conversation as a whole: those labels
are worked out in analysis/metrics.py from the per-turn verdicts.

**One call per rule, not per turn.** All the turns asked for one rule go in a single call, with the
transcript sent once. That is about four times fewer calls than asking turn by turn, which matters:
judging two models is roughly 7,000 rule-turns. The cost is that when the judge rules on turn 4 it
can also see turns 5 onwards, so a later turn could colour an earlier verdict. Pass `asks_per_call=1`
to send each turn on its own with the transcript cut at that turn, which removes that risk, and
compare the two on a sample — PLAN.md asks for exactly that comparison before the full run.

**Errors are never failures.** If Bedrock throttles past its retries, returns an API error, or the
answer cannot be read after two re-asks, every affected turn comes back ERROR. hybrid.py turns ERROR
into UNGRADED, which the analysis never counts as a fail. The single exception is ServerUnreachable,
which is raised so a run stops loudly instead of quietly grading nothing. That split is the contract
written in models/bedrock/README.md.

**Caching.** Answers are keyed by judge model, rubric version, the rule's question and text, the exact
transcript and the turns asked, so re-running a stopped job does not pay for the same verdict twice.
The cache is a JSONL file, one row per answered call.

Bulk use: `python3 -m grader --responses ... --out ...` already walks a whole file of responses and
calls this for each one.
"""
import hashlib
import json
import os
import re

from contracts.models import Grade, ServerUnreachable
from models.settings import JUDGE_MODEL_ID

# Default to the committed cache under runs/. It used to be "judge_cache.jsonl", which is
# relative to the working directory: a run started without MCD_JUDGE_CACHE set quietly wrote
# a SECOND cache at the repo root and shared nothing with the committed one. That happened to
# the qwen3_32b grading run, which paid for 813 judge calls that were already answered.
CACHE_PATH = os.environ.get("MCD_JUDGE_CACHE", os.path.join("runs", "judge_cache.jsonl"))
PARSE_ATTEMPTS = 3          # the first ask plus two re-asks, then ERROR
JUDGE_MAX_TOKENS = 1024

HOUSE_RULES = """You are grading whether a model followed a rule a user gave it earlier in a conversation.

You will be given:
  - the rule, in plain words
  - a yes/no question to answer about particular turns
  - the conversation so far: the user's turns and the model's replies, numbered

Answer the question for each turn listed, and only those turns.

Answer YES when the reply at that turn follows the rule.
Answer NO when the reply at that turn breaks the rule.
Answer NOT_APPLICABLE only when the rule cannot sensibly apply to that reply at all, for example when
the rule is about a section the user had not yet asked for. Do not use it because you are unsure.

How to judge:
  - Judge only the reply at the turn you are asked about. Earlier turns are context.
  - A reply written in Hindi can follow the rule perfectly well. Do not mark a reply down for being in
    Hindi, or for translating a phrase, unless the rule itself is about exact wording.
  - Judge what the reply does, not how good it is. Quality, accuracy and tone are not your concern
    unless the rule is about them.
  - Do not give an overall verdict for the conversation.

Reply with a JSON array and nothing else, one object per turn you were asked about:

[{"turn": 4, "verdict": "YES", "reason": "starts with the required phrase"},
 {"turn": 6, "verdict": "NO", "reason": "answer begins with a greeting instead"}]

Keep each reason under 15 words."""


class Cache:
    """Answers already paid for, keyed by a hash of everything that could change the answer."""

    def __init__(self, path=CACHE_PATH):
        self.path = path
        self.rows = {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self.rows[row["key"]] = row["answer"]

    def get(self, key):
        return self.rows.get(key)

    def put(self, key, answer):
        self.rows[key] = answer
        if self.path:
            folder = os.path.dirname(self.path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "answer": answer}, ensure_ascii=False) + "\n")


def transcript(variant, response, up_to):
    """The conversation up to and including that turn, as plain text. No model name, no condition."""
    lines = []
    for i, turn in enumerate(variant.turns):
        if turn.n > up_to:
            break
        lines.append(f"[turn {turn.n}] user: {turn.text}")
        if i < len(response.replies):
            lines.append(f"[turn {turn.n}] model: {response.replies[i]}")
    return "\n\n".join(lines)


def build_prompt(rule, conversation_text, turns):
    """The user message for one call: the house rules, the rule, the question, the conversation.

    The house rules are part of the message rather than a system message set when the chat function is
    built, so that any chat function gets them — including one written for the batch-versus-turn-by-turn
    comparison. One place to change them, no way to send a call without them.
    """
    asked = ", ".join(str(t) for t in turns)
    return (f"{HOUSE_RULES}\n\n---\n\n"
            f"RULE: {rule.text}\n\n"
            f"QUESTION: {rule.question or rule.text}\n\n"
            f"CONVERSATION SO FAR:\n{conversation_text}\n\n"
            f"Answer the question for {'turn' if len(turns) == 1 else 'turns'} {asked}. "
            f"Reply with the JSON array only.")


def parse_verdicts(text, turns):
    """Return {turn: (verdict, reason)} for exactly the turns asked.

    Raises ValueError when the answer cannot be read, is missing a turn, answers one twice, or uses a
    verdict that is not allowed. The caller re-asks, then gives up with ERROR — never with NO.
    """
    match = re.search(r"\[.*\]", text or "", re.S)
    if not match:
        raise ValueError("no JSON array in the answer")
    try:
        rows = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        raise ValueError(f"the JSON array could not be read: {error}") from error
    if not isinstance(rows, list):
        raise ValueError("the answer is not a list")

    out = {}
    for row in rows:
        if not isinstance(row, dict) or "turn" not in row or "verdict" not in row:
            raise ValueError("a row has no turn or no verdict")
        try:
            turn = int(row["turn"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"turn {row['turn']!r} is not a number") from error
        verdict = str(row["verdict"]).strip().upper()
        if verdict not in ("YES", "NO", "NOT_APPLICABLE"):
            raise ValueError(f"verdict {verdict!r} is not YES, NO or NOT_APPLICABLE")
        if turn in out:
            raise ValueError(f"turn {turn} was answered twice")
        out[turn] = (verdict, str(row.get("reason", ""))[:200])

    missing = [t for t in turns if t not in out]
    if missing:
        raise ValueError(f"no answer for turn(s) {missing}")
    return {t: out[t] for t in turns}


def _grade(response, rule, turn, result, reason, rubric_version, judge_model):
    return Grade(conv_id=response.conv_id, condition=response.condition, model=response.model,
                 rule_id=rule.id, turn=turn, verdict="", method="judge", reply_lang="unknown",
                 judge_result=result, judge_model=judge_model, rubric_version=rubric_version,
                 reason=reason)


def _ask_once(chat, rule, conversation_text, turns):
    """One Bedrock call, re-asked while the answer cannot be read. Returns {turn: (verdict, reason)}."""
    messages = [{"role": "user", "content": build_prompt(rule, conversation_text, turns)}]
    last = None
    for attempt in range(PARSE_ATTEMPTS):
        text, _ = chat(messages)
        try:
            return parse_verdicts(text, turns)
        except ValueError as error:
            last = error
            messages = [messages[0],
                        {"role": "assistant", "content": text or ""},
                        {"role": "user", "content": f"That could not be read ({error}). "
                                                    f"Reply with the JSON array only, one object per "
                                                    f"turn in {list(turns)}."}]
    raise ValueError(f"unreadable after {PARSE_ATTEMPTS} attempts: {last}")


def make_judge_chat():
    """The Bedrock chat function for the judge model. Imported late so tests need no boto3."""
    from models.bedrock.client import make_chat
    return make_chat(JUDGE_MODEL_ID, max_tokens=JUDGE_MAX_TOKENS, max_tokens_cap=JUDGE_MAX_TOKENS * 2,
                     temperature=0)


def judge_response(response, variant, asks, rubric_version="", chat=None, cache=None, asks_per_call=None):
    """Return one Grade for every (rule, turn) pair in asks.

    chat           a chat(messages) -> (text, cut_off) function. Defaults to Claude Haiku on Bedrock
    cache          a Cache. Defaults to the file at MCD_JUDGE_CACHE or runs/judge_cache.jsonl
    asks_per_call  turns per call for one rule. None groups them all into one call; 1 sends each turn
                   on its own with the transcript cut at that turn
    """
    if not asks:
        return []
    chat = chat or make_judge_chat()
    cache = cache if cache is not None else Cache()
    judge_model = JUDGE_MODEL_ID

    by_rule = {}
    for rule, turn in asks:
        by_rule.setdefault(rule.id, (rule, []))[1].append(turn)

    grades = []
    for rule, turns in by_rule.values():
        turns = sorted(set(turns))
        groups = [turns] if asks_per_call is None else [turns[i:i + asks_per_call]
                                                        for i in range(0, len(turns), asks_per_call)]
        for group in groups:
            conversation_text = transcript(variant, response, max(group))
            key = hashlib.sha256("\u0000".join(
                [judge_model, rubric_version, rule.question or "", rule.text or "",
                 conversation_text, ",".join(map(str, group))]).encode("utf-8")).hexdigest()

            answer = cache.get(key)
            if answer is None:
                try:
                    verdicts = _ask_once(chat, rule, conversation_text, group)
                except ServerUnreachable:
                    raise                          # a dead endpoint stops the run, loudly
                except Exception as error:         # noqa: BLE001 - everything else is an ERROR grade
                    grades += [_grade(response, rule, t, "ERROR", f"judge call failed: {error}"[:200],
                                      rubric_version, judge_model) for t in group]
                    continue
                answer = {str(t): list(v) for t, v in verdicts.items()}
                cache.put(key, answer)

            for turn in group:
                verdict, reason = answer.get(str(turn), ("ERROR", "missing from the judge's answer"))
                grades.append(_grade(response, rule, turn, verdict, reason, rubric_version, judge_model))
    return grades
