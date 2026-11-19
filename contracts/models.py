"""The shared data shapes. Every module reads and writes these."""
from dataclasses import dataclass, field


class ServerUnreachable(Exception):
    """Raise this when a model server cannot be reached at all. The runner stops instead of failing every conversation."""


@dataclass
class Turn:
    n: int                        # turn number, starting at 1
    lang: str                     # "en" or "hi"
    text: str
    side_question: bool = False   # an unrelated question, never graded

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


@dataclass
class Rule:
    id: str
    conv_id: str
    text: str                     # the rule in plain words
    stated_turn: int              # the turn where the user gave the rule
    from_turn: int                # first turn where the rule is checked
    method: str                   # "static", "judge" or "both"
    to_turn: int | None = None    # last turn where it is checked, None means the last turn
    check: dict = field(default_factory=dict)  # settings for the static check
    question: str = ""            # yes/no question for the judge
    rubric: str = ""              # name of the house rule group for the judge, if any
    language_scope: str = "any"   # "english_only" skips Hindi replies

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


@dataclass
class Conversation:
    id: str
    source: str                   # where it came from, for example "written by a team member"
    turns_en: list[Turn]
    turns_hi: list[Turn]          # same turns, same order, in Hindi
    rules: list[Rule]

    @classmethod
    def from_dict(cls, d):
        return cls(
            id=d["id"],
            source=d.get("source", ""),
            turns_en=[Turn.from_dict(t) for t in d["turns_en"]],
            turns_hi=[Turn.from_dict(t) for t in d["turns_hi"]],
            rules=[Rule.from_dict(r) for r in d["rules"]],
        )


@dataclass
class Variant:
    """One conversation in one language type. This is what the runner sends to a model."""
    conv_id: str
    condition: str                # one of CONDITIONS
    turns: list[Turn]

    @classmethod
    def from_dict(cls, d):
        return cls(
            conv_id=d["conv_id"],
            condition=d["condition"],
            turns=[Turn.from_dict(t) for t in d["turns"]],
        )


@dataclass
class Response:
    """All the replies a model gave for one variant."""
    conv_id: str
    condition: str
    model: str
    replies: list[str]            # one reply per turn, same order as the variant
    cut_off: list[bool]           # one flag per reply
    settings: dict                # must hold the keys in SETTINGS_KEYS
    finished_at: str = ""

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


@dataclass
class Grade:
    """The result for one rule at one turn of one response."""
    conv_id: str
    condition: str
    model: str
    rule_id: str
    turn: int
    verdict: str                  # one of VERDICTS
    method: str                   # one of GRADE_METHODS
    reply_lang: str               # one of REPLY_LANGS
    reason: str = ""
    static_result: str = ""       # "PASS", "FAIL" or "" when no static check ran
    judge_result: str = ""        # one of JUDGE_VERDICTS, or "" when no judge call
    judge_model: str = ""
    rubric_version: str = ""
    escalated: bool = False       # True when a static FAIL was sent to the judge

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
