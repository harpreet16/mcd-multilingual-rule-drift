"""Fixed words used across the project. Change them only through a contract change."""

# The four language types. Each one is a pattern of languages over all the turns of a conversation.
# A "rule turn" is any turn where a rule is stated (Rule.stated_turn).
#   EN        every turn English (baseline)
#   EN_HI_EN  rule turns and the last turn English, every other turn Hindi
#   HI_ALL    every turn Hindi (control)
#   HI_EN_HI  first and last turn Hindi, every turn in between English
# Example, 6 turns with rules stated at turns 1 and 5:
#   EN_HI_EN = en hi hi hi en en      HI_EN_HI = hi en en en en hi
# dataset/remix.py builds them.
CONDITIONS = ("EN", "EN_HI_EN", "HI_ALL", "HI_EN_HI")

# Language code for one turn.
TURN_LANGS = ("en", "hi")

# Language code for one reply, as detected by the grader.
REPLY_LANGS = ("en", "hi", "mixed", "unknown")

# How a rule is checked.
METHODS = ("static", "judge", "both")

# "english_only" means the rule is not graded on a Hindi reply.
LANGUAGE_SCOPES = ("any", "english_only")

# The result for one rule at one turn.
VERDICTS = ("PASS", "FAIL", "NOT_APPLICABLE", "UNGRADED")

# What decided a grade.
GRADE_METHODS = ("static", "judge", "static_then_judge", "none")

# What the judge may answer.
JUDGE_VERDICTS = ("YES", "NO", "NOT_APPLICABLE", "ERROR")

# What happened to one rule over a whole conversation.
RULE_LABELS = ("KEPT", "RECOVERED", "LOST", "UNKNOWN")

# What happened to a whole conversation.
CONVERSATION_LABELS = ("CLEAN", "RECOVERED", "LOST", "UNKNOWN")

# A rule should be testable at this many turns, so one lucky reply cannot hide forgetting.
MIN_TEST_TURNS = 2

# A rule with no FAIL is only called KEPT when at least this share of its turns was graded.
MIN_COVERAGE = 0.8

# Settings every response must record about how it was made.
SETTINGS_KEYS = ("backend", "model_id", "temperature", "max_tokens")
