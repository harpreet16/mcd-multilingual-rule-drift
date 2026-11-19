"""Merge the contributors' conversation files into one checked dataset.

Input:  several conversations.jsonl files (format in contracts/models.py)
Output: one conversations.jsonl with unique ids

Build: read each file, run contracts.check.check_conversation and warn_conversation on every conversation,
report every problem and warning, renumber the ids, and write the file only when there are no problems.
The rules for writing a conversation are in PLAN.md.
"""


def compile_dataset(input_files, output_path):
    """Return the list of problems found. Write the output only when there are none."""
    raise NotImplementedError("Not built yet")
