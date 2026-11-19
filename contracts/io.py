"""Read and write the project files. All files are JSON lines: one JSON object per line."""
import json
from dataclasses import asdict, is_dataclass

from .models import Conversation, Grade, Response, Variant


def _plain(row):
    # Accept either a dataclass or a plain dict.
    return asdict(row) if is_dataclass(row) else row


def read_jsonl(path):
    """Return the rows of a file as a list of dicts. Blank lines are skipped."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    """Write all rows to a file, replacing what was there."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_plain(row), ensure_ascii=False) + "\n")


def append_jsonl(path, row):
    """Add one row to the end of a file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_plain(row), ensure_ascii=False) + "\n")


def load_conversations(path):
    return [Conversation.from_dict(d) for d in read_jsonl(path)]


def load_variants(path):
    return [Variant.from_dict(d) for d in read_jsonl(path)]


def load_responses(path):
    return [Response.from_dict(d) for d in read_jsonl(path)]


def load_grades(path):
    return [Grade.from_dict(d) for d in read_jsonl(path)]
