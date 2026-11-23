"""Turn the team's spreadsheet of conversations into a conversations.jsonl (without rules).

Input:  Lost_in_the_Switch_Merged_S01-S100.xlsx, one sheet "Conversations" with one row per
        conversation, version and turn (columns: Conversation ID, Version, Turn, Language, Turn text, Source, Source link)
Output: conversations.jsonl in the format of contracts/models.py. The "rules" list is empty: the spreadsheet holds
        no rules, so the authors add them (see dataset/README.md) and dataset/compile.py merges them in.

What it does:
  - reads the .xlsx with the standard library only (no openpyxl needed)
  - uses the "All English" and "All Hindi" versions. The old "English + Hindi" version is not used, because
    dataset/remix.py builds the four language types itself. It is only checked: its turns must equal the
    English or Hindi turn they claim to be
  - renames LIS-S01 to S001, the id format of the work division. The old id is kept in `source`
  - applies --hindi-fixes: the few Hindi turns whose draft translation changed a word the rules check (see the file)
  - marks a turn as a side question when it is one of the unrelated questions added for the experiment
    ("Quick side question: ...", "A small unrelated question: ...", "One more quick detour: ...").
    The other turns added for the experiment (the final "Please now write the full ..." turns) are real turns

  python3 -m dataset.import_xlsx dataset/source/Lost_in_the_Switch_Merged_S01-S100.xlsx \
      --hindi-fixes dataset/data/hindi_fixes.jsonl -o draft.jsonl
"""
import argparse
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict

from contracts.io import read_jsonl, write_jsonl

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_SIDE_QUESTION = re.compile(r"^(Quick side question|A small unrelated question|One more quick detour)\b", re.I)
_VERSIONS = {"All English": "en", "All Hindi": "hi"}


def read_sheet(path, sheet="sheet1"):
    """Return the rows of one sheet as a list of dicts keyed by the header names."""
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for item in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(_NS + "si"):
                shared.append("".join(t.text or "" for t in item.iter(_NS + "t")))
        rows = []
        for row in ET.fromstring(z.read(f"xl/worksheets/{sheet}.xml")).iter(_NS + "row"):
            cells = {}
            for cell in row.findall(_NS + "c"):
                column = re.match(r"[A-Z]+", cell.get("r")).group()
                value = cell.find(_NS + "v")
                if cell.get("t") == "s" and value is not None:
                    cells[column] = shared[int(value.text)]
                elif cell.get("t") == "inlineStr":
                    cells[column] = "".join(t.text or "" for t in cell.iter(_NS + "t"))
                else:
                    cells[column] = value.text if value is not None else ""
            rows.append(cells)
    header = rows[0]
    return [{header[col]: text for col, text in row.items() if col in header} for row in rows[1:]]


def new_id(old_id):
    """LIS-S01 becomes S001."""
    match = re.fullmatch(r"(?:LIS-)?S(\d+)", old_id.strip())
    if not match:
        raise ValueError(f"cannot turn conversation id {old_id!r} into the S001 format")
    return f"S{int(match.group(1)):03d}"


def build_conversations(rows, hindi_fixes=()):
    """Return (conversations as dicts, problems). Every problem is plain words.

    hindi_fixes is a list of {"conv_id", "turn", "text"}: a replacement for one Hindi turn (conv_id as S001).
    """
    problems = []
    grouped = defaultdict(lambda: defaultdict(list))
    sources = {}
    for row in rows:
        grouped[row["Conversation ID"]][row["Version"]].append(row)
        # The benchmark the conversation came from. Turns "Added for this experiment" are not the source.
        name = row["Source"].split("\n")[0]
        if not name.startswith("Added"):
            sources.setdefault(row["Conversation ID"], name)
    conversations = []
    for old_id in sorted(grouped, key=new_id):
        versions = grouped[old_id]
        for needed in _VERSIONS:
            if needed not in versions:
                problems.append(f"{old_id}: no {needed} version")
        if problems and any(p.startswith(old_id) for p in problems):
            continue
        turns = {}
        for version, lang in _VERSIONS.items():
            ordered = sorted(versions[version], key=lambda r: int(r["Turn"]))
            turns[lang] = [{"n": int(r["Turn"]), "lang": lang, "text": r["Turn text"],
                            "side_question": False} for r in ordered]
        # A turn is a side question when its English text is one of the added unrelated questions.
        for en, hi in zip(turns["en"], turns["hi"]):
            if _SIDE_QUESTION.match(en["text"]):
                en["side_question"] = hi["side_question"] = True
        # The old mixed version must agree with the English and Hindi turns it is built from.
        for row in versions.get("English + Hindi", []):
            side = turns["en"] if row["Language"] == "English" else turns["hi"]
            n = int(row["Turn"])
            if n > len(side) or side[n - 1]["text"] != row["Turn text"]:
                problems.append(f"{old_id}: turn {n} of the English + Hindi version does not match its source turn")
        for fix in hindi_fixes:
            if fix["conv_id"] == new_id(old_id):
                turns["hi"][fix["turn"] - 1]["text"] = fix["text"]
        conversations.append({
            "id": new_id(old_id),
            "source": f"{sources[old_id]} (spreadsheet id {old_id})",
            "turns_en": turns["en"], "turns_hi": turns["hi"], "rules": [],
        })
    return conversations, problems


def main():
    parser = argparse.ArgumentParser(prog="dataset.import_xlsx")
    parser.add_argument("xlsx")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("--hindi-fixes", help="a .jsonl file of replacement Hindi turns (conv_id, turn, text)")
    args = parser.parse_args()
    fixes = read_jsonl(args.hindi_fixes) if args.hindi_fixes else []
    conversations, problems = build_conversations(read_sheet(args.xlsx), fixes)
    for problem in problems:
        print("problem:", problem, file=sys.stderr)
    if problems:
        sys.exit(1)
    write_jsonl(args.out, conversations)
    sides = sum(t["side_question"] for c in conversations for t in c["turns_en"])
    print(f"wrote {len(conversations)} conversations ({sides} side-question turns) to {args.out}; rules are empty")


if __name__ == "__main__":
    main()
