# dataset

Builds the dataset of 100 conversations and the four language types the runner plays.

```
source/*.xlsx --import_xlsx--> draft.jsonl --compile (+ rules_grading.jsonl, hindi_fixes.jsonl)--> data/conversations.jsonl --remix--> data/variants.jsonl
```

| File | What it does |
|---|---|
| `source/Lost_in_the_Switch_Merged_S01-S100.xlsx` | The conversations: English turns, Hindi turns, source of each turn |
| `data/rules_grading.jsonl` | The rules, one `Rule` per line (233 rules, every conversation has at least one) |
| `data/hindi_fixes.jsonl` | Two Hindi turns (S035, S055) whose translation changed a word the rules check |
| `data/conversations.jsonl` | The compiled dataset: 100 conversations with their rules |
| `data/variants.jsonl` | 400 variants, the input of the runner |
| `import_xlsx.py` | Spreadsheet to a draft conversations file (ids `LIS-S01` become `S001`, side questions marked, no rules) |
| `compile.py` | Merges the rules, runs the contract checks, prints warnings, writes the dataset only when there are no problems |
| `follow_ups.py` | Adds two follow-up turns to conversations whose rules are stated too late to be tested after a switch |
| `hindi_checks.py` | Automatic checks on the Hindi turns (NFC, zero-width, digits, kept quotes). Helps the reviewer, does not replace one |
| `remix.py` | Builds `EN`, `EN_HI_EN`, `HI_ALL`, `HI_EN_HI` for each conversation |

## Rebuild (from the repo root)

```
python3 -m dataset.import_xlsx dataset/source/Lost_in_the_Switch_Merged_S01-S100.xlsx --hindi-fixes dataset/data/hindi_fixes.jsonl -o draft.jsonl
python3 -m dataset.compile draft.jsonl --rules dataset/data/rules_grading.jsonl --follow-ups -o dataset/data/conversations.jsonl
python3 -m dataset.remix dataset/data/conversations.jsonl -o dataset/data/variants.jsonl
```

`draft.jsonl` is only an intermediate file. `compile` currently reports 0 problems and 6 warnings.

## About the rules

The workbook has no rules, so `rules_grading.jsonl` is the team's rules file with these changes:

- ids changed from `LIS-S03` to `S003`
- static checks on an English word are `english_only` (the word is always absent from a Hindi reply)
- rules added for the conversations that had none, and for S014 and S031 (51 rules in all, drafted from the turns and the Metadata sheet)
- turn 1 of S021, S022, S026, S028, S044, S045 is only an instruction, so those rules are graded from turn 2
- fixes to S034 and S067

## Follow-up turns

A rule stated at or near the last turn has no later turn where it can slip, and in `EN_HI_EN` the last turns are
English. `compile --follow-ups` adds two user turns ("second version of that answer, keep following everything I asked
for earlier", then "third version") to the 52 conversations that need them. In `EN_HI_EN` the first is Hindi and the
second English, so every such rule is checked after a switch and a switch back.

## Where the dataset stands

`data/rules_grading.jsonl` (233 rules, all 100 conversations) compiles with 0 problems into `data/conversations.jsonl`,
and `data/variants.jsonl` holds the 400 variants (4 language types) for the runner. 52 conversations got two follow-up
turns (`--follow-ups`). Left before the freeze:

1. **Rules by the authors.** 51 of the 233 rules were drafted from the workbook, not by the authors. They check them.
2. **Hindi review.** Not done: the reviewer columns of the workbook are empty. A named native reviewer goes through the
   Hindi turns, the follow-up turns and `data/hindi_fixes.jsonl`.
3. **Warnings.** The 6 warnings below.
4. **Freeze.** Tell the team when the files above are final. Re-run the three commands if anything changes.

## Warnings that remain

`compile` prints 6 warnings. Each is a user turn that claims something about an earlier assistant reply (S001 turn 4,
S002 turns 6 and 9, S065 turn 8, S088 turn 9, S095 turn 3). Replies are made live, so the claim may not match what the
model said. Rewriting them would change the benchmark turns (S002 turn 9 is a deliberate challenge to the assistant),
so they stay as they are.

32 rules are one-shot checks: their window ends before the last turn on purpose (a single request such as "exactly
five suggestions", or a rule the user replaced). `compile` counts them but does not warn about them.

## Notes

- Ids are not renumbered, because rules and results refer to them. A duplicate id is a problem.
- The spreadsheet's `English + Hindi` version is not used (the remixer builds the language types from the rules). It
  is only checked against the English and Hindi turns.
- A turn is a side question when it is one of the added unrelated questions ("Quick side question", "A small unrelated
  question", "One more quick detour"). The final "Please now write the full ..." turns are normal turns.
