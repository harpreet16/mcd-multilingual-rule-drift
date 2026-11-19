# Plan

## What we are testing

After a mid-conversation language switch (English to Hindi and back), do language models keep following the rules the user gave earlier?

- 100 conversations, each built in 4 language types: `EN`, `EN_HI_EN`, `HI_ALL`, `HI_EN_HI`
- 5 models: Qwen3-4B, Llama 3.2-3B and Llama 3.1-8B run on Sol (vLLM). Qwen3-32B and Llama 3.3-70B run on Amazon Bedrock
- About 2,000 conversation runs

## How it works, in plain words

Think of a teacher checking whether a student remembers instructions.

1. **Give an instruction early.** For example: "From now on, start every answer with 'Let's talk about that.'"
2. **Keep chatting.** Ask normal questions, with a few unrelated ones mixed in. Some of the chat is in Hindi.
3. **Check every answer after the instruction.** Easy rules are checked by a simple program. Judgement rules ("is the tone formal?") are checked by an AI judge.
4. **Fill in a report card.** Each rule gets a row, each turn gets a pass or a fail.

```
Rule: start with "Let's talk about that."
Turn:     2    4    5    6
Result:   ok   ok   X    ok
```

5. **Label what happened.** Kept (never broke), Recovered (broke in the middle, followed again later), Lost (still broken at the end).
6. **Compare.** The same conversations run in English and in the Hindi types. Do the models slip more when the language switches?

## Files and formats

Data files are JSON lines (`.jsonl`), one object per line. The exact fields are in `contracts/models.py`, the allowed words in `contracts/constants.py`, and one example of each is in `contracts/examples/`.

| File | One row is | Written by | Read by |
|---|---|---|---|
| `conversations.jsonl` | A conversation: user turns in English and Hindi, plus rules | `dataset/compile.py` | `dataset/remix.py` |
| `variants.jsonl` | One conversation in one language type | `dataset/remix.py` | `runner` |
| `responses.jsonl` | All the replies a model gave to one variant | `runner` | `grader` |
| `grades.jsonl` | One verdict for one rule at one turn | `grader` | `analysis` |
| `results.json` | The final numbers per model and language type | `analysis` | `dashboard` |

**Turns are user messages only.** Assistant replies in the source data are dropped. The runner makes the model's own replies live, and each reply is added to the history before the next turn is sent.

**Verdicts for one rule at one turn:** `PASS`, `FAIL`, `NOT_APPLICABLE`, `UNGRADED`. The last two are never counted as a fail. An API error or an unreadable judge answer is `UNGRADED`.

**Rule labels:** `KEPT` (no fail, and at least 80% of its turns graded), `RECOVERED` (a fail, but the last graded turn passes), `LOST` (the last graded turn fails), `UNKNOWN` (no fail, but under 80% graded).
**Conversation labels:** `LOST` if any rule is lost, else `RECOVERED` if any rule recovered, else `UNKNOWN` if any rule is unknown, else `CLEAN`.
The judge never gives these labels. Code in `analysis/` works them out from the grades.

## The hand-offs between modules

1. **Model client:** `chat(messages) -> (text, cut_off)`. It asks again with double the tokens when a reply is cut off (up to a cap) and raises `ServerUnreachable` when it cannot connect. Files: `models/sol/client.py`, `models/bedrock/client.py`
2. **Static check:** `run_static_check(rule, reply, reply_lang) -> "PASS" or "FAIL"` in `grader/static_checks.py`
3. **Judge:** `judge_response(response, variant, asks, rubric_version) -> list[Grade]` in `grader/judge.py`. `asks` is a list of `(rule, turn)` pairs. The judge returns one Grade per ask, with `judge_result` set to YES, NO, NOT_APPLICABLE or ERROR

To change a format, everyone agrees first, then `models.py`, the example file, the checker and this plan change in one pull request.

## How grading works

Every rule is graded at every turn after it was stated. Side-questions are skipped. The steps are in the docstring of `grader/hybrid.py`.

- **Static rules** (clear answer): checked by code. A static fail on a Hindi or mixed reply goes to the judge, which decides, because a correct Hindi reply can fail a word check. In an audit of 98 Hindi replies, 16 of 18 static disagreements were false fails
- **Meaning rules:** the judge answers a yes or no question about the turn
- **The judge never sees the model name or the language type.** It gives a verdict per rule and turn, never an overall verdict
- **Judge:** Claude Haiku 3.5 on Bedrock, run from Sol. It may answer with one call per ask or one call per conversation. Pilot both on about 50 conversations and keep the one-call version if it agrees on about 90% of turn verdicts

## Analysis steps

Fixed before results are seen and not changed afterwards.

1. **Label every rule** with the labels above, using only turns graded PASS or FAIL. Record the first lapse turn and the recovery turn (`analysis/metrics.py`, `label_rule`)
2. **Label every conversation**
3. **Two headline numbers** per model and language type, over conversations that are not UNKNOWN:
   - `slip_rate` = (RECOVERED + LOST) / (CLEAN + RECOVERED + LOST). Did the model ever forget a rule?
   - `loss_rate` = LOST / (CLEAN + RECOVERED + LOST). Did a rule end up broken?
   Neither is called a fail. Report both with equal weight
4. **Compare English with each Hindi type.** Each conversation against itself, one model at a time, with an exact McNemar test on both rates and a correction for 5 models (Holm). Report beside them: pass rate per turn, lapse rate, first lapse turn, reply language at each turn
5. **Forgetting or leakage.** Split failures by whether the reply was in the right language. A model that answers in Hindi when it should keep English is being pulled along, which is not the same as forgetting

Always report the share of `UNGRADED` turns and the cut-off reply rate next to every number.

## Runner and models

- **Turn by turn:** turn 1 is sent, the reply is added to the history, then turn 2 is sent with everything so far
- **Resume:** progress is saved to `partial.json` after every reply. Run the same command again and saved replies are put back without asking the model again
- **Finished runs are skipped** when the same model and settings are already saved. Change a setting and the run is redone
- **Cut-off replies:** asked again with double the tokens (4,096 then 8,192). Still cut off at the cap: saved as cut off, and graded `UNGRADED`
- **Failures:** one failed conversation does not stop the others. Run again and only the failed ones are redone. If the server cannot be reached, the run stops
- **Conversations at once:** 16 on Sol, 4 on Bedrock, set in `models/settings.py`, which also explains the reasons for each setting
- **Same for every model:** temperature 0, 4,096 tokens, thinking off for Qwen. The model id, backend and settings are saved with every response

## Writing a conversation

- User messages only. Drop assistant replies from the source data
- Flag any user turn that quotes or points to something the assistant said. Rewrite it so it works with the model's own reply, or mark it as a special case
- Each rule is testable at 2 or more turns after it is stated, so a later slip would show (`contracts.check.warn_conversation` tells you)
- Rules are stated early, in the middle and late
- A rule the user later changes or cancels must end its window (`to_turn`) at that point
- No two rules in a conversation may conflict
- Hindi is written or checked by someone other than the author of the English

## Making sure the results can be trusted

- The English baseline shows normal forgetting, so only the difference from English counts
- Several measures side by side. If they agree, the result is robust
- A hand check of about 50 judged rows: open `grades.jsonl` in a spreadsheet, read them, and write down how many you agree with
- Optional: a whole-conversation judge on about 50 conversations, compared with the computed result
- A few rule by turn grids read against what the models really said

## Build order

1. Contracts, runner and Sol job (done)
2. In parallel: dataset and remixer, Bedrock client, static checks, judge, analysis, dashboard
3. Freeze the dataset, then the full runs. There is no separate pilot. Watch the first conversations that finish and check them: no `<think>` text, no empty replies, how many replies were cut off. Runs are resumable, so a fix does not lose work
4. Grading, then analysis, graphs, dashboard and the report

## Still to decide

- Confirm Haiku 3.5 is available in the Bedrock account and region, and pin its model id
- Can Sol compute nodes reach Bedrock? The judge is run from Sol
- Who reviews the Hindi, and how many items
- Bedrock budget. Generation on Bedrock (two models) and the judge both cost money, and every turn sends the whole history again
- Deadlines: contributors' conversations, and the frozen dataset

## What the study does not cover

- Reply quality or factual accuracy
- Whether the Hindi translations are right, unless a native speaker reviewed them
- How good the judge is, beyond the hand check
- A rule the user changes or cancels later, unless its window was ended in the data
- Rules that conflict with each other
- Rules that need English words when the model answers in Hindi
- The language a rule was stated in, beyond what the remixer chose
