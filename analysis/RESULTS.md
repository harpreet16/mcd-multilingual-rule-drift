# Results

Numbers produced by `python3 -m analysis.preregistered_tests --json runs/preregistered_results.json`.
Re-run it after any change to the grades; the JSON it writes is committed so the prose below can be
checked against it.

Five models, 87 conversations each (the ones Qwen3-32B could finish), four language types.
1,740 conversation labels in total = 5 x 4 x 87, every response labelled exactly once.

## The short version

Rules survive in English and break when the conversation moves to Hindi. The effect is consistent
across every model but is **not significant model by model** at this sample size; it is significant
when the five models are taken together.

## The headline

**Direction.** Comparing each model's English baseline with each of its three Hindi conditions gives
15 cells. The Hindi condition is worse in:

| metric | cells worse under Hindi | sign test |
|---|---|---|
| `slip_rate` | **15 of 15** | p = 6.1e-05 |
| `loss_rate` | 14 of 15 | p = 0.00098 |

**Size, pooled over the five models** (exact McNemar on the summed discordant pairs, inclusive
counting):

| comparison | `slip_rate` | `loss_rate` |
|---|---|---|
| EN vs HI_ALL | p = 3.3e-06 | p = 2.5e-07 |
| EN vs HI_EN_HI | p = 2.0e-05 | p = 2.5e-07 |
| EN vs EN_HI_EN | p = 0.0095 | p = 0.018 |

**The ordering is the finding.** Staying in Hindi (`HI_ALL`) and ending in Hindi (`HI_EN_HI`) cost
far more than switching out and back (`EN_HI_EN`). More Hindi, more drift — a dose-response, not a
flat "switching breaks rules".

## What this does not support

**Per-model significance.** Under PLAN.md step 4 — one model at a time, Holm over the five models —
only two or three of the fifteen comparisons survive, and which ones depends on the metric. With 87
conversations that is an underpowered test, not evidence of no effect: the direction is consistent
in 15 of 15 cells, which is exactly the pattern of a real effect measured too coarsely.

Report the aggregate. Do not report fifteen separate significant results.

**Turn-level tests.** An earlier version of these numbers treated every rule-turn as an independent
paired observation, which gave much smaller p-values. It should not be used. Once a model drifts
into Hindi at turn 4, turns 5 to 12 are not independent evidence of anything; they are the same
event counted nine times. PLAN.md step 4 specifies the conversation as the unit, and it is right.

## Counting policy

Both policies are reported in the JSON. The difference is whether a `NOT_APPLICABLE` produced by an
`english_only` rule meeting a Hindi reply counts as the rule being broken.

- `strict` — it does not. PLAN.md as written.
- `inclusive` — it does. The judge's own NOT_APPLICABLE never does, under either policy.

The headline table above is `inclusive`. Under `strict` the direction is identical (still 15 of 15
and 14 of 15) and the pooled comparisons against `HI_ALL` and `HI_EN_HI` stay significant; only
`EN_HI_EN` loses significance. **The true effect lies between the two**, because a Hindi reply to a
Hindi user neither proves the rule was broken nor proves it was kept. See `runs/README.md`.

## Leakage, and a split worth keeping

Of the 2,945 rule failures outside the English baseline, 2,536 (86%) happened while the model was
replying in Hindi or mixed script. That average hides two different failure modes:

| family | failed while replying in Hindi/mixed |
|---|---|
| llama31_8b, llama32_3b, llama33_70b | 95-98% |
| qwen3_4b, qwen3_32b | 63-71% |

The Llamas are dragged into Hindi and the rule goes with it. The Qwens break rules **in English**
about a third of the time, which is forgetting rather than leakage. Figure 4 reports both.
