# V5 results

## Engineering measurements

The implemented frozen graph has 7,160,080 unique parameters:

| Group | Parameters |
|---|---:|
| Root/baseline path | 3,677,728 |
| Returned-state encoder | 1,633,808 |
| Hypothesis adapter | 65,792 |
| Evidence initializer | 136,960 |
| Shared evidence block | 789,872 |
| Shared hypothesis block | 789,872 |
| Correction readout | 66,048 |

The 2026-10-03 debug CPU preflight passed Q2/Q4/Q8 x R1/R2/R4
forward/backward/AdamW, 50 repeated Q8/R4 updates, R8 forward, exact paired-null
equality, reader gradient coverage, frozen-baseline integrity and full
model/optimizer restoration. Its timings are diagnostic debug timings, not the
final release qualification envelope. See
docs/evidence/v5/cpu-qualification-debug.json.

This is engineering evidence only. The exact P25 TRAIN artifact is still absent,
so no memorization drill, Stage A training, Stage B training, DEV evaluation or
pilot classification has run.

V4_TUNE_V1 AND HOLDOUT_C REMAIN UNEVALUATED.
