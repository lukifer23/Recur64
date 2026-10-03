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

The formal release qualification is bound to source
`028025da1c7486eb0aa9140a88509c2822275e8a`. CPU and the intended RTX 2050 CUDA
device both passed Q2/Q4/Q8 x R1/R2/R4 forward/backward/AdamW, 50 repeated
Q8/R4 updates, R8 forward, exact paired-null equality, reader gradient coverage,
frozen-baseline integrity and full model/optimizer restoration at physical
microbatch 2.

| Measurement | CPU FP32 | CUDA FP32 |
|---|---:|---:|
| Worst warm matrix update | 0.7318495 s | 0.4693646 s |
| Repeated Q8/R4 first update | 0.7213376 s | 0.1897180 s |
| Repeated Q8/R4 last update | 0.7013312 s | 0.2006881 s |
| Paired all-null centered error | 0 | 0 |
| Payload input-gradient L2 | 0.0003146815 | 0.00038374067 |
| R8 forward-only | 0.4802795 s | 0.0818821 s |

The CUDA device-wide used-memory peak was 338 MiB versus a 144 MiB pre-run
baseline, a 194 MiB delta and well below the 3,072 MiB preference. This is
device-wide `nvidia-smi` accounting under WDDM, not a process-only resident peak.
See `docs/evidence/v5/cpu-qualification-release.json` and
`docs/evidence/v5/cuda-qualification.json`.

This is engineering evidence only. The exact P25 TRAIN artifact is still absent,
so no memorization drill, Stage A training, Stage B training, DEV evaluation or
pilot classification has run.

V4_TUNE_V1 AND HOLDOUT_C REMAIN UNEVALUATED.
