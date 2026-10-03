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

The final formal release qualification is bound to scientific source
`df6e2aa650c12726ad7094dae04a7c73339139a8`. CPU and the intended RTX 2050 CUDA
device both passed Q2/Q4/Q8 x R1/R2/R4 forward/backward/AdamW, 50 repeated
Q8/R4 updates, R8 forward, exact paired-null equality, reader gradient coverage,
frozen-baseline integrity and full model/optimizer restoration at physical
microbatch 2.

| Measurement | CPU FP32 | CUDA FP32 |
|---|---:|---:|
| Worst warm matrix update | 0.7699340 s | 0.4107268 s |
| Repeated Q8/R4 first update | 0.7603199 s | 0.1865621 s |
| Repeated Q8/R4 last update | 0.7372230 s | 0.2088036 s |
| Paired all-null centered error | 0 | 0 |
| Payload input-gradient L2 | 0.0003146815 | 0.00038374067 |
| R8 forward-only | 0.4794547 s | 0.0801630 s |

The CUDA device-wide used-memory peak was 338 MiB versus a 144 MiB pre-run
baseline, a 194 MiB delta and well below the 3,072 MiB preference. This is
device-wide `nvidia-smi` accounting under WDDM, not a process-only resident peak.
See `docs/evidence/v5/cpu-qualification-release.json` and
`docs/evidence/v5/cuda-qualification.json`.

This is engineering evidence only. The exact P25 TRAIN artifact is still absent,
so no memorization drill, Stage A training, Stage B training, DEV evaluation or
pilot classification has run.

The full authorized harness is implemented, including the disposable Q8 drill,
update-0/update-800 six-cell evaluation, payload/feedback/relation/null
interventions, composition analysis, exact bootstrap gates and conditional R8.
Implementation is not a result: none of those dataset-dependent commands ran.

V4_TUNE_V1 AND HOLDOUT_C REMAIN UNEVALUATED.
