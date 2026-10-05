# V5 results

> CURRENT V2 HANDOFF: consumer source d11659e, fresh CPU/RTX2050 CUDA exact D9
> PASS,12 repeated normal/profile comparisons exact, graph provenance PASS.
> Native TRAIN27000/DEV4500/CONFIRM4500 generated/audited/byte-identically
> regenerated, zero intersections, all actual local custody PASS. CONFIRM sealed
> and UNEVALUATED. Data-v2/recipe-v3 bound.24-position Q8/R4 drill PASS:
> 2.987107 ->0.026625 loss,99.11% reduction,200 updates, baseline exact.
> STOP BEFORE STAGE A. Stage A/B, DEV model science/pilot, controller, replication,
> V4_TUNE/HOLDOUT_C/CONFIRM evaluation NOT RUN. V1/P25 commands and earlier
> statuses below are preserved history. See V5_DATA_V2_REPORT.md.


> CURRENT CUDA RECOVERY (003d296): fresh CPU/CUDA qualification PASS under the
> unchanged exact D9 and synchronized execution contract. Three repetitions of
> each independent normal/profile order are EXACT; the proven cause was a
> pitched boolean-mask reshape stride defect. See V5_CUDA_MASK_RECOVERY.md.
> Historical failed reports are unchanged. Release workspace: 587 passed,
> zero failed, two preserved ignores. Changed-file rustfmt/V5 Clippy PASS;
> workspace-wide formatting has pre-existing drift in 16 unchanged files.
> Native dataset quota STOP remains; no drill, Stage A/B, DEV or pilot authorized.
> Historical status/commands below do not override this recovery or data lock.


> MEASURED DATA STOP (df261ab): exhaustive KRvK M1 capacity is only 189
> canonical classes, required 2000. Two full enumerations reproduce exactly;
> 189 independent audits, zero failures. No TRAIN/DEV/CONFIRM dataset accepted,
> no custody/seal or recipe-v2 integration. See V5_RECOVERY_REPORT.md. D9 FAIL.


> CURRENT OWNER AMENDMENT: P25 transfer/dependency is retired. New requested
> lineage is V5_HP_TRAIN_V1/DEV_V1/CONFIRM_V1; see V5_DATA_PLAN.md, registered
> before generation. Production scientific data loaders are locked until complete
> measured native DATA-B and recipe-v2 integration. The legacy P25 counts, digests,
> partition and commands below are historical and cannot authorize current work.
> CUDA diagnostic ad80002 is CASE C: clone purity PASS, each mode self-exact,
> cross-mode FAIL localized to frozen-root completion/lift fences. D9 remains
> FAIL. No Stage A/B, DEV science or drill authorized by this recovery result.


**HISTORICAL d970049 STATUS: SYNCHRONIZED CPU PASS / CUDA PROFILING PARITY FAIL; STOP.**

Historical scientific source: `d97004981f52eb077da6bb72584efaca341b8c5b`.
Full release workspace: **581 passed, zero failed, two preserved ignores**;
native suite exit 0. Scoped fmt/V5 Clippy and the serial pinned CUDA build pass.
The new same-device profiling comparison is exact on CPU, but its combined
output/gradient and post-AdamW parameter/moment checks FAIL on RTX 2050 CUDA.
The failed qualifier exited 1 and its report is retained. No gate/tolerance was
loosened. `V5_PROFILING_ROOT_CAUSE.md` separates measured facts from unverified
causes and specifies the next bounded investigation. This is not a reader pilot.

All nine paired shapes, 50 resident updates, null correction, baseline integrity,
graph-free reference, full checkpoint/moment restoration and ordinary resumed
continuation pass on both devices; these cannot override the profiling failure.

| New synchronized fixture measurement | CPU FP32 | CUDA FP32, FAILED qualification |
|---|---:|---:|
| Qualifier wall | 50.8978161 s | 32.0531012 s |
| Worst warm update, final AdamW completion fence | 0.7291423 s | 0.3128205 s |
| Resident first / last update | 0.7210959 / 0.7019132 s | 0.1977508 / 0.2020739 s |
| Dedicated instrumented Q8/R4 update wall | 0.7159244 s | 0.1837027 s |
| Frozen root encoder/candidate path | 0.0301599 s | 0.0051423 s |
| Returned-state encoder | 0.1201092 s | 0.0061372 s |
| Factual stream, initialization + four E/H iterations | 0.0669607 s | 0.0175878 s |
| Null stream, initialization + four E/H iterations | 0.0648321 s | 0.0192178 s |
| Backward, both streams | 0.4049157 s | 0.0759242 s |
| AdamW + completion fence | 0.0182239 s | 0.0375022 s |

These single fixture timings include fences; transfer/host work is combined,
and GPU profiling is unqualified. No full-data stage projection or online
decision latency is inferred. Per-loop phases and checkpoint measurements are
in `qualification-d970049-{cpu,cuda}.json`. Requested/actual Q8 was [8,8], depths
[3,3], branch coverage [4,5], 2 root encoder examples, 16 state encoder examples,
16 physical state rows/zero padding, 34 legal candidates per position/68 physical
candidate rows. Both streams execute 16 core applications per example, 32
example-applications for the batch. Host preparation duplicates 32 raw packet
preparations and four CandidateFacts calls/136 root successor-board inspections;
these common root facts do NOT consume Q. Exact reply enumeration is not counted.

CUDA sampled device-wide baseline/peak: 138/1,100 MiB (962 MiB increase); every
resident sample is 364 MiB. These are not continuous/process-only peaks. Microbatch
2 fits these fixtures, but the execution failure blocks qualification and training.
The actual parameter/configuration identities remain unchanged. TRAIN custody,
FIT drill, baseline/reader training and ALL DEV/pilot science remain NOT RUN.

## Historical 64c4dd4 functional fixture qualification

Historical scientific source: `64c4dd4008a9b5bc8d715515279ec67e6e1173a1`.
Full release workspace: **578 passed, zero failed, two explicitly ignored**.
The corrected paired graph passed CPU and RTX 2050 CUDA in FP32 at physical
microbatch 2: all nine Q2/Q4/Q8 Ã— R1/R2/R4 forward/backward/AdamW conditions,
50 resident Q8/R4 updates, and forward-only engineering R8. Both streams are
differentiated. Exact graph-free/autodiff baseline reference, frozen baseline
outputs and ALL parameters after reader updates, ALL model/moment restoration,
and one continued model/moment update passed on both devices. CPU trainer
sampler/schedule/graph continuation is also exact in the release contract suite.

| Qualification measurement | CPU FP32 | RTX 2050 CUDA FP32 |
|---|---:|---:|
| Worst warm qualifier update | 0.7248367 s | 0.4127371 s |
| Resident Q8/R4 first / last update | 0.7139785 / 0.7074989 s | 0.2029461 / 0.1985860 s |
| Paired all-null centered error | 0 | 0 (limit 1e-6) |
| Returned-payload input gradient L2 | 0.0039228043 | 0.00059487484 |
| Engineering R8 forward | 0.4974181 s | 0.0796470 s |
| Raw two-position Q8 acquisition time | 0.0043778 s | 0.0044342 s |

State encoder, evidence block, hypothesis block and correction readout all have
finite nonzero gradients. The four fixed Q2/R2 candidate-relative perturbations
have input-gradient L2 7.208392617030768e-6 and relative finite-difference errors
0.04229402317664573, 0.052588879717833686, 0.0002938875522951624,
0.05313823775366243, below unchanged 0.12. Their test-only same-weight FP64 tensor
reference retains pinned FP32 RMS statistics; production remains FP32.

CUDA device-wide sampled used-memory baseline/peak was **138 / 1,068 MiB**
(930 MiB increase). Every one of the 50 resident update samples was **364 MiB**;
no growth was observed in that bounded series. The larger sampled peak includes
checkpoint restoration, not just the resident update loop. These are WDDM
device-wide snapshots, NOT a process-only or continuously sampled true peak.
Microbatch 2 is resolved; accumulation 32 for Stage A and 18 for Stage B.

Timings are qualifier-harness wall intervals, including gradient-coverage host
reads, not isolated kernel/encoder/backward/transfer benchmarks or online active
decision latency. Loss/gradient reads synchronize execution; no explicit
post-AdamW barrier is present at each timer end, so optimizer completion may be
charged at the next synchronization. Do not present these as fully synchronized
per-component timings or extrapolate a full-data stage duration from them.
CPU/CUDA initializers differ by backend; no same-seed cross-device weight or
policy parity is claimed. Within-device exact baseline/null/resume comparisons
use identical tensors and weights.

Reports: `qualification-64c4dd4-{cpu,cuda}.json`, `gradient-64c4dd4.json` and
`softmax-execution-parity.json` under `docs/evidence/v5/`. Canonical qualification
paths held the same reports at that milestone; historical source reports remain
archived. V5-E12/E13 preserve the Windows build-sharing and missing-header runtime
failures and their serial-build/process-local setup corrections.

This clears the recorded FUNCTIONAL fixture qualification, not the FIT drill,
all performance-accounting requirements, or any learned-reader pilot gate.
No exact P25 custody, 24-FIT drill, Stage A, Stage B, DEV reader matrix, treatment/
composition measurement, bootstrap classification or conditional scientific R8
has run. Detailed component/end-to-end performance accounting remains to be
completed before measured data training; do not infer it from these warm timings.

## Retained depth-amended baseline parity failure

Source 7737640 passed the full release workspace (573 tests, two ignored), but
its measured CPU qualification exited 1 solely at graph-free/autodiff exact
baseline equality. Context error was 1.6689300537109375e-6, hypotheses
2.205371856689453e-6 and z0 7.078051567077637e-8. Null error was zero; baseline
outputs/parameters after reader updates and full parameter/optimizer restoration
and continuation were exact. These passing components do not clear the gate.
`V5_EXECUTION_PARITY.md` records the isolated dispatch cause and corrective
execution experiment. The fresh functional qualifications above now pass.
The complete uniform-frontier correction now includes all 19 legal edges at the
previous failing boundary. Depths 6..16 and their structural fields are tested.
The focused release suite passes 28 tests, with one explicitly invoked numerical
diagnostic ignored by default. Production FP32 autodiff matches a same-weight
FP64 numerical reference under the unchanged 12% limit in all four fixed
directions. See `V5_ROOT_CAUSE.md` and `V5_NUMERICAL_ROOT_CAUSE.md` for retained
failures and corrections. No dataset-dependent result exists.

The amended model's measured count is **7,162,896**. Its evidence initializer is
139,776, up by 2,816; all other groups below are unchanged. Current execution
amendment model-info config digest:
`849133a5cdf169f187778bace2f858aa4747d2e8defef3bb5ac1bffc839774ee`.

## Historical pre-amendment engineering measurements

The superseded depth-five graph had 7,160,080 unique parameters:

| Group | Parameters |
|---|---:|
| Root/baseline path | 3,677,728 |
| Returned-state encoder | 1,633,808 |
| Hypothesis adapter | 65,792 |
| Evidence initializer | 136,960 |
| Shared evidence block | 789,872 |
| Shared hypothesis block | 789,872 |
| Correction readout | 66,048 |

The historical network-fixture qualification is bound to scientific source
`df6e2aa650c12726ad7094dae04a7c73339139a8`. CPU and the intended RTX 2050 CUDA
device both passed Q2/Q4/Q8 x R1/R2/R4 forward/backward/AdamW, 50 repeated
Q8/R4 updates, R8 forward, exact paired-null equality, reader gradient coverage,
frozen-baseline integrity and model-output restoration with optimizer loaded at physical
microbatch 2.

The later 97921bd CUDA build passed, but its new measured CPU/CUDA qualification
was stopped before launch by the acquisition-contract failure. No pilot result
or architectural falsification can be inferred from this implementation defect.

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
See the archived `docs/evidence/v5/qualification-df6e2aa-cpu.json` and
`docs/evidence/v5/qualification-df6e2aa-cuda.json`, not the current canonical paths.

This is engineering evidence only. The exact P25 TRAIN artifact is still absent,
so no memorization drill, Stage A training, Stage B training, DEV evaluation or
pilot classification has run.

The full authorized harness is implemented, including the disposable Q8 drill,
update-0/update-800 six-cell evaluation, payload/feedback/relation/null
interventions, composition analysis, exact bootstrap gates and conditional R8.
Implementation is not a result: none of those dataset-dependent commands ran.

V4_TUNE_V1 AND HOLDOUT_C REMAIN UNEVALUATED.
