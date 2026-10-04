# V5 synchronized profiling qualification failure

2026-10-03, source `d97004981f52eb077da6bb72584efaca341b8c5b`.
Decision V5-D9 remains unchanged. CPU PASS; RTX 2050 CUDA FAIL. STOP before
the FIT drill or pilot. This is an unqualified execution experiment, not a
negative learned-reader pilot or architectural falsification.

## Measured boundary

Reports: `docs/evidence/v5/qualification-d970049-{cpu,cuda}.json`.
Architecture/configuration remain unchanged: 7,162,896 parameters, FP32,
microbatch 2, configuration digest
`849133a5cdf169f187778bace2f858aa4747d2e8defef3bb5ac1bffc839774ee`.
Operational timing digest:
`af2a950d922443a73d448fbdc3ab87bfc50a011dea4ea350516ffffa14cad20f`.

| Current check | CPU | CUDA |
|---|---|---|
| Combined logits/centered delta/loss/input gradient/all parameter gradient equality, normal versus profiled | exact | FAIL |
| Combined ALL post-AdamW parameter/moment equality, normal versus profiled | exact | FAIL |
| Observed phase accounting | PASS | PASS |
| Paired all-null centered correction | 0, exact | 0, below unchanged 1e-6 |
| Graph-free baseline versus reference | exact | exact |
| Frozen baseline outputs and ALL parameters after updates | exact | exact |
| ALL checkpoint parameters and optimizer moments restored | exact | exact |
| Uninterrupted versus resumed parameters and moments after one unprofiled update | exact | exact |
| State/evidence/hypothesis/readout gradient groups | finite/nonzero | finite/nonzero |

Both devices actually executed all nine Q2/Q4/Q8 x R1/R2/R4 paired training
shapes, full backward/AdamW, 50 resident Q8/R4 updates and engineering-only
forward R8. First-legal fixture targets are arbitrary execution-test labels,
NOT proof-correct chess labels, the 24-FIT drill or evidence of chess learning.

CUDA exited 1 with its FAILED report written. Sampled device-wide baseline/peak
was 138/1,100 MiB (962 MiB increase), and all 50 resident samples were 364 MiB.
Neither a continuous/process-only memory peak nor a passing execution envelope
is claimed. No unrelated process was stopped and no CPU substitution occurred.

## What is and is not isolated

The comparison replays the same model clone, graph, targets and starting AdamW
state within each device. Its instrumented arm adds pinned Backend::sync fences
at component boundaries; normal execution already has the final AdamW fence.
CPU equality and CUDA ordinary checkpoint continuation equality pass. The
additional CUDA profiling comparison does not.

The current report stores aggregate parity booleans and hashes, not per-component
numeric differences or a normal-normal/profile-profile control. Consequently it
does NOT establish which output/gradient field first differs, the magnitude of
the difference, whether both parameter and moment hashes differ individually,
or whether the original model/optimizer clone remained untouched. An underlying
Burn/CubeCL defect, layout-dependent FP32 reduction order, allocator/alias effects
or CUDA nondeterminism are hypotheses, not verified root causes. Do not claim
that all logits differ or that the discrepancy is harmless roundoff.

## Next bounded investigation, not a remedy already applied

Before any further qualification/training, add per-field numeric/hash diagnostics
and original-model/optimizer fingerprints before and after each replay. On the
SAME real fixture reader and stored weights/state, compare normal-normal,
profile-profile and normal-profile, including every gradient component and
post-AdamW parameters/moments. Localize a first differing boundary with identical
acquired graphs. Preserve failed evidence and retain the exact D9 gate; do not
change precision, geometry, gradients, LR, samples, tolerances or null treatment
to obtain a pass. This investigation does not authorize a redesign or LR screen.

A separate read-only contract audit found standalone `v5 graph generate` emits
an AcquiredGraph with configuration/graph digests but no explicit source-code
SHA field. Qualification/evaluation bundles carry source identity, but standalone
graph provenance needs an explicit implementation/validation audit before the
pilot. Do not describe all scientific-identity gates as closed.

Exact P25 TRAIN is also still absent. Custody, FIT drill/Q16, Stage A/B, DEV,
pilot gates, composition/interventions, scientific conditional R8, replication,
controller training and sealed confirmation remain NOT RUN. No parameters from
this disposable qualification may initialize the drill or pilot.

## Recovery measurement at ad80002: independent CUDA snapshots

Engineering-only diagnostic, native exit 0; not a qualifying PASS. Source:
`ad80002`, Burn 0.21.0 / CubeCL 0.10.0, RTX 2050 driver 616.92,
pinned CUDA 12.9.1, FP32 physical 2, unchanged config. Full report:
`docs/evidence/v5/profile-diagnostic-ad80002-cuda.json`.

Canonical model and populated AdamW state saved by the tested full-precision
checkpoint recorder after one ordinary Q8/R4 fixture update. Snapshot includes
both real fixture graphs, first-legal execution targets (not chess labels), model
and optimizer digests, config and source. Every replay loads both snapshot files
and verifies exact initial model/moment identity. This is a fresh diagnostic
starting state, not a recovered historical d970049 resident-update checkpoint.

All five original fingerprints remain exact: clone forward plus ordinary,
profiled, reversed profiled/ordinary backward+AdamW. No model/optimizer Clone
contamination is measured. All three NORMAL/NORMAL pairs and all three
PROFILE/PROFILE pairs are exact over every recorded field. All three cross-mode
and three reverse-order pairs differ identically: CASE C.

| Field (first named tensor when applicable) | Max absolute | RMS | Max ULP |
|---|---:|---:|---:|
| logits [2,34] | 0.000786900520324707 | 0.00027925268468156 | 105616 |
| centered delta [2,34] | 0.0007868991233408451 | 0.00027925296461462506 | 13174912 |
| correct-set loss [1] | 0.00039267539978027344 | same | 1647 |
| correction_hidden.bias gradient | 0.000015350407920777798 | 0.000003216976790882095 | 1793660161 |
| correction_hidden.bias post-AdamW | 0.000070914626121521 | 0.000007956640117224964 | 339856 |
| correction_hidden.bias first moment | 0.000001535042429168243 | 0.00000032169778276294743 | 1744092657 |

Large signed ULP distances include sign crossings near zero; relative denominator
floor is fixed at 1e-12. They do not certify harmless roundoff. All 34 differing
logits/centered elements are in the second fixture row. First logits flat index
34: normal -0.08269035816192627, profiled -0.08347725868225098. First centered
value: 0.003384978976100683 versus 0.002598079852759838. Scalar loss:
3.5191752910614014 versus 3.5195679664611816. All optimizer counters exact.
Report contains every tensor summary/digest and absent marker, not tensor dumps.

Forty fresh single-boundary replays were tested. Only fences at
`frozen_root_encoder_and_candidate_path` and `frozen_base_lift` reproduce forward,
gradient, post-AdamW and moment divergence. Earlier upload/model-view fences and
later reader/loss/backward/optimizer fences are exact against ordinary execution.
This localizes a fence-sensitive frozen baseline execution/lift boundary; it does
not isolate an individual Burn/CubeCL kernel or prove its internal cause.
No permanent diagnostic fence, model-math correction or backend change applied.
CUBLAS_WORKSPACE_CONFIG and CUDA_LAUNCH_BLOCKING are unset and not tested here.

Historical D9 remains FAIL exactly as measured. Current diagnostic refutes clone
contamination on this snapshot and does not meet CASE D. No unchanged D9 rerun,
qualification, drill or training is authorized by this measurement.

Future owner-review proposal (NOT ADOPTED): a separately versioned
`v5_execution_profile_qualification_v2` could explicitly distinguish ordinary
execution correctness/repeatability from fence-instrumented profiling semantics.
It must require a proven baseline-boundary cause and same-weight numerical
reference, frozen forward/gradient/optimizer limits established before pilot,
repeatability controls and complete model/moment resume checks. The measured
~7.9e-4 forward shift must first be explained; no proposed numeric limit or new
contract is accepted here, and profiling results remain unqualified.

## Separate environment-only diagnostic results (source df261ab)

CUBLAS_WORKSPACE_CONFIG=:4096:8 and CUDA_LAUNCH_BLOCKING=1 were tested separately,
not adopted as production settings. Both preserve CASE_C and the same numerical
forward divergence and two frozen-baseline fence boundaries. Each mode is exact
within itself in all three repetitions; both cross-mode orders fail all three.
Clone-purity checks PASS. These tests do not prove a kernel-level cause and do
not clear exact D9. Actual pinned Cargo features do not enable Burn fusion.

Receipt: `evidence/v5/profile-environment-df261ab.json`. Full numerical reports
are committed as losslessly compressed JSON with verified decompression hashes.
Each variant has its own independently verified model/AdamW starting snapshot;
no cross-environment optimizer checkpoint identity is asserted.
No fresh qualification, permanent fence, production environment amendment or
new execution contract was adopted. Historical D9 remains FAIL.

## 2026-10-04 pitched-mask recovery (source 003d296)

The subsequent independent diagnosis proves a returned-payload boolean-mask
layout defect: the pinned backend loses pitched batch stride during two
successive trailing dimension insertions. One mathematically equivalent reshape
preserves the stride. CPU and CUDA qualification now PASS under the unchanged
exact D9 contract. All independent normal/normal, profile/profile and both
cross-mode orders are EXACT in three repetitions. Historical measurements and
failed reports above remain unchanged. No new execution contract is adopted.
Detailed source, controls, archives and validation: V5_CUDA_MASK_RECOVERY.md.
Native data quota prerequisites remain unresolved; no drill or learning run.
