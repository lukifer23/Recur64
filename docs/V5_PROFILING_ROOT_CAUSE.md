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
