# V5 Stage A seed5301 results and final B0 publication stop

**Stage A training COMPLETE, EXECUTION VALID. Final DEV B0 report FAILED to
publish; owner review required.** No chess-strength metrics are available.
No post-hoc performance gate is applied.

Starting remote HEAD: `066270cc8818c528c1eeaca0217e85664449ac78`.
Pre-run documentation-only authorization commit: `176044c` (pushed before training).
Scientific source: `d11659eca0774e0064bed0ef64ead2b725886d93`.
Architecture `counterfactual_relational_loop_v1`, 7,162,896 parameters,
config `849133a5cdf169f187778bace2f858aa4747d2e8defef3bb5ac1bffc839774ee`. Model/data/training/evaluation code and Cargo/config
files unchanged throughout this pass. Existing current-source CPU/RTX2050 CUDA
qualifications remain PASS under unchanged exact D9; no qualification rerun or
new scientific source is claimed. Prior workspace validation remains596 passed,
zero failures, two preserved ignores; no source tests/build were needed for this
documentation-only pass.

Data `v5_hp_data_v2`; recipe `v5_stage_recipe_v3`.
Contract digest `4485032edfc1de9a106d66009ccacb33d20e593d7408e9cef218790a5cbba9de`.
Stage A recipe digest `6642579e1f2472bda955ca7ada5bb3b8a435634c023b665684da1e4676347e70`.
Fresh local pre-run raw-file custody and recipe receipts passed before model
initialization. The drill parameters were not reused.

## Fixed Stage A measurement

Run: `runs/v5/v2/seed-5301/stage-a`.
Fresh random seed5301, CUDA FP32, physical2 x accumulation32/effective64,
1200 updates, warmup80, peakLR3e-4, unchanged AdamW/correct-set loss and cosine
schedule. Q0; reader not executed. No DEV during training, early stopping,
adaptation, extra updates or checkpoint selection.

One bounded invocation, native exit0, 1043.721400s
(17.3954 minutes); zero resumes. Recorded update
wall sum 846.809718s; total process time includes setup,
checkpoint/console overhead. CLI training-loop wall was1042.8s.

| TRAIN diagnostic | Measured |
|---|---:|
| First recorded update loss | 3.064389243722 |
| Final recorded update loss | 0.637722653802 |
| First50 updates mean loss | 2.726584702618 |
| Last50 updates mean loss | 0.570068903945 |
| Minimum loss (diagnostic only) | 0.275708013214 |
| Maximum loss (diagnostic only) | 3.099014408886 |
| Final applied LR | 5.900987282647651e-10 |

These are sampled training-update losses, not an initial/final all-TRAIN model
evaluation. All1200 losses finite. Exactly76,800 sampled examples.

| TRAIN cell | Exposure |
|---|---:|
| KQQvK M1 | 8534 |
| KQQvK M2 | 8534 |
| KQQvK M3 | 8534 |
| KQRvK M1 | 8533 |
| KQRvK M2 | 8533 |
| KQRvK M3 | 8533 |
| KRRvK M1 | 8533 |
| KRRvK M2 | 8533 |
| KRRvK M3 | 8533 |

Final immutable checkpoint: `checkpoints/update-000000001200`.
Model SHA256: `2d1c770a43a6455148b774e9ddb552b6ca33efd9fdd5d37593cefe7c8ae0bb00`.
Optimizer SHA256: `cbef56e557f71a9205e34f65c782d9264cabd8fe960e5ab3915790a5f368f03d`.
Exact source/config/recipe/seed/backend/layout and1200 sequential history entries
verified. All175 model tensors (7,162,896 FP32 values) and168 optimizer moment
tensors (7,355,456 FP32 values) finite;84 parameter optimizer states have both
moments and counters1200. Frozen recipe plus update reconstructs deterministic
schedule/sampler; history exposure verified. No gradients were serialized, so
an independent per-gradient finite scan is not claimed. All91 reader parameter
tensors are bit-identical between first saved update10 and final1200; all history
graph/reader-condition records empty. No resume was required; existing exact
checkpoint/moment/resume qualification remains source-bound and passing.

## Exactly one final DEV B0 invocation; validator failure

The authorized CUDA final B0 command was invoked once on update1200 and DEV4500,
physical microbatch2. Native exit1 after19.8672492s:
`baseline sorted DEV identity digest mismatch`.
No `final-baseline-dev.json` was written. No retry, model re-evaluation, source
repair or result reconstruction from another checkpoint was performed.

Read-only control-flow inspection places this error at the final report
validation after all4500 records and preceding identity/metric/alignment checks.
The model ran on all DEV positions once, but its in-memory report was rejected
before serialization. Aggregate metrics, six per-cell rows and primary KQRvK M3
metrics are **UNAVAILABLE**, not zero and not a negative chess-learning result.

| DEV cell | Authoritative n | B0 metrics |
|---|---:|---|
| KQRvK M1 | 750 | UNAVAILABLE: report not serialized |
| KQRvK M2 | 750 | UNAVAILABLE: report not serialized |
| KQRvK M3 | 750 | UNAVAILABLE: report not serialized |
| KRRvK M1 | 750 | UNAVAILABLE: report not serialized |
| KRRvK M2 | 750 | UNAVAILABLE: report not serialized |
| KRRvK M3 | 750 | UNAVAILABLE: report not serialized |

Proven identity-contract mismatch in unchanged `study.rs`:
`BaselineEvaluation.validate` hashes sorted IDs plus newline and compares it to
`DEV_DIGEST`. In V2, that constant is the ProofTargets digest, not the sorted-ID
digest. Actual sorted DEV ID digest:
`8e52094f6e2ec32726efc675acca408b8e2ff4f63a73910d0d00aefe97bc1b5e`.
Compared ProofTargets digest:
`82d4578a62d0727ebca51f412d45e2dcf4e9461838f83148aa70d98b0a69d2a0`.
Raw DEV bytes passed custody both before and after the run. This is a report
validator defect, not evidence of dataset corruption or CUDA D9 regression.

The owner's source-preservation and unexpected-failure stop rules prohibit
repairing evaluation code or repeating this measurement in this pass. A future
owner decision must authorize a narrow identity-contract repair, appropriate
source validation and disposition of the missing B0 measurement. Stage B remains
unauthorized; do not initialize its update0, optimizer or graphs.

## Post-run custody and preserved evidence

Actual TRAIN/DEV/CONFIRM raw-file custody PASS. All three pairwise exact-FEN and
canonical intersections remain0. CONFIRM sealed=true/evaluated=false; no model
was run on CONFIRM. Immutable final checkpoint hashes preserved after evaluation.
Operational chunk/evaluation logs and full checkpoints remain ignored in runs/.
Compact [summary](evidence/v5/stage-a-seed5301-summary.json),
[failure](evidence/v5/stage-a-seed5301-baseline-failure.json),
[pre-run recipe](evidence/v5/pre-stage-a-recipe-d11659e.json),
[pre-run custody](evidence/v5/pre-stage-a-custody-d11659e.json) and
[post-run custody](evidence/v5/post-stage-a-custody-d11659e.json) are committed.

NOT RUN: Stage B including update0/optimizer/graphs; reader pilot; Q/R DEV;
pilot classification; ablations; R8; learned query controller; LR screen; extra
Stage A updates; best-checkpoint selection; seeds5302/5303; self-play; CONFIRM;
V4_TUNE; HOLDOUT_C; repeat B0 evaluation; evaluation-code repair.

V5 STAGE A COMPLETE ? OWNER REVIEW REQUIRED.
V5 STAGE B NOT RUN.
V5 READER PILOT NOT RUN.
LEARNED QUERY CONTROLLER NOT TRAINED.
MULTI-SEED REPLICATION NOT RUN.
V5_HP_CONFIRM_V2 REMAINS SEALED AND UNEVALUATED.
V4_TUNE_V1 AND HOLDOUT_C REMAIN UNEVALUATED.

## V5-E44 / Decision: baseline v3 publication recovery completed

Classification: **EVALUATION IDENTITY-CONTRACT REPAIR**. The first failed attempt
and all prior text above remain historical and unchanged. Owner authorization
for this recovery supersedes only the prior code-repair/replacement-measurement
stop; Stage B remains prohibited.

Starting remote HEAD `8bd4971e8635e94ebb0aea5bbf4dbf75a7647362`.
Preregistration `14318e7`; repair/evaluator scientific source
`b00569b33ced75a0169804a4a3d5b746a1e0e654`. Stage A producer source remains
`d11659eca0774e0064bed0ef64ead2b725886d93`. Only scientific files changed:
`crates/recur64-v5/src/study.rs` and `crates/recur64-cli/src/v5.rs`.
Hash-contract validation, baseline report/provenance, baseline-only exact
predecessor compatibility and focused tests changed. Model/config/mask/training,
loss/optimizer/sampler/data/graph/inference math are byte-identical. Stage B's
source equality path is byte-identical and still refuses the predecessor source.
No Stage B run directory, update0, optimizer or training graphs were created.

Schema `v5_final_baseline_evaluation_v3` explicitly separates evaluator `source_sha`
from model `stage_a_source_sha`, and separates DEV ProofTargets digest
`82d4578a62d0727ebca51f412d45e2dcf4e9461838f83148aa70d98b0a69d2a0` from measured record-ID digest
`8e52094f6e2ec32726efc675acca408b8e2ff4f63a73910d0d00aefe97bc1b5e`. The latter comes from the measured DEV role binding;
report IDs are independently recomputed, sorted with newline framing, and checked.
The normal completed-checkpoint integrity loader still precedes the exact
predecessor bridge. Other predecessor source/model/optimizer/recipe/update/config/
backend/layout/precision/seed/data bindings refuse in focused tests.

Validation: affected rustfmt PASS; focused V5 release52 passed/zero failed/one
preserved ignore; CLI V5 unit3 passed; CLI V5 boundaries7 passed; full release
workspace599 passed/zero failed/two preserved ignores (68 result blocks).
V5 all-target CUDA Clippy -D warnings PASS; CLI CUDA Clippy PASS with only the
existing V4 collapsible_if/manual_is_multiple_of allowances. Serial pinned CUDA
release build PASS. Only affected formatting checked; the known16 unchanged
historical formatting files were not rewritten. Fresh CPU and RTX2050 CUDA
qualification PASS at the evaluator source: FP32/microbatch2,7,162,896 parameters,
unchanged config/exact D9, all nine Q/R shapes, null error0,50 resident updates,
complete checkpoint/moment restore and exact continuation. No older report
was used to authorize new-source measurement.

Exactly one replacement invocation: overall evaluation_attempt=2,
published_baseline_measurement=1, prior_attempt_metrics_exposed=false,
recovery_reason=identity_contract_validator_failure. Native exit0; wall
21.5802839s. No performance-driven retry or threshold.
Raw report is ignored at `runs/v5/v2/seed-5301/baseline-recovery/final-baseline-dev-v3.json` (outside the completed Stage A
run); SHA256 `b59eed52aa09d1c16a0baa367403c1a3ae254065371c87c7ed2b7208e5f74fb4`.
Final update1200, model `2d1c770a43a6455148b774e9ddb552b6ca33efd9fdd5d37593cefe7c8ae0bb00`, optimizer
`cbef56e557f71a9205e34f65c782d9264cabd8fe960e5ab3915790a5f368f03d` unchanged. Baseline fingerprint
`12b272a941e5b29589195a65c779a60509626d2108975e4793674ad5867d75c9`.

| DEV cell | n | Top1 | Correct mass | Set loss | Uniform-correct CE | Entropy |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 4500 | 0.792888889 | 0.723600115 | 0.663078197 | 1.470871026 | 0.871348876 |
| KQRvK M1 | 750 | 1.000000000 | 0.999951952 | 0.000048089 | 0.543484446 | 0.192209719 |
| KQRvK M2 | 750 | 0.726666667 | 0.644706410 | 0.821942259 | 1.875614160 | 1.127769660 |
| KQRvK M3 | 750 | 0.576000000 | 0.448158435 | 1.272790006 | 2.569922149 | 1.580259616 |
| KRRvK M1 | 750 | 1.000000000 | 0.999954517 | 0.000045488 | 0.138887868 | 0.063289175 |
| KRRvK M2 | 750 | 0.750666667 | 0.663929857 | 0.829609338 | 1.622745508 | 0.989415134 |
| KRRvK M3 | 750 | 0.704000000 | 0.584899519 | 1.054034004 | 2.074572028 | 1.275149951 |

Primary KQRvK M3:432/750 top1 (**57.6%**), correct mass0.448158435,
set loss1.272790006, uniform-correct CE2.569922149, entropy1.580259616.
Overall3568/4500 correct (**79.2889%**); legal width mean35.400666667,
range18..49; mean correct-action count1.784000000.
All per-cell width/action/timing diagnostics are in the compact summary. Root
encoder examples4500; returned encoder examples, exact queries and reader core
applications all0. These are B0 baseline measurements for owner interpretation;
no Stage A strength pass/fail threshold is imposed.

TRAIN context is unchanged: first-update loss3.064389243721962; first50 mean
2.7265847026184202; last50 mean0.5700689039449207; final update loss
0.6377226538024843;1200 updates/76800 sampled examples/all finite. Reader parameter
tensors remained bit-identical. Sampled TRAIN losses are not held-out evidence.

Fresh pre/post actual TRAIN/DEV/CONFIRM custody PASS; all FEN/canonical pairwise
intersections0. CONFIRM remains sealed=true/evaluated=false; no model ran on it.
All364 preserved Stage A/failed-attempt files match their pre-repair hashes;
all360 Stage A run files and the entire file set are unchanged. Failed first
attempt JSON/operational evidence remains byte-for-byte intact. Full run files,
weights/optimizer blobs/raw data and raw4500-record report remain ignored.

Compact evidence: `baseline-recovery-decision.json`,
`baseline-recovery-source-scope.json`, `baseline-recovery-validation.json`,
`qualification-b00569b-cpu.json`, `qualification-b00569b-cuda.json`,
`baseline-recovery-pre-custody.json`, `baseline-recovery-post-custody.json`,
`baseline-recovery-preservation.json`, `baseline-v3-seed5301-summary.json`
under docs/evidence/v5/. Full workspace log is losslessly compressed there.

NOT RUN: Stage A retraining/resume; Stage B/update0/optimizer/graphs; reader
pilot; Q/R DEV; pilot report/classification; ablations; scientific R8; drill
rerun; learned query controller; LR screen; checkpoint selection; seeds5302/5303;
self-play; CONFIRM/V4_TUNE/HOLDOUT_C evaluation; evaluation attempt3.
STOP for owner review. Stage B compatibility has not been amended.

V5 STAGE A TRAINING REMAINS COMPLETE AND UNMODIFIED.
V5 FINAL B0 MEASUREMENT PUBLISHED - OWNER REVIEW REQUIRED.
V5 STAGE B NOT RUN.
V5 READER PILOT NOT RUN.
LEARNED QUERY CONTROLLER NOT TRAINED.
MULTI-SEED REPLICATION NOT RUN.
V5_HP_CONFIRM_V2 REMAINS SEALED AND UNEVALUATED.
V4_TUNE_V1 AND HOLDOUT_C REMAIN UNEVALUATED.
