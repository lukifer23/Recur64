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
