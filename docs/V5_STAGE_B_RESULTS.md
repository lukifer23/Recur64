# V5 Stage B: training complete; DEV evaluation blocked

## Engineering

Starting HEAD: `e40521fe661c87bf8a8e1f832200803bd1d50cae`.
Final scientific source: `3db24a926815159d592e93b60a8ae51852abad13`.
The preregistered `v5_stage_b_predecessor_bridge_v1` changes only exact CLI
predecessor/publication validation and focused tests. Model math, architecture,
data, loss, optimizer, sampler and acquisition are unchanged: 7,162,896 parameters,
FP32, configuration `849133a5cdf169f187778bace2f858aa4747d2e8defef3bb5ac1bffc839774ee`.

Fresh CPU and RTX 2050 CUDA qualification PASS, including unchanged exact D9,
all nine Q/R shapes, zero null error, 50 resident updates and checkpoint/moment
continuation. Graph provenance PASS. Full release workspace: 601 passed, zero
failed, two ignored. Affected formatting and V5 all-target Clippy PASS; unrelated
historical formatting was untouched. Fresh disposable Q8/R4 drill PASS:
2.9871070881684623 to 0.0977521538734436 mean loss, 96.7275% reduction, baseline
exact. Its weights were never reused.

See [validation](evidence/v5/stage-b-validation.json),
[lineage receipt](evidence/v5/stage-b-lineage-receipt.json) and
[source scope](evidence/v5/stage-b-source-scope.json).

## Training

Seed 5301 completed the frozen 800 updates in one bounded CUDA invocation:
1140.8644862 seconds, native exit zero, no resumes. Training execution is VALID.
Recipe remains `v5_stage_recipe_v3`; complete Stage B recipe digest:
`55533c9a9d9125fd166542667d619b1d40b95119b5129d9a1c63072e296018fb`.
Run: `runs/v5/v2/seed-5301/stage-b`.

Update-0 model SHA256:
`2d1c770a43a6455148b774e9ddb552b6ca33efd9fdd5d37593cefe7c8ae0bb00`.
Update-800 model SHA256:
`c7f6b10a0b60982199cd352c157a377115c6f00a46c699a6a9b3bb859e8bd273`.
Final optimizer SHA256:
`bb4138e1dc6ff83dcdc065dbd768b403b6e7f0f995d28968b56317c4862fe929`.

All 800 losses were finite: first 0.6637426661327481, first-50 mean
0.6186572668612643, last-50 mean 0.5513546268940809, final
0.820938692195341; diagnostic minimum 0.08405776543077081 and maximum
1.9292012327932753. Final LR zero. These are sampled TRAIN losses, not held-out
performance. Exactly 28,800 examples, two per condition per update.

| TRAIN cell | Exposure |
|---|---:|
| KQQvK M1 | 3204 |
| KQQvK M2 | 3204 |
| KQQvK M3 | 3204 |
| KQRvK M1 | 3204 |
| KQRvK M2 | 3204 |
| KQRvK M3 | 3204 |
| KRRvK M1 | 3204 |
| KRRvK M2 | 3186 |
| KRRvK M3 | 3186 |

Each of 18 condition samplers exposed 1600 examples, balanced at 177/178 per
cell. All 81 checkpoint generations retain the exact 84 baseline parameter
tensors; all model and final optimizer values are finite. Optimizer state contains
91 reader parameter IDs with counters 800 and zero baseline parameter IDs.
Baseline fingerprint before/after:
`12b272a941e5b29589195a65c779a60509626d2108975e4793674ad5867d75c9`.
The complete baseline parameter digest, computed read-only from checkpoints,
is also identical for Stage A, update 0 and update 800.

See [training integrity](evidence/v5/stage-b-training-summary.json),
[optimizer integrity](evidence/v5/stage-b-optimizer-integrity.json) and
[baseline parameters](evidence/v5/stage-b-baseline-parameters.json).

## DEV measurement

DEV was first used only after update 800 completed and passed integrity checks.
Three update-0 random-reader control cells completed: KQRvK M1/M2/M3, 750 each,
2250 unique positions. Normal replay parity PASS. Their 4500 B0 schedule records
agree exactly with the published Stage A report on all common policy metrics
and action indices. This is a partial cross-check, not a complete all-DEV result.

The fourth process, update-0 KRRvK M1, exited 1 after 111.594 seconds:

```text
Error: cannot derange V5_HP_DEV_V2-KRRvK-m1-.......................................K.R.............R....k... UniformFrontierV1 depth 6 within family/depth cell
```

The existing `shuffled_graphs` boundary in `study.rs:393` requires a donor from
a different root in the same family/root-mate-depth cell at the same acquired-node
depth. This recipient has zero eligible other-root donors at node depth 6.
Root mate depth M1 and acquired-node depth 6 are different quantities. The failure
does not establish a label error. No failed-cell report serialized and no metrics
were recovered from partial state. The original log and native receipt are preserved.

All further measurements stopped. No retry, changed donor pool, changed seed,
omitted position/control or scientific source change was made.

The [partial update-0 summary](evidence/v5/stage-b-update-zero-partial-summary.json)
contains every measured cell/schedule/Q/R/control policy row. For the primary
KQRvK M3 random-reader control, Q8/R1 and Q8/R4 both have top1 432/750 (0.576)
under both schedules, with zero action changes versus B0. Mean set losses:

| Schedule | Q8/R1 | Q8/R4 |
|---|---:|---:|
| uniform_frontier_v1 | 1.2727858525 | 1.2727949426 |
| base_ranked_depth_v1 | 1.2727835980 | 1.2727922221 |

These are untrained-reader controls. No update-800 DEV result exists. Neither
six-cell merge exists. The trained KQRvK M3 and KRRvK M3 Q/R matrices are NOT RUN.
See [failure evidence](evidence/v5/stage-b-evaluation-failure.json).

## Primary pilot gates

All five scientific contrasts/CIs and the final classifier invocation are NOT RUN:
the required complete update-800 evaluation is unavailable. Engineering
qualification passed, but complete evaluation integrity/accounting is unavailable.
No classifier label is assigned, including ENGINEERING_FAILURE or NO_SIGNAL.
No threshold or bootstrap contract changed.

## Mechanism diagnostics

Partial update-0 serialized controls/composition records remain preserved as
random-reader evidence. Final trained payload/null/shuffle, feedback/relation-bias,
composition, loop-state and action-transition analyses are NOT RUN. They cannot
be inferred from TRAIN loss or update-0 controls.

## Conditional R8

NOT RUN: PILOT_CANDIDATE was not established.

## Interpretation and next authorization

Stage B training is complete and valid; reader task benefit remains unmeasured.
The observed blocker is the evaluation shuffle donor eligibility contract.
The owner's final-source freeze and operational stop prohibit repairing/retrying
it within this experiment. A future authorization must prospectively specify
the control-contract treatment of empty donor pools, preserve both completed
checkpoints, and qualify any new evaluator/explicit predecessor bridge before
resuming measurement. No repair is implemented or scientific rescue proposed here.

Post-run actual TRAIN/DEV/CONFIRM custody PASS, all pairwise exact-FEN and canonical
overlaps zero. CONFIRM sealed=true, evaluated=false; no model invoked against it.
All 608 preserved files verified byte-for-byte, including the complete Stage A
file set, published B0 and all Stage B checkpoints. Historical failures remain.
Unavailable workstation-only raw-data cross-disjointness remains NOT VERIFIED.

See [post custody](evidence/v5/stage-b-post-custody.json),
[preservation](evidence/v5/stage-b-preservation.json) and
[compact run summary](evidence/v5/stage-b-seed5301-summary.json).

## Not run

Remaining update-0 KRRvK cells; every update-800 DEV cell; six-cell merges;
final ablation/composition analyses; pilot gates/classification; R8; Stage A
retraining; extra Stage B updates; LR tuning; intermediate DEV evaluation;
checkpoint selection; seeds 5302/5303; CONFIRM evaluation; query controller;
self-play; V4_TUNE; HOLDOUT_C. No architecture or dataset rescue.

V5 STAGE B TRAINING COMPLETE — PILOT CLASSIFICATION NOT RUN; EVALUATION BLOCKED.

V5_HP_CONFIRM_V2 REMAINS SEALED AND UNEVALUATED.
MULTI-SEED REPLICATION NOT RUN.
LEARNED QUERY CONTROLLER NOT TRAINED.
SELF-PLAY NOT RUN.
V4_TUNE_V1 AND HOLDOUT_C REMAIN UNEVALUATED.
