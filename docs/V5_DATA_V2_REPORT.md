# V5 native data V2 completion and final pre-training gate

Starting HEAD: `0930ac6895ff27e24b37c037ce66512754fc8880`.
Scientific consumer source: `d11659eca0774e0064bed0ef64ead2b725886d93`.
Dataset producer source: `738db983084a998664ce962f87fa3c4a6153f526`.
Model implementation source: `003d296e094c28fc488cd56ef0944b60299983f9`.

Model math, architecture and returned-mask repair are unchanged. Parameters: **7,162,896**. FP32, physical microbatch **2**.
Configuration digest: `849133a5cdf169f187778bace2f858aa4747d2e8defef3bb5ac1bffc839774ee`.

## Measured datasets

Family `v5_hp_heavy_endgames_v2`; selection `v5_hp_dataset_select_v1`; audit `v5_hp_independent_audit_v1`. Generated TRAIN first, then DEV excluding TRAIN, then CONFIRM excluding both.

| Identity | Seed | Records | Scientific record-content digest |
|---|---|---:|---|
| V5_HP_TRAIN_V2 | 0x7A502001 | 27000 | `7ca17f824d24a629d8b54e0a2be48590f6cf78e2bd42d6c0d5db62ddeadf2867` |
| V5_HP_DEV_V2 | 0x7A502002 | 4500 | `aac1be9d74af801ea2fb32e88a1f3eb9e4e93062f6a49ada8512b953c91847c5` |
| V5_HP_CONFIRM_V2 | 0x7A502003 | 4500 | `6a4e355fdb38d46cefb8b25974938432dd2979b893c788a537ada9e1b60aa4f0` |

| Cell | TRAIN | DEV | CONFIRM |
|---|---:|---:|---:|
| KQQvK M1 | 3000 | - | - |
| KQQvK M2 | 3000 | - | - |
| KQQvK M3 | 3000 | - | - |
| KQRvK M1 | 3000 | 750 | 750 |
| KQRvK M2 | 3000 | 750 | 750 |
| KQRvK M3 | 3000 | 750 | 750 |
| KRRvK M1 | 3000 | 750 | 750 |
| KRRvK M2 | 3000 | 750 | 750 |
| KRRvK M3 | 3000 | 750 | 750 |

| Identity | Exact-FEN-set digest | Canonical-set digest |
|---|---|---|
| V5_HP_TRAIN_V2 | `b7ebc90955fd8959c8e7266d1afac7a913c94959f3bca7a353b80e9cbc30f61a` | `6170bee242a7af46d4b3959f34d52a01e189c3a6b7d06f85925673e218c5f987` |
| V5_HP_DEV_V2 | `4c2f3bb3bd2b594ee0921974a151ad71a19e149457b59c9061977fffd96f0e06` | `1201ea0c6dc8789b1d9cf0212bfb088590b4473948e28355c9ecd0523d4e37ec` |
| V5_HP_CONFIRM_V2 | `6f3cb6f27932dee45c328cc17531bd99d0bd6d98063c18d8e1ab30bf1fad084c` | `2860b55ad1c581f90691a52907fed779bd68a6f820bda83a264f73cfbbe37be4` |

Every accepted record independently audited: TRAIN27000 / DEV4500 / CONFIRM4500, **zero failures**. Full re-selection, authoritative re-labeling and independent re-audit of all three splits reproduced bit-identical raw files and scientific digests. Verified exhaustive pools were reused; an independent second pool census is not claimed.

Pairwise exact-FEN / canonical intersections: TRAIN-DEV **0/0**, TRAIN-CONFIRM **0/0**, DEV-CONFIRM **0/0**. No within-split duplicates. Cross-disjointness from unavailable workstation-only raw datasets was NOT verified.

CONFIRM: generated=true, audited=true, sealed=true, evaluated=false. Seal: [confirm-seal](evidence/v5/data/v2/confirm-seal.json). Raw files exist under ignored runs/v5/data/v2/v5-hp-{train,dev,confirm}-v2.json and actual bytes passed custody.

## Scientific integration

Data contract `v5_hp_data_v2`; recipe `v5_stage_recipe_v3`. Nine TRAIN cells feed the unchanged cell_balanced_v1 sampler. Stage A1200/batch64/LR3e-4/warmup80/Q0; Stage B800/batch36/same LR and warmup,18 conditions, two examples per condition. Both remain unrun. Each future update records per-cell and per-condition exposure. No TRAIN-to-DEV partition exists. DEV4500, primary KQRvK M3 n750; practical thresholds and bootstrap unchanged.

Frozen recipe contract digest: `4485032edfc1de9a106d66009ccacb33d20e593d7408e9cef218790a5cbba9de`.
Complete Stage A recipe digest: `6642579e1f2472bda955ca7ada5bb3b8a435634c023b665684da1e4676347e70`.
Future Stage B initialization/model/baseline hashes remain unbound until a separately authorized Stage A exists.

Actual production boundaries: TRAIN accepts only TRAIN-V2; DEV accepts only DEV-V2; cross-role loads refuse; ordinary code refuses CONFIRM. Custody verifies sealed CONFIRM without a model. Changed raw bytes and source/config/content/FEN/canonical/count/cell/audit manifest mutations refuse. [Actual boundary evidence](evidence/v5/actual-boundaries-d11659e.json).

## Current-source engineering qualification

Full release workspace: **596 passed, zero failed, two preserved ignores**; all focused V5 model/data contracts included. V5 CUDA all-target Clippy -D warnings and serial pinned CUDA release build PASS. Affected-file formatting PASS. Workspace-wide formatting reports16 independently verified unchanged historical files; no unrelated reformatting.

Fresh CPU and RTX2050 CUDA qualifications **PASS**, FP32/microbatch2, unchanged exact D9, all nine Q/R shapes,50 resident Q8/R4 updates, complete checkpoint/model/moment restore and exact continuation. CUDA normal/normal, profile/profile, normal/profile and reverse order: **12 independent comparisons ALL_EXACT**, clone purity PASS. No numerical divergence or new execution contract required. Historical failed reports, including d970049 CUDA D9 FAIL, remain byte-identical.

Fresh standalone graph provenance **PASS**: actual valid artifact accepted; stale source, wrong configuration, graph tamper, copied metadata and naked graph refused. Scientific graph semantics/configuration unchanged. [Current-source validation](evidence/v5/validation-d11659e.json).

All three local custody checks PASS; confirmation seal/evaluated=false and all pairwise disjointness PASS before drill. [Final custody](evidence/v5/final-custody-d11659e.json).

## Disposable engineering drill

**PASS** at Q8/R4, FP32/microbatch2, seed5301, 200 updates, peak LR1e-3/warmup20. Exactly24 stable-hash TRAIN positions, four per KQR/KRR x M1/M2/M3 cell;48 schedule-specific graphs. Mean correct-set loss **2.987107088 -> 0.026624600**, **99.1087% reduction**. Finite training and exact baseline fingerprint PASS.40 of48 schedule-specific predictions changed. Wall 530.483s. Weights were disposable and are not reused. [Drill evidence](evidence/v5/drill-d11659e-q8.json). Q16 NOT RUN because Q8 passed. Post-drill actual custody/CONFIRM sealed/evaluated=false PASS. STOP before Stage A.

## Not run

Stage A; Stage B; DEV model evaluation/science; reader pilot/pilot classification; learned query controller; seeds5302/5303/multi-seed replication; self-play; LR screen; KQ/KR auxiliary training; V4_TUNE; HOLDOUT_C; CONFIRM model evaluation.

V5 STAGE A NOT RUN.
V5 STAGE B NOT RUN.
V5 READER PILOT NOT RUN.
LEARNED QUERY CONTROLLER NOT TRAINED.
MULTI-SEED REPLICATION NOT RUN.
V5_HP_CONFIRM_V2 REMAINS SEALED AND UNEVALUATED.
V4_TUNE_V1 AND HOLDOUT_C REMAIN UNEVALUATED.
