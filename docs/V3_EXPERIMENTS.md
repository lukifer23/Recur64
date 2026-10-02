# Recur64 V3 — Experiment Ledger (append-only)

Labels: PRE-REGISTERED / MEASURED / INFERRED / NOT RUN. Never edit past entries; append
corrections as new entries.

## V3-E0 — branch creation and P0 specification

- **Date:** 2026-09-30
- **Status:** MEASURED (repository facts); no science.
- **Commands:**
  - `git fetch origin`
  - `git checkout -b experiment/workstation-v3-active-search feb86236f24eeaca2a1dc16f7c9e45bca4dc51de`
- **Verified refs:** `main` = `fef1ffcf…`; `origin/experiment/hp-r15` = `3430e6ca…`;
  `origin/experiment/hp-r15-h3-integration` = `79ffd142…`; base `feb86236…` (clean tree at start).
- **Read-only findings recorded:** V2.5 headline numbers (CF .659 M3, L .631, C0 .604) and
  HOLDOUT_C digest match the committed evidence; HOLDOUT_C bytes exist locally at
  `runs/v25/p25/holdouts/proof-holdout_c.json` (git-ignored); exposure log lists A and B only.
  Digest is a content digest (`ProofTargets::compute_digest`), not a raw-file SHA.
- **Decisions:** V3-D1 … V3-D3 (see `docs/DECISIONS.md`).
- **Gate:** n/a (specification). Gates I–VI pre-registered in `V3_RESEARCH_PLAN.md`; ⏳ items
  are frozen at P4/P6 before CONFIRM.
- **NOT RUN:** everything else (P1–P11). HOLDOUT_C not evaluated, not loaded, not hashed.

## V3-E1 - P0 review addendum

- **Date:** 2026-09-30
- **Status:** PRE-REGISTERED (rules) / MEASURED (tests). Commit `e61c2e5`.
- **Changes:** one meaning of "frozen"; deterministic P4 feasibility rule
  (`C_8(KQRvK M3) >= 0.25` on P25_DATA_V1 TRAIN, else NOT SCIENTIFICALLY QUALIFIED / BUDGET
  MIS-SPECIFIED); set-valued ProofTraceV1 (uniform over admissible set `A(S)`); primary FIXED
  comparator `fixed_bfs_actionid_v1` frozen before any active result; ALL-INFO scoped as a
  separately trained information-sufficiency control; `semantic_id` replaces `state_hash`.
  Decisions V3-D3 (amended), V3-D4..D7.
- **Deviation recorded:** P1 (`bb6a5c0`) was committed before the addendum arrived; the addendum
  amends P1 (field rename and semantic identity) in a follow-up commit. No science was run.
- **NOT RUN:** the feasibility measurement (P4).

## V3-E2 - P1 StateQueryV1

- **Date:** 2026-09-30
- **Status:** MEASURED. Commits `bb6a5c0` (crate), `e61c2e5` (semantic identity).
- **Commands:**
  - `cargo test -p recur64-statequery --release`
  - `cargo fmt --all -- --check`
  - `cargo clippy -p recur64-statequery --all-targets`
- **Result:** fmt clean, clippy 0 warnings on the crate; 27 tests pass (17 query/identity, 5
  differential, 3 whitelist, 2 dependency-boundary). Differential: 203,426 random-descent edges,
  105,670 fixture edges (castling, en passant, promotion, stalemate, fifty-move, mate) and a
  start-position depth-3 BFS, every packet equal to the reference child (observation, legal list,
  terminal reason, check, repetition, clock, semantic id, perspective piece counts); the
  reference path decodes ActionIds itself and calls `GameState::apply`.
- **Finding:** `GameState::legal_actions()` returns moves at threefold/fifty-move terminal states
  (its doc says empty for terminal). StateQueryV1 forces the empty frontier itself and a test
  covers it.
- **Gate:** query-tool correctness gate PASSED for the tested surface. The optional `shakmaty`
  oracle was not used in this crate (GPL, core dev/test only); legality and mate parity with it is
  covered by `recur64-core` tests, which are unchanged and not re-run here.
- **NOT RUN:** workspace-wide release tests after the P2 config change (pending).

## V3-E3 - P1.1 cleanup of StateQueryV1

- **Date:** 2026-09-30
- **Status:** MEASURED (tests) / PRE-REGISTERED (rule edits). Decision V3-D8.
- **Changes:**
  - `content_digest()` replaced by `state_digest()` over every state-content field, plus a
    separate persistent `QueryIdentity` (parent semantic id, incoming ActionId, ply from root,
    child state digest). `node_id` / `parent_id` are ephemeral and in no persistent digest.
  - `query()` no longer regenerates the parent's legal moves: membership in the stored legal set,
    direct ActionId decode, one authoritative `GameState::apply`, one child legal generation.
    `legal_generations() == 1 + successful_queries()`.
  - The nonexistent tool depth refusal is removed from the architecture text; the tool has no
    depth limit and the model enforces its own representable depth (`ACTIVE_MAX_DEPTH`).
  - The owner-adjustment sentence for `C_8 >= 0.25` is removed; Gate I `0.75` is a reported
    diagnostic only; the job-length guard is stated exactly (2 hours wall-clock).
- **Commands:** `cargo test -p recur64-statequery --release`; `cargo fmt --all -- --check`;
  `cargo clippy -p recur64-statequery --all-targets`.
- **Result:** fmt clean; clippy 0 warnings on the crate; 35 tests pass (8 digest/identity/counter,
  17 query/identity, 5 differential, 3 whitelist, 2 dependency-boundary). The independent
  differential reference (which decodes ActionIds itself) still matches every edge through the new
  query path: 203,426 random-descent edges, 105,670 fixture edges, start-position depth-3 BFS.
  Mutation tests: every declared state-content field changes `state_digest`; every query identity
  component changes its digest; ephemeral fields change neither; every packet field is in exactly
  one class. Refused queries leave all counters unchanged.
- **Preserved unchanged:** crate boundary, one edge = one query, authoritative GameState per node,
  root-only CandidateFacts, prohibited fields, terminal nodes exposing no frontier, semantic
  history identity, visible refusals, sealed HOLDOUT_C, set-valued ProofTraceV1,
  `fixed_bfs_actionid_v1`, ALL-INFO interpretation, the B0/B2/B4/B8/B16 design.
- **NOT RUN:** workspace-wide release tests (P2 model work is uncommitted in the working tree).

## V3-E4 - P2 active_search_v3 model skeleton (CPU)

- **Date:** 2026-09-30
- **Status:** MEASURED (engineering). Detail in `docs/V3_BUILD_RESULTS.md`.
- **Commands:**
  - `cargo test -p recur64-model --release --test active_v3`
  - `cargo test --workspace --release`
  - `cargo fmt --all -- --check`
  - `cargo clippy --workspace --all-targets`
- **Result:** 17 new engineering tests pass; full workspace release suite exit 0 (historical V2.5 and
  mainline tests and pinned hashes unchanged); fmt and clippy clean. Total unique parameters
  30,853,790. Root encoder runs once at every budget (measured by in-function counters), query
  encoder and planner run once per round, counts equal successful queries, parameter count is
  budget independent, every non-STOP parameter gets a finite non-zero gradient on update one,
  4x4 architecture cross-refusal and all 11 V3 contract ids enforced, resume matches an
  uninterrupted run (max abs diff < 1e-4 in log-probs).
- **Findings:** inert planner key bias found by the gradient test and removed; inherited inert key
  biases in the V2.5 `Block`/`CandidateBlock` recorded and deliberately not changed (see build results).
- **Gate:** P2 CPU correctness gate PASSED for the tested surface.
- **NOT RUN:** CUDA, compute and VRAM measurement (P3); all science.


## V3-E5 - P2.1 hardening after the P2 review

- **Date:** 2026-09-30
- **Status:** MEASURED (engineering). Decision V3-D9; detail in `docs/V3_BUILD_RESULTS.md`.
- **Commands:**
  - `cargo test -p recur64-model --release --test active_v3` (21 tests)
  - `cargo test -p recur64-cli --release --test active_boundary` (5 tests)
  - `cargo test --workspace --release`; `cargo fmt --all -- --check`; `cargo clippy --workspace --all-targets`
- **Result:** full workspace release suite exit 0, 372 passed, 0 failed; fmt clean; clippy 0 warnings.
  The real `recur64` binary refuses `active_search_v3` for 18 historical commands (before any output)
  and `model-info` reports the real active graph (30,853,790 parameters, 123,415,160 fp32 bytes, 11
  contracts, budgets B0/2/4/8/16, recurrence 1). CPU resume is bit-exact (0e0 over every parameter,
  restore and two further optimizer steps). Mixed-batch compaction executes exactly the rows that
  queried (9, not 16). A tamper test rejects 12 accounting corruptions; invariants run before every
  `run()` returns. Terminal roots (checkmate, stalemate, threefold, fifty-move) are refused with zero
  CandidateFacts and zero neural work. `ACTIVE_MAX_BUDGET = 16`; an engineering-stress mode is marked
  engineering only. `query_heads = 4` frozen. Tree depth consistency is enforced.
- **Deviations / incidents:** the gradient-coverage test was flaky because of the inherited inert
  key biases; fixed by exempting them by name with a noise bound (10 consecutive clean loops). A
  stale test FEN (adjacent kings) revealed that `from_fen` accepts illegal positions; recorded, not
  changed.
- **Gate:** P2.1 hardening gate PASSED for the tested surface.

## V3-E6 - P3 CUDA and system qualification (real FP32 CUDA)

- **Date:** 2026-09-30
- **Status:** MEASURED (engineering). Evidence `docs/evidence/v3/v3-qual-cuda-fp32.json`,
  `docs/evidence/v3/model-info-active-search-v3.json`. No science.
- **Command:** `recur64 v3-qual --config configs/v3/active-search-v3-cuda.toml --output runs/v3/cuda-qual
  --positions 64 --budgets 0,2,4,8,16 --batches 1,8,16 --reps 5 --train-budgets 2,4,8 --train-batch 8
  --train-updates 3 --lifecycle-reps 12 --sustained-seconds 10` (binary built with `--features cuda`,
  CUDA 12.9.1 user-space runtime, RTX 2000 Ada; seed 20250930; positions generated in the tool).
- **Result:** `all_sections_ok = true`. Device known-answer guard passed. Finite forward and backward at
  B0/2/4/8/16 (ACTIVE and FIXED); root encoder once at every budget (in-function counters equal the
  accountant); forced budget exact; gradient coverage 234 tensors, all covered except the masked STOP
  head; teacher-forced AdamW updates 0.18 to 0.27 s each at batch 8; checkpoint save/load max abs policy
  difference 0.0; resident-model VRAM plateau (150 inference calls flat at 2001 MB, 40 training updates
  flat at 1489 MB). Measured wall B16 over B0: 16.1x at batch 1, 5.6x at batch 8, 7.8x at batch 16; exact
  CPU query at most 2.4% of wall; planner + selector 62 to 67% of wall at B16, about 6 to 7 ms per round
  regardless of batch; sustained utilization mean 39%, max 47% (structural, see results).
- **Findings:** first-seen dynamic widths cost a one-off stall up to about 0.4 s (first pass 1.22x the
  second); repeated model build/drop grows VRAM about 13 to 16 MB per cycle, reproduced on the V2.5 model
  (pre-existing; not fixed; not present inside a single-model run). Suggestions for the contract are in
  `docs/V3_BUILD_RESULTS.md` (none changes a frozen rule).
- **Gate:** P3 CUDA/system qualification gate PASSED for the tested surface.
- **NOT RUN:** HOLDOUT_C, P4 feasibility measurement, ProofTraceV1, TUNE generation or training, P5 to P11,
  BF16/TF32, fusion/graph capture/shape bucketing.

## V3-E7 - Commit index for P0 to P3

- **Date:** 2026-09-30
- **Status:** RECORD (no new measurement).

| Entry | Commit | Content |
|---|---|---|
| V3-E0 | `2cf14b3` | P0 lineage, architecture, research plan, ledger |
| V3-E2 | `bb6a5c0` | P1 StateQueryV1 crate |
| V3-E1 | `e61c2e5` | P0 review addendum (frozen meaning, feasibility rule, set-valued traces, semantic identity) |
| - | `fffdec4` | ledger entries E1 and E2 |
| V3-E3 | `3072caf` | P1.1 digest contract, one-generation query path, no tool depth cap |
| V3-E4 | `8b709a5` | P2 model skeleton and CPU correctness |
| V3-E5, E6 | `81f9c61` | P2.1 hardening and P3 CUDA qualification, docs and evidence |

Base `feb86236f24eeaca2a1dc16f7c9e45bca4dc51de` (`experiment/workstation-v25`). Main
(`fef1ffcf9c38381d4adc671e5e2c5ead9f141e33`) and the V2.5 branch are unchanged.


## V3-E8 - P3.1 qualification-harness hardening

- **Date:** 2026-10-01
- **Status:** MEASURED (new run) / DERIVED (reclassification of existing evidence). Decision V3-D10.
- **What was wrong (found in review):** the original `all_sections_ok` did not aggregate every
  condition the prose called a qualification condition (FIXED rows, the gradient-coverage list, STOP
  gradient, non-finite gradient). The original P3 evidence itself shows those conditions held.
- **Code, before any new measurement:** `v3_verdict::evaluate` (pure function), 15 unit tests proving
  that each of ACTIVE row, FIXED row, accounting invariant, non-finite loss, missing non-STOP
  gradient, non-finite gradient, non-zero STOP gradient, checkpoint, resident inference VRAM and
  resident training VRAM independently flips the qualification to failure (and that nothing else
  flips with it), and that a repeated-build `plateau = false`, a huge dynamic-shape outlier and very
  low utilization do NOT fail it; an empty report fails closed. 3 CLI tests prove the verdict
  command exits non-zero on a failing report while still writing its summary, and leaves the source
  report byte-identical.
- **DERIVED (existing evidence):** `docs/evidence/v3/p3-qualification-summary-v2.json` applies the
  hardened verdict to the unmodified `v3-qual-cuda-fp32.json`: `qualification_gates_ok = true`, all 11
  gates pass (ACTIVE and FIXED 15 cells each, training coverage 234 tensors with none missing, STOP
  gradient exactly zero, no non-finite gradient, checkpoint exact, resident inference and training
  plateaus). One limitation is recorded rather than hidden: the original report did not record
  FIXED-output finiteness, only structural success of the FIXED runs. The repeated build/drop slope is
  recorded as a non-gating finding. The original run did not use this logic; it is not claimed to have.
- **MEASURED (new run):** `recur64 v3-qual ... --lifecycle-reps 12 --sustained-seconds 10` with the
  hardened harness (same command shape as V3-E6), exit status 0, 2 m 20 s:
  `docs/evidence/v3/v3-qual-cuda-fp32-p31-hardened.json` and
  `docs/evidence/v3/p31-qualification-summary-hardened-run.json`. All 11 gates pass, FIXED finiteness
  is now recorded (no limitations), `diagnostics_complete = true`. Diagnostics: build/drop slope
  12.8 to 41 MiB per cycle in some modes (non-gating, as in V3-E6), first-seen shape worst case 5.4x
  the second-pass median, 205 positions/s sustained at 40% mean utilization (non-gating). These are
  new measurements of the same envelope; V3-E6's numbers are unchanged and not superseded.
- **Incident:** the first hardened run overflowed the main-thread stack (exit 127 from the shell,
  "thread main has overflowed its stack"); it produced no report and is not used. Cause and fix in
  V3-D10.
- **Gate:** P3.1 hardened qualification gate PASSED. Tests: fmt clean, clippy 0 warnings, active-boundary,
  v3-verdict, v3-qual CLI, StateQuery and active V3 model tests all pass.
- **NOT RUN:** P4 and later.


## V3-E9 - P4 pre-registration and custody

- **Date:** 2026-10-01
- **Status:** PRE-REGISTERED (rules, committed before any TRAIN trace exists) / MEASURED (custody reads).
  Decisions V3-D11 to V3-D14; specification `docs/V3_P4_PLAN.md`.
- **Custody (read-only, `recur64 v3-p4 custody`):** P25_DATA_V1 TRAIN loaded through
  `ProofTargets::load`: 44,332 positions, digest `3b25dc8549dd2fc9d47c30e294c273b3306aecb3eba91b964715326ddf74f2e6`
  read from the file, every position passes the existing independent label audit, 44,332 unique FENs
  and canonical classes, zero overlap with HOLDOUT_C. HOLDOUT_C digest equals the frozen
  `4ab951c6edd8dd4f531bb87d2f4373895d1fddb70efdf052b24c09a1b71d87d5`; `evaluated = false`; no exposure
  is recorded for the integrity check. Exclusion inventory: 11 datasets, 70,624 canonical classes.
  HP/X1/X2 datasets are not available locally; cross-line disjointness is not claimed.
- **Code committed with this entry, before any TRAIN trace:** ProofTraceV1 generator, `Q*` and `A(S)`,
  independent trace audit, sharded resumable store with strict validation, the feasibility table and its
  exact-integer rule, the V3_TUNE_V1 generator, the custody guards, and 18 tests (DP versus brute-force
  enumeration on random queried sets, live order-invariance through `QueryManager` and `Tree`, ten audit
  corruptions, thread-count determinism, resume and corruption refusal, threshold boundaries,
  HOLDOUT_C guards, answer-free packet).
- **NOT RUN:** TRAIN trace generation, the audit of TRAIN traces, `C_k`, the feasibility classification,
  `v3_tune_v1` generation, TUNE traces. P5 is not authorized.


## V3-E10 - P4 data, ProofTraceV1 and the frozen feasibility measurement

- **Date:** 2026-10-01
- **Status:** MEASURED (results) under rules PRE-REGISTERED in V3-E9 and `docs/V3_P4_PLAN.md` (commit `156cc0f`).
  Detail: `docs/V3_P4_RESULTS.md`; evidence: `docs/evidence/v3/v3-p4-*.json`, `v3-tune-v1-manifest.json`,
  `v3-confirm-seal.json`.
- **Commands (binary `target/release/recur64`):**
  - `v3-p4 custody --train runs/v25/p25/data/proof-train.json --holdout-c runs/v25/p25/holdouts/proof-holdout_c.json --inventory-dirs runs/v25/proof,runs/v25/proof-v2,runs/v25/p25/data,runs/v25/p25/holdouts --threads 20`
  - `v3-p4 tune-gen --inventory-dirs <same> --data-dir runs/v3/data --manifest runs/v3/p4/v3-tune-v1-manifest.json --threads 20`
  - `v3-p4 trace-gen --data runs/v25/p25/data/proof-train.json --dir runs/v3/p4/trace-train --threads 20`
  - `v3-p4 trace-audit --data runs/v25/p25/data/proof-train.json --dir runs/v3/p4/trace-train --threads 20`
  - `v3-p4 determinism --data <train> --dir <trace dir> --shards 0,30,60,88 --threads 20`
  - `v3-p4 feasibility --data runs/v25/p25/data/proof-train.json --dir runs/v3/p4/trace-train --role primary`
  - the same trace, audit, determinism and (diagnostic role) feasibility commands for `v3_tune_v1`.
- **Data digests:** P25_DATA_V1 TRAIN `3b25dc8549dd2fc9d47c30e294c273b3306aecb3eba91b964715326ddf74f2e6` (44,332
  positions); `v3_tune_v1` `c66018657009c9c5eade58369b5f451466d6662c910f76b8aacddcac99921b53` (4,500); HOLDOUT_C
  `4ab951c6edd8dd4f531bb87d2f4373895d1fddb70efdf052b24c09a1b71d87d5` (verified, `evaluated = false`); exclusion
  manifest `b3d5a7eeab183be9b501aea503ef2511857ffe2e39d5d407c2f1092743314cb7` (70,624 canonical classes).
- **Trace digests:** TRAIN trace manifest `8160734ed5e3a145c12dd72d8e9dc49cc893984fc1cf714ea93eba904f58c488`, audit
  manifest `895636669bbc149a89c0de7e1512606f5c674c5d626309f7260c9144401003b0` (0 failures of 44,332).
- **Result:** `C_8(KQRvK M3) = 2168 / 5000 = 0.4336 >= 0.25`. Classification:
  **SCIENTIFICALLY QUALIFIED FOR THE B8 PRIMARY EXPERIMENT** (V3-D15). Full table in the results document.
- **Seeds:** `v3_tune_v1` `0x7A130004` (2048065540); no other randomness in the measurement (the traces are exact
  and deterministic).
- **Incidents:** the first `trace-gen` was cut short by a `| head` pipe after five shards; resumed with the five
  shards re-validated and reused. A mate-in-three audit-corruption test was added after the measurement
  (test-only). The measurement was taken once.
- **Gate:** P4 gate: data custody, `v3_tune_v1`, `proof_trace_v1` implemented and audited, frozen feasibility
  measurement taken and classified. PASSED as specified.
- **NOT RUN:** P5 and everything after it; any model evaluation; HOLDOUT_C evaluation, tracing or analysis.
  **P5 NOT RUN - awaiting owner review and approval.**


## V3-E11 - P4.1 and P5 plan: training interface and bounded LR screen (pre-measurement)

- **Date:** 2026-10-01
- **Status:** PRE-REGISTERED. Plan: `docs/V3_P5_PLAN.md`; decisions V3-D16 to V3-D19. No LR-screen result on
  `v3_tune_v1` exists at this entry.
- **Built (tested on CPU with real positions, traces and the real model graph):** `crates/recur64-runtime/src/p5/`
  (recipe and run digest, verified data loaders and per-budget samplers, `proof_teacher_seeded_v1` with the
  completion latch, whole-update loss normalisation, the resumable trainer with a strict sidecar, the TUNE
  evaluator and offline selector diagnostics) and `recur64 v3-p5 {recipe,preflight,train,select}`.
- **NOT RUN:** the CUDA preflight, any screening run, any TUNE evaluation of a trained model, P6.


## V3-E12 - P5 resolved layout (TRAIN-only CUDA preflight, before any screen result)

- **Date:** 2026-10-01
- **Status:** MEASURED (engineering only). Commit B; no `v3_tune_v1` result on a trained model exists.
- **Command:** `v3-p5 preflight --train runs/v25/p25/data/proof-train.json --train-trace runs/v3/p4/trace-train --device cuda --layout 16x8 --updates 3 --output ...` (binary built with `--features cuda`, CUDA runtime 12.9.1 on `PATH`).
- **Result:** the full-geometry graph (30,853,790 parameters, micro16 x accum8, sequence [0,2,4,8,0,2,4,8], health checks on) ran on CUDA without OOM:
  peak VRAM 3,133 MB; update wall 20.8 s cold, then 2.2 s and 1.2 s (steady mean over the two warm updates 1.71 s);
  losses finite (total 7.28 to 7.33, grad norm about 2.9, TRAIN only). Projected single-run wall time 0.47 h
  (train 0.38 h, TUNE evaluation 0.07 h, checkpoints 0.02 h assumed); limit 2 h: within.
- **Resolved layout:** **micro16 x accum8** (the 8x16 fallback was not needed and is not used).
  Contract digest `105ac3133877f954ed00e6ce9caaadabf6d5da1cf78a7ab99d44a195d03009d2` (`docs/evidence/v3/v3-p5-recipe.json`).
- **Not claimed:** nothing about learning. The three updates were at warmup learning rates and show no trend.
- **Incident:** the first attempt failed visibly because the CUDA runtime was not on `PATH` (the device known-answer
  guard refused to run); no substitution happened.
- **NOT RUN:** the six screening runs, TUNE evaluation of trained models, P6.


## V3-E13 - P5 screen interrupted (workstation move); resume instructions

- **Date:** 2026-10-01
- **Status:** INTERRUPTED, no screen result. The six-run screen was launched from the committed contract; run 1
  (LR 7.5e-5, seed 5101) reached update 150 (checkpoint saved) before being stopped by the owner. Runs 2 to 6 not started.
- **Resume:** `docs/V3_P5_RESUME.md` and `scripts/v3_p5_run_screen.sh`. Nothing pre-registered changed.
- **NOT RUN:** five of six runs, the update-800 evaluation of all six, the selection rule, P6.

## V3-E14 - P5 interrupted attempt voided; P5.1 integrity patch; screen restarts from update 0

- **Date:** 2026-10-01
- **Review:** the local artifacts of the interrupted run were inspected before anything changed. Highest valid state:
  update 150 (`state-1`, sampler draws 4,800 per budget, history length 150, recipe digest
  `adc844428c65486c0b0a37604ec99a7a8b761c229a01443b3ee7925bdd171b01`, checkpoint metadata step/update_counter/
  lr_schedule_step 150, seed 5101, lr 7.5e-5, structurally complete); `state-0` update 100. Update-0 TUNE
  `S_run = 3.5532696635894676`. Train-side losses stayed finite; a transient gradient-norm spike (about 121 and 145 at
  updates 110 and 120) recovered by update 130. These are TRAIN-side mechanism observations, not held-out evidence.
  Compact record: `docs/evidence/v3/v3-p5-interrupted-attempt1.json` (no weights).
- **Disposition:** VOID for LR selection. Reason: external operational interruption before the first trained TUNE
  evaluation; restarted from update 0 to preserve uninterrupted symmetry across all six P5 screen runs. This is not a
  failed learning run. Artifacts were moved (not deleted) to the git-ignored
  `runs/v3/p5/quarantine/interrupted-lr7.5e-5-seed5101-u150/`.
- **P5.1 integrity patch (reporting/resume only; the training function and the Recipe are untouched):**
  strict `Trainer::load` invariants (history length and labels, finite values, lr per the schedule, sampler draws equal
  to `updates_done x draws-per-update` derived from the layout, checkpoint metadata step/update_counter/lr_schedule_step/
  seed/peak lr/precision/backend/recurrence/deep supervision/architecture); a fresh model is refused in a non-empty run
  directory; the completed summary persists the final per-budget sampler exposure and run provenance (fresh vs resumed,
  start update, prior resumptions); evaluation files carry the run digest; `v3-p5 select` validates every summary
  (schema, lr, seed, digests, update counts, evaluation set, ACTIVE budgets, recomputed `S_run`, exposure, no
  HOLDOUT/CONFIRM reference, preregistered ineligibility class), refuses if either output exists, and performs the
  same-seed update-0 pairing integrity check before the rule (never used to choose an LR); the launcher stops on any
  non-zero exit; `.gitattributes` pins `*.sh` to LF.
- **Unchanged:** contract digest `105ac3133877f954ed00e6ce9caaadabf6d5da1cf78a7ab99d44a195d03009d2`, layout micro 16 x accum 8.
- **TESTED:** the new corruption, refusal, validation and pairing tests; the full workspace release suite; clippy and
  fmt clean; the CUDA release binary builds. CUDA resume is still not claimed bit-identical (CPU resume only).
- **NOT RUN:** the clean six-run screen, any trained TUNE evaluation, the selection rule, P6.

## V3-E15 - P5 clean six-run screen, frozen selection and results

- **Date:** 2026-10-01
- **Status:** COMPLETE. Six fresh uninterrupted runs (LR {7.5e-5, 1.5e-4, 3e-4} x seed {5101, 5102}), `exit 0`, 800 updates,
  TUNE evaluated at 0/200/400/600/800, exposure 25,600 examples per budget per run.
- **Result (frozen rule, applied once):** `S_lr` = 1.6964 (7.5e-5), 1.5570 (1.5e-4), 1.5098 (3e-4). **Selected peak LR 3.0e-4**
  (the largest candidate). Selected-recipe digest (without seed)
  `a069ba9d18befed65f970aca253b47780365fd7019be38283f270d79d6c1db33`. Contract digest unchanged (`105ac313...009d2`).
- **Integrity:** same-seed update-0 evaluations bitwise identical across LRs; the new seed-5101 update-0 equals the voided
  attempt's. A validator label defect (`"ACTIVE"` vs `"active"`) was found by the first `select` call, which refused before
  writing anything; it was fixed (reporting code only) and `select` re-run. See V3-E14 for the interruption record.
- **Diagnostics (TUNE, not gate results):** ACTIVE policy CE is lowest at B0 and higher at B2/B4/B8; teacher-forced top-1 about
  0.99 (confounded: the query pattern reveals the answer); ACTIVE proof completion falls below the ideal ceiling as the budget
  grows (B8: 0.54-0.55 vs 0.80; KQRvK M3 0.04-0.05 vs 0.43). Throughput: about 1.2 s/update, CPU-bound, GPU about 52% busy
  (not profiled; to be addressed before future long runs).
- **Evidence:** `docs/V3_P5_RESULTS.md`, `docs/evidence/v3/v3-p5-run-*.json`, `v3-p5-pairing-check.json`,
  `v3-p5-lr-selection.json`, `v3-p5-selected-recipe.json`.
- **NOT RUN:** P6, Gate I/II/III, B16, DAgger, HOLDOUT_C evaluation.

## V3-E16 - P5.2 diagnostic code and P6 ALL-INFO pre-registration

- **Date:** 2026-10-01
- **P5.2 code (committed before any new diagnostic value):** refined selector diagnostic splitting queries at proof
  completion (asserting no proof-admissible edge exists after completion), first-completion-step and completion-after-k
  distributions, per-position results, and `v3-p5 rediagnose` (re-evaluates the two selected final checkpoints and refuses
  unless the policy reproduces the committed P5 evidence). Values: see the P5.2 evidence once taken.
- **P6 pre-registration:** `docs/V3_P6_PLAN.md`, V3-D22 to V3-D25. Baseline replication seed 5103 trained (exact selected
  recipe, fresh, uninterrupted; update-800 S_run 1.5707). `all_info_v1` implemented and tested (30,842,524 parameters,
  0.0365% from ACTIVE; permutation invariance, exhaustive no-truncation trees, gradient coverage, strict identity and
  five-architecture refusals, exact CPU resume). TRAIN-only census (median 140 future states/position, max 353) and CUDA
  preflight (micro16 x accum8, 1.76 h projected, 11.1 GiB peak). No ALL-INFO model had been trained or evaluated on TUNE.
- **Finding recorded:** same-seed ACTIVE and ALL-INFO shared modules do not start with identical weights (lazy,
  order-dependent initialisation); not claimed.
- **NOT RUN:** ALL-INFO training, B0 reference evaluation, Gate I, DAgger, P7.

## V3-E17 - P6 ALL-INFO runs, B0 references and Gate I

- **Date:** 2026-10-01
- **Runs:** three ALL-INFO models (seeds 5101/5102/5103; micro16 x accum8; 800 updates; LR 3e-4), each fresh, uninterrupted,
  `exit 0`, about 92-94 min of training; TUNE evaluated once after update 800. B0 references (ACTIVE selected recipe, zero queries)
  for the same seeds re-evaluated bit-identically to the committed evidence (seed 5103 is the pre-registered replication).
- **Result:** Gate I **PASS**: `Delta = +0.2351`, paired 95% CI [0.2044, 0.2671], threshold +0.20; per-seed +0.2560 / +0.2227 /
  +0.2267; ALL-INFO KQRvK M3 top-1 0.692 vs B0 0.457. Pooled top-1 0.867 vs 0.757; pooled CE 0.976 vs 1.357.
- **P5.2 results:** refined selector accounting (no proof-admissible edge after completion; M3 selector weakness is genuine);
  query-content ablation (teacher-forced accuracy does not depend on state content: query-pattern leakage; ACTIVE does not use
  queried content productively).
- **Evidence:** `docs/V3_P6_RESULTS.md`, `v3-p6-gate1.json`, `v3-p6-allinfo-seed*.json`, `v3-p6-b0-manifest.json`,
  `v3-p5.2-selector-diagnostics.json`, `v3-p5.2-query-content-ablation.json`.
- **NOT RUN:** DAgger/scheduled sampling, P7, Gate II, Gate III, B16, adaptive STOP, CONFIRM, HOLDOUT_C.

## V35-E1 - V3.5 pre-registration and implementation (before any V3.5 measurement)

- **Date:** 2026-10-02
- **Status:** PRE-REGISTERED / IMPLEMENTED. No V3.5 model has been trained or evaluated.
- **What exists:** branch `experiment/workstation-v35-onpolicy` (from `7e508df`); `docs/V35_RESEARCH_PLAN.md`; V35-D1..D5;
  the `QueryTargetProvider` / `Selection::ActiveLabelled` API; the label-only `ProofTargetProvider`; the two-pass trainer
  (`p35`), weights-only init with identity checks, resumable state (`v35_state_v1`); the Gate II / Gate III / Content-Use
  estimators and the outcome classifier; the `recur64 v3-p35` CLI (`preflight`, `train`, `eval-final`, `gate`).
- **Tests (CPU, tiny geometry of the same architecture, real exact positions):** provider cannot change the trajectory or any
  forward value; labels change only the selector gradient; off-target branches receive legal non-empty `A_refute` targets;
  arbitrary learner prefixes get legal targets; completion empties targets while ACTIVE continues; rollout/replay parity;
  microbatch-independent whole-update normalisation; selector weight independent of B0 count; recipe/init identity refusals;
  bit-exact CPU resume; equal budget exposure.
- **NOT RUN:** the throughput pass, the CUDA preflight, V3.5 training, any TUNE evaluation, B16, HOLDOUT_C.

## V35-E2 - V3.5 TRAIN-only throughput profile and CUDA preflight

- **Date:** 2026-10-02
- **Status:** MEASURED (TRAIN only). No TUNE evaluation of any V3.5 model has occurred.
- **Result:** see V35-D6. micro16 x accum8 frozen; steady 1.62 s/update; rollout/replay parity exact; init checkpoints for all
  three seeds verified; no execution-only optimisation adopted (profile does not support one).
- **Evidence:** `docs/evidence/v35/v35-preflight-16x8.json`.
- **NOT RUN:** V3.5 training runs, TUNE evaluation, B16, HOLDOUT_C.

## V35-E3 - V3.5 runs and gates

- **Date:** 2026-10-02
- **Runs:** three on-policy runs (seeds 5101/5102/5103), micro16 x accum8, 800 updates, LR 3e-4, init from each seed's selected P5
  final weights, fresh `adamw-v1`; all exit 0, uninterrupted, ~21-23 min training each; TUNE evaluated at 0/200/400/600/800
  (diagnostic) with the full evaluation at 800.
- **Result (rules applied once):** Gate II FAIL (-0.0222), Gate III FAIL (-0.0227), Content-Use PASS (+0.0369 nats, CI wholly
  positive, every seed), Gate VI PASS. Outcome PARTIAL - CONTENT. M3 ACTIVE B8 proof completion 2.5-4.3% vs 43.2% ceiling.
- **Evidence:** `docs/V35_RESULTS.md`, `docs/evidence/v35/`.
- **NOT RUN:** B16 / Gate V, HOLDOUT_C, any further training, V4 implementation.
