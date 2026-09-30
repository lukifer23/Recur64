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
