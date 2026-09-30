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
