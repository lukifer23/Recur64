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
