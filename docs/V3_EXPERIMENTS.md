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
