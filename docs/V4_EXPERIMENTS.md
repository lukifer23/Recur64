# Recur64 V4 - Experiment Ledger (append-only)

Labels: PRE-REGISTERED / MEASURED / INFERRED / NOT RUN. Never edit past entries; append corrections as new entries.

## V4-E0 - branch creation and pre-registration

- **Date:** 2026-10-02
- **Status:** MEASURED (repository facts); no science.
- **Commands:** `git checkout -b experiment/workstation-v4-evidence-belief 40916509936e83468e2fe58c455a33a246123545`
- **Result:** branch created from a clean tree. V3 and V3.5 branches unchanged.
- **Pre-registered here (docs/V4_RESEARCH_PLAN.md):** laws A-D; module contracts; staged protocol; mechanism questions A-G
  and pass rules; TRAIN-dev partition (`sha256("v4_train_dev_v1|"+canon)` u64 mod 10 == 0 -> DEV); `V4_TUNE_V1` rule
  (KQRvK/KRRvK x M1/M2/M3, 1,000/cell, seed `0x7A40_0001`, disjoint by FEN and canonical class from all prior sets).
- **NOT RUN:** any V4 model, data generation, training, or evaluation. V4_TUNE_V1 not generated. HOLDOUT_C not evaluated.

## V4-E1 - V4-B build: architecture, checkpoint identity, invariants (engineering, no science)

- **Date:** 2026-10-02
- **Status:** MEASURED (CPU, FP32, tiny test geometry for behaviour; full geometry for parameter count). No training data
  was used and no V4 mechanism question was answered.
- **Commands:** `cargo test -p recur64-v4`, `cargo test --workspace` (before the V4 custody additions), `cargo test -p recur64-runtime --lib v4`.
- **Result:**
  - New crate `recur64-v4`; new identity `evidence_belief_v4` (`Architecture::EvidenceBeliefV4`, `EvidenceContracts`,
    `CheckpointMeta.evidence_contracts`); `ModelConfig::refuse_evidence_v4` is called from `bench`, `cuda-smoke`, the mainline
    runtime config, and `proof train/eval`, `v25-qual`, `model-info` bail on the new variant.
  - Full-geometry parameters: **30,023,684** (base 27,731,859 = `root.*` + `base.readout`; evidence 820,328; utility path 1,471,497;
    breakdown printed by `parameter_count_is_independent_of_the_budget_and_groups_sum_exactly`). Budget-independent.
  - 19 invariant tests pass (`crates/recur64-v4/tests/invariants.rs`): B0 independent of budget and bitwise `z0`; zero
    content => bitwise-zero message => bitwise identity update for any routing metadata; permutation invariance (acquisition
    order and ledger slot order, tolerance 1e-5); real vs zero vs shuffled content differ; deltas finite and bounded
    (extreme content x1e4); exact StateQuery accounting incl. leak detection; utility scoring executes no query and the
    `FrontierEdgeView` has no child field; probe forks leave the real manager/tree/ledger/belief bit-identical and the
    final trajectory identical with and without probes, `probe_queries` counted separately; V4 refused by every historical
    identity and vice versa (metadata level + checkpoint round trip + tampered contract); frozen base gets no gradient,
    evidence gets finite non-zero gradient, B0 bit-identical after optimiser steps; zero-content replay along a deep
    reply chain returns exactly to B0.
  - 5 training-plumbing tests pass (`tests/training.rs`).
  - Workspace regression: 513 tests passed, 0 failed (`cargo test --workspace`, run before the V4 custody edit).
- **Defect found and fixed during this entry:** with `Freeze::BASE_AND_EVIDENCE` the pooled root context
  (`BaseStage::root_node`) was still attached to the base graph, so a utility loss could have trained the base root
  encoder. Caught by `utility_losses_train_only_the_utility_path`; fixed by detaching it with the tokens and `z0`.
- **NOT RUN:** CUDA (no GPU test yet), the CLI boundary test for V4, any training on P25 TRAIN, `v4_tune_v1` generation.
