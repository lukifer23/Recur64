# Recur64 V3 — Lineage

Status: P0 (frozen specification). No science has been run.

## Ancestry

| Item | Ref | SHA |
|---|---|---|
| V3 branch | `experiment/workstation-v3-active-search` | created from the SHA below |
| **Base (exact)** | `experiment/workstation-v25` | `feb86236f24eeaca2a1dc16f7c9e45bca4dc51de` |
| Mainline (unchanged, not merged) | `main` | `fef1ffcf9c38381d4adc671e5e2c5ead9f141e33` |
| HP R15 (reference only) | `origin/experiment/hp-r15` | `3430e6ca23be18ced66d5d16b9d4663eff50850c` |
| HP R15/H3 integration (donor) | `origin/experiment/hp-r15-h3-integration` | `79ffd1429e55e00d2767ffa482f89ced63668720` |

The HP branches are **not merged**. No HP commit is an ancestor of V3. History is not rewritten;
historical experiment documents are not edited.

## Donor references (concepts only; reimplemented)

All from `79ffd14` (HP Chimera V2). Authorship of those files is the HP branch's. V3 re-implements
the concept against V3 contracts; nothing is copied wholesale.

| Donor | Concept taken | How V3 differs |
|---|---|---|
| `crates/recur64-model/src/chimera2.rs` `planner_step` | Bounded gated update `Z' = RMSNorm((1-u)Z + u·proposal)`; RMSNorm everywhere; no unbounded residual | Triggered once per *new exact state*, never on identical information; workspace 256 wide, K=8; per-root-branch memory added |
| `chimera2.rs` `ComputeCounts`, `PlannerDiag` | Per-run work counters and planner diagnostics | Counters extended to query-budget accounting (spec §14) |
| `crates/recur64-coproc/src/world.rs` (WorldModelV2) | Staged exact tool; terminal children expose no future; `fresh_no_history_v1` refusal; semantic differential suite against `GameState` | V3's tool is one edge per call, retains history, and exposes no reply summaries |
| `crates/recur64-cli/src/x2_run.rs` | Compute-honesty discipline (cached training vs live inference wall) | Same rule, re-stated in V3 §28 |
| D-record "uniform_1_4_v1" budget-final training | Loss only at the final allowed step | V3 policy `budget_0_2_4_8_v1` |

## Explicitly NOT imported

- Chimera V1 or V2 model code as a model (no recurrent root transformer, no successor/reply
  encoders).
- `ReplySummaryToken` and any reply-summary / next-player-mate-count field. Those made the V2
  confirmation task near-directly decodable (HP_X2 §analysis) and are prohibited in V3.
- Bulk reveal of the candidate/reply tree.
- WASM guest, ComputeBank/coproc provider artifacts (no portability benefit is claimed yet).
- Same-target deep supervision at every recurrent step (V1 defect).
- The `logit += gain * facts` shortcut (V1). V3 keeps V2.5's first-class fact tokens.
- HP decision numbers D50–D63, which collide with mainline D50–D54 and V2.5 D55–D59. V3 uses
  `V3-D<n>` ids; HP decisions are cited as `HP D<n>`.

## Imported from V2.5 (same branch lineage)

- Root geometry `candidate_v25` CF: width 640, 10 heads, FFN 1280, 8 board blocks; candidate dim
  256, 4 heads, FFN 512, one candidate block, facts hidden 64.
- `CandidateFactsV1` (root only), `ObservationV1`, `ActionId`, `GameState`.
- Exact-data infrastructure: `proof_targets_v1`, `P25_DATA_V1` (44,332 TRAIN positions),
  `HOLDOUT_C` (digest `4ab951c6edd8dd4f531bb87d2f4373895d1fddb70efdf052b24c09a1b71d87d5`, never
  evaluated), exposure-log guard, paired bootstrap, AdamW / accumulation machinery.
- Checkpoint identity and cross-refusal machinery (`CheckpointMeta`, `check_model`,
  `check_contracts`).

## Data custody notes (verified by read-only inspection at P0)

- Data bytes live in git-ignored `runs/v25/p25/` (TRAIN `data/proof-train.json`; TUNE
  `data/proof-tune.json`; holdouts under `holdouts/`). They are not in git.
- The HOLDOUT_C digest is the `ProofTargets::compute_digest` content digest, **not** the SHA-256
  of the file bytes. It must be verified through `ProofTargets::load` (P4). Verifying it is not
  an evaluation.
- Cross-branch disjointness from HP/X1/X2 datasets cannot be verified unless those exact positions
  are present; that limitation will be recorded in P4 rather than assumed away.
