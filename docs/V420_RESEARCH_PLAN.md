# V420 — adaptive-depth recurrence: research plan and pre-registration

**Status: PRE-REGISTERED (owner approved 2026-11-11; thresholds, arms, kill rule and the 3 GPU-hour ceiling are FROZEN as written; a failed S1 ends V420; option 2 stays unbuilt unless S1 passes). No code, no data and no run exists for V420 at this commit.** Where the text says PROPOSED it now means approved. Branch `experiment/workstation-v420-recurrent-depth`, from V4 HEAD `d7248e7`.
V420 is a NEW identity. It touches no frozen V4/V3.5/V3 identity, no sealed set (`v4_tune_v1`, HOLDOUT_C), and no V4 weights.

## 1. Why this, and what carries over from V4

V3, V3.5 and V4 all built an exact, instrumented auxiliary mechanism (state queries / evidence messages) that did not change
decisions. The V4 saddle diagnostic (V4-E8, V4-D4) MEASURED zero gradient on the evidence path at the end of training. The lesson
(INFERRED) is that a gated, zero-initialised side channel beside a strong frozen base starves itself of gradient. V420 therefore:

1. Has **no side channel and no frozen base**: one shared block is applied K times in the main path, all trained together.
2. Is **signal-first**: a falsifiable smoke (section 5) decides in about one GPU-hour whether depth helps at all, before any
   architecture work.
3. Uses **exact labels** (proof/tablebase-style targets from the P25 TRAIN data) so a result is attributable.
4. Pre-registers a **gradient health check** (section 6) so a dead path is detected in minutes, not after 1,200 updates.

## 2. Hypothesis and the honest prior

**H:** on hard exactly-labelled endgame positions, the accuracy of one shared-parameter block improves as the number of applications
K grows, including for K larger than any value used in training, and the gain is not explained by extra parameters or extra training.

**Prior evidence against (must be stated):** the HP branch found R=1 beating R=4 for VALUE learning at 15M, two seeds, matched LR
(`docs/HP_BRANCH_COMPARISON.md`); the dedicated R10 recurrence study was deferred (`docs/STATUS.md`). The V4 novelty review also
records recurrence over unchanged information as having shown no benefit. The difference here (supervised exact targets, a
stratified hard cell, test-time K extrapolation, depth-aware training) is a reason to test, not evidence of success. A flat curve
is the most likely outcome; the smoke is built to find that out cheaply.

## 3. Task and data (TRAIN/DEV only until the smoke passes)

- Positions and labels: the existing P25 TRAIN data (`runs/v25/p25/data/proof-train.json`) and the V4 `V4_TRAIN_DEV` split / `DEV1000`
  protocol (same fit/dev digests as V4-E2), root-move correct-set targets, CE and top-1 as in V4 Stage A.
- Difficulty: report per cell; the headline cell is the hardest available cell (the longest proof depth). Which cell that is will be
  read from the data manifest in P1, before any model runs, and written into the ledger.
- NOT used: `v4_tune_v1`, HOLDOUT_C, any V4/V3.5 weights. If the smoke passes, a fresh sealed tune set is generated under its own
  custody entry before any claim; if it fails, nothing is sealed or opened.

## 4. Model arms (one identity, `recurrent_depth_v420`)

Shared geometry, same parameter count in every arm unless stated. Exact block geometry is fixed in P1 from the existing
`recur64-model` recurrent core (shared core R=1/2/4, full backprop, deep supervision switch, already gradient-verified in
`docs/STATUS.md`) adapted to the candidate-token policy readout; P1 must record the resolved layout before any training.

| arm | description | role |
|---|---|---|
| R1 | one application (K=1) | same-parameter baseline |
| RK | the shared block applied K times, trained with a fixed K in {2, 4} (separate runs) | does depth help when trained for it |
| RV | shared block, K sampled per update from {1..4}, evaluated at K in {1,2,4,8,16} | extrapolation and adaptive depth |
| WIDE (control) | an unshared model with the compute of K=4 and 4x the block parameters | separates "more compute" from "shared recurrence" |

All arms: identical data order, optimiser contract, updates and seeds (5101, 5102, 5103), fresh initialisation. Training length
is fixed in P1 from the measured R1 plateau on the dev loss (not tuned per arm).

## 5. Smoke and gates (all on TRAIN-DEV, headline cell)

| gate | PROPOSED pass rule |
|---|---|
| S0 health | every arm: finite loss, no non-finite gradient, shared-parameter gradient non-zero at every K used (the AGENTS.md recurrence rule), and the section 6 check passes |
| S1 depth helps | top-1(RK, K=4) − top-1(R1) on the headline cell: pooled paired-bootstrap 95% CI wholly > 0, every seed > 0, pooled gain at least +0.03 |
| S2 extrapolation | top-1(RV at K=8) ≥ top-1(RV at K=4) − 0.01 on the headline cell, pooled, and RV at K=16 not worse than K=4 by more than 0.02 (no collapse past the training depth) |
| S3 not just compute | top-1(RK K=4) − top-1(WIDE) not below −0.02 (recurrence is competitive with an unshared model of equal compute); reported, not a kill |
| S4 baseline replicated | R1 reproduces the V4 Stage A top-1 band (0.840–0.850, all positions) within 0.01 on the full dev set |

**KILL rule (pre-registered):** if S0 fails, fix the plumbing and re-run once with the fix logged; if S1 fails (CI includes 0, any seed
≤ 0, or gain < +0.03), the answer is "depth does not help here": V420 stops, the result is written up, nothing is retried or retuned. If S1
passes and S2 fails, record "helps at trained depth, does not extrapolate" and stop; S3 and the follow-ons are decided by the owner.

## 6. Gradient health check (the V4 lesson, applied from the start)

At update 0, 50 and 200 of every run, log per-layer gradient and weight norm for the shared block and the readout, plus any gate or
normalisation scale. Abort the run and report if the shared-block gradient norm is below 1e-6 relative to its weight norm at
update 50 or has fallen by more than 3 orders of magnitude from update 0. The check is read-only, uses the same visitor as
`recur64 v4 saddle`, and costs seconds.

## 7. Compute and scope

- Hardware: this workstation (RTX 2000 Ada, 16 GB), CUDA FP32 as in V4. Budget ceiling for the whole smoke: 3 GPU-hours; stop and ask if
  the first measured R1 update implies more.
- Files owned (new): a new crate or module `recur64-v420` only after P0 approval; `docs/V420_*.md`; `docs/evidence/v420/`; scripts `scripts/v420_*.sh`.
- Prohibited: any change to V3/V3.5/V4 crates or frozen identities; self-play, search, UCI, arenas, diffusion; opening any sealed set; a
  second run after a failed gate; claiming a device/precision that did not run; claiming recurrence works from a gradient or a loss alone.

## 8. Phases

- **P0** — this plan approved (owner). Output: approving commit.
- **P1** — read the existing recurrent core and P25 data manifest; write the resolved arm layouts, the headline cell, training length
  and the exact commands into `docs/V420_EXPERIMENTS.md` BEFORE any training; CPU correctness tests (R1 parity with the explicit
  control, shared-gradient non-zero at K=4, zero-new-parameters with K); CUDA smoke. No result is produced in P1.
- **P2** — run the smoke (S0–S4), apply the rules once, write `docs/V420_RESULTS.md` with DETECTED vs TESTED kept distinct.
- **P3 (only if S1 and S2 pass; separate approval)** — adaptive halting and a fresh sealed tune set.

## 9. Open decisions for the owner

1. Approve the arms, gates and thresholds as PROPOSED, or change them now (they freeze on approval).
2. Approve the 3 GPU-hour ceiling.
3. Confirm that a negative S1 ends V420 (recommended) rather than prompting a larger-scale retry.
4. Option 2 (iterative plan refinement) stays unbuilt unless S1 passes; confirm that is acceptable.
