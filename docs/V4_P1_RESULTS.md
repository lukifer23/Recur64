# V4 P0/P1 — `evidence_belief_v4`: mechanism results

**Outcome: ARCHITECTURE-STOP at Stage B (TRAIN only).** The evidence mechanism is built, correct and instrumented, but as trained
it does not improve decisions: A holds only by a formality (2e-6 nats), B and G fail, C passes. Per the pre-registered stop rule,
architecture development stopped; Stage C (utility head) and Stage D were not run, so D, E, F are unanswered.

`V4 FINAL SCIENCE NOT RUN.` `HOLDOUT_C REMAINS SEALED AND UNEVALUATED.` `v4_tune_v1` generated and sealed, NOT evaluated.

Plan: `docs/V4_RESEARCH_PLAN.md`. Novelty review: `docs/V4_NOVELTY_REVIEW.md`. Ledger: `docs/V4_EXPERIMENTS.md` (V4-E0..E7),
`docs/DECISIONS.md` (V4-D1..D3). Evidence: `docs/evidence/v4/`. Branch `experiment/workstation-v4-evidence-belief`, from V3.5
HEAD `40916509936e83468e2fe58c455a33a246123545`.

## 1. What was built (MEASURED)
- New identity `evidence_belief_v4`, refused by every historical command and vice versa (the boundary test runs the real binary).
- 30,023,684 parameters, budget independent: base 27,731,859; evidence 820,328; utility path 1,471,497.
- Law A (immutable `z0`), Law B (exact-zero content causality by bias-free, f(0)=0 construction), Law C (explicit
  `EvidenceMessage` ledger with norm, branch, depth, trust, per-candidate delta), Law D (utility head with no child field).
- 20 architectural invariant tests, 5 training-plumbing tests and 4 CLI boundary tests pass; workspace regression 513 passed, 0 failed.
- CUDA smoke (TESTED on the GPU): CPU/CUDA parity 5.4e-5, B0 bit-identical after a GPU update, checkpoint round trip.
- Cost: B0 about 19 ms and B8 about 41 ms per batch of 16; Stage B update 0.3-0.5 s at 0.85-1.5 GB VRAM; about 940 probe queries/s.
- `v4_tune_v1`: 6,000 positions, 1,000 per cell, digest `b83624e4...af77`, audited, disjoint from 15 datasets, regeneration identical,
  sealed, never evaluated. HOLDOUT_C digest re-verified only.

## 2. Mechanism questions (TRAIN-only, `V4_TRAIN_DEV`, 4,403 positions, seeds 5101-5103)
Stage A bases (recorded, not gated): top-1 0.840 / 0.846 / 0.850, CE 1.261 / 1.243 / 1.229, correct mass about 0.72-0.73.
Stage B: LR screen {3e-4, 1e-3, 3e-3} on seed 5101 (the frozen rule picked 3e-3 by a 1e-7 CE margin, i.e. noise), then 1,200 updates per seed.

| Q | Pass rule | Measured (FIXED and RANDOM schedules alike) | Verdict |
|---|---|---|---|
| A | CE(B0) − CE(B8) CI wholly > 0, every seed > 0, delta non-degenerate | mean +2.3e-6 / +2.2e-6 nats (CI [3.2e-7, 5.8e-6]); per seed 4e-8, 6.7e-6, 2.9e-7 | passes the letter, **no practical effect** |
| B | zero- AND shuffled-content CE worse than normal, CI wholly > 0 | zero − normal +2.3e-6 (positive); shuffled − normal +3.0e-8, CI [−3.6e-11, 8.7e-8], one seed negative | **FAIL** |
| C | zero content = B0 bitwise | max CE difference 0.0 | **PASS** |
| G | top-1 B8 − B0 CI wholly > 0 and mass rises B0 → B2 → B4 → B8 | top-1 change exactly 0.0; mass 0.729616 → 0.729617 → 0.729618 → 0.729618 | **FAIL** (a confidence-only change at the 1e-6 scale) |
| D, E, F | utility head | NOT RUN (stop rule) | unanswered |

All three LR screens gave the same picture (CE change between −9e-5 and 0 nats; A, B, G failing). Multi-step (BMPS) utility
diagnostics: NOT RUN.

## 3. Reading, and what is NOT established
- Established: the machinery is exact (content causality, B0 immutability, accounting, probe isolation, GPU parity), and an
  evidence path trained with this parameterisation for 1,200 updates does not use content to change decisions.
- **INFERRED, NOT TESTED:** the cause is an optimisation dead-start. `delta` is a product of `trust = tanh(t^2)` (about t^2 at
  initialisation), a small-std output layer and a gate: a near-zero saddle with vanishing gradients (measured evidence delta norm about
  0.002 after 400 updates). If so, this is a parameterisation or initialisation defect, not evidence against active evidence.
  It could equally be a capacity or signal problem; the two have not been separated.
- Not demonstrated is not disproved: nothing here says exact-state evidence cannot help (V3 Gate I, +0.2351, says it can).

## 4. Defects found and fixed during the pass (all logged; none changed a pass rule)
Root-context leak into the frozen base's gradient (caught by test, V4-E1); Stage B/C graph retention of the frozen base causing
CUDA out-of-memory (V4-E5/E6, fixed by computing it graph-free; losses identical before and after); a character-literal typo in the
tune-custody code (compile-time). The first Stage B failure also went unnoticed for about half an hour because the run was not checked.

## 5. Recommendation for V4-P2 (owner decision; NOT started)
1. Diagnose cheaply on the existing Stage B models (trust and gradient magnitudes per parameter group): confirm or refute the saddle.
2. If confirmed, pre-register ONE parameterisation change on TRAIN only (for example a non-vanishing trust form, a larger output
   initialisation, or an evidence warm-start), rerun Stage B under the same pass rules, and proceed to Stage C only if A, B, C and G
   all hold with a practically meaningful effect.
3. Keep `v4_tune_v1` and HOLDOUT_C sealed until that qualification passes.

## 6. Proposed final V4_TUNE gates (NOT APPLIED)
`ACTIVE_B8 − B0` CE and top-1 CI wholly > 0; `ACTIVE_B8 − FIXED_B8` and `− RANDOM_B8` CI wholly > 0; Content-Use (`CE_ablated − CE_normal`
> 0); B0 bit-identity before and after active training; selector regret and positive-utility rate vs random. Seeds, thresholds and
bootstrap seeds to be frozen before `v4_tune_v1` is opened. Applied to nothing in this pass.

`V4 FINAL SCIENCE NOT RUN.` `HOLDOUT_C REMAINS SEALED AND UNEVALUATED.`
