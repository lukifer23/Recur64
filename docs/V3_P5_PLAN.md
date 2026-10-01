# V3 P5 plan: P4.1 training-interface hardening and the bounded LR screen

Status: **PRE-REGISTERED.** This document and decisions V3-D16 to V3-D19 are committed before any
learning-rate-screen result on `v3_tune_v1` exists. Nothing here introduces a performance threshold.
P6 is not authorised by it.

Base: `f56e5d3af773c9676beb391357678887baca87b9` (P4 results; `C_8(KQRvK M3) = 0.4336`, QUALIFIED, V3-D15).

## 1. P4.1: what "completion in exactly Q* queries" means

`proof_trace_v1`, `Q*`, the 0.4336 result, `StateQueryV1` and the generic `A(S) = A_proof U A_refute` are
unchanged. Clarification only:

- `Q*` queries complete the proof **for an on-proof trajectory from the empty queried set** that always
  follows an `A_proof` edge.
- A refutation or off-proof query does not reduce the proof residual. Such a trajectory is longer than `Q*`
  (or never completes inside the budget).
- The generic `A(S)` includes `A_refute` edges, because a certificate needs every defender reply.
  The generic `ProofTraceTeacher` stays as the reference adapter. The P5 training teacher is a different,
  versioned object (section 2).

Tests: the existing completion test was renamed to say "on-proof", and
`off_proof_queries_do_not_reduce_the_residual_and_lengthen_the_trajectory` was added. No behaviour changed.

## 2. `proof_teacher_seeded_v1`

For an example whose proof is incomplete (residual > 0):

1. **Selector target** = uniform over the full tied `A_proof` set (every `A_proof` edge on the frontier).
2. **Followed edge** = one `A_proof` edge chosen by a deterministic seeded uniform draw. The draw is
   `mix(key ^ mix(step)) % |A_proof_on_frontier|` over the edges **sorted by root-relative action path**
   (so it does not depend on frontier slot order). `key = FNV-1a(run key base, position id, ordinal, budget)`.
   No `DefaultHasher` or any process-randomised hash is used. Different occurrences of one position
   traverse different tied orderings; the same (seed, id, ordinal, budget) always gives the same trajectory.
3. At least one `A_proof` edge must exist while incomplete, and `A_refute` must be empty from a pure
   on-proof trajectory; otherwise the teacher raises an error (visible failure, no fallback).
4. **No `A_refute` edge** is ever a primary teacher-forcing target or a followed edge. DAgger rescue is not
   authorised.

**Completion latch.** When the residual first reaches 0, the example latches complete for the rest of the
episode. A latched example never again has a selector target, even if a later filler query opens an
incorrect root branch (which would re-activate refutation edges in the generic `A(S)`). The remaining
budget is spent in `fixed_bfs_actionid_v1` order and contributes policy gradient only (no selector loss).
Regression test: an M1 proof completes on query 1, then a FIXED filler opens an incorrect root branch,
and the teacher still emits no target.

## 3. Inputs (hard refusal)

Only these are accepted; anything else is refused with a visible error.

| input | digest | positions | trace manifest |
|---|---|---|---|
| TRAIN `P25_DATA_V1` | `3b25dc8549dd2fc9d47c30e294c273b3306aecb3eba91b964715326ddf74f2e6` | 44,332 | `8160734ed5e3a145c12dd72d8e9dc49cc893984fc1cf714ea93eba904f58c488` |
| TUNE `V3_TUNE_V1` | `c66018657009c9c5eade58369b5f451466d6662c910f76b8aacddcac99921b53` | 4,500 | `acb786fad82573c9b4427652e5e1b060f7b312c5f5e9121e02980c21cfaa6d25` |

Also refused: a missing trace, a trace whose id or FEN differs from its position, the wrong split, a
failing or incomplete independent audit, HOLDOUT / CONFIRM splits. HOLDOUT_C
(`4ab951c6edd8dd4f531bb87d2f4373895d1fddb70efdf052b24c09a1b71d87d5`) is **never loaded in P5**.

## 4. Objective

`L = mean policy CE (all examples of the optimizer update) + 1.0 * mean selector NLL (supervised decisions only)`.
No WDL term. Both normalisers are the **whole optimizer update** (all 128 examples; all supervised decisions
of those examples), never a microbatch or the B0 count. The teacher does not depend on the model, so a
model-free dry simulation gives the update's supervised-decision count before any gradient; one backward per
microbatch with final weights; gradients are summed by `GradientsAccumulator`. A reference test compares
the accumulated gradient and loss with a monolithic computation and across microbatch splits, and a test
shows the selector weight does not depend on the number of B0 examples. Optimizer: `adamw-v1`, unchanged.

## 5. Schedule

- Training budgets {0, 2, 4, 8} only. B16 is never trained and never used for selection.
- Layout micro16 x accum8, budget sequence [0,2,4,8,0,2,4,8]; 128 examples per update (32 per budget).
- The only permitted fallback is micro8 x accum16, and only if a TRAIN-only CUDA preflight at full geometry
  genuinely runs out of memory. The resolved layout is committed before any LR screen.
- Four independent `cell_balanced_v1` samplers, one per budget, seeds derived from the run seed and a frozen tag.
- 800 updates, warmup 80, the existing linear-warmup-plus-cosine schedule. No early stopping, no best
  checkpoint; selection uses exactly update 800.
- Projected single-run wall time must be under 2 hours (measured in the preflight and committed).

## 6. Screen

Peak LR {7.5e-5, 1.5e-4, 3.0e-4} x seeds {5101, 5102}: six runs. For one seed, all learning rates share the
initial weights, the sampled examples, the tie choices and the budget sequence. A run is ineligible only on
non-finite loss or gradient, a health refusal, a query correctness error, or a checkpoint/resume error.

TUNE evaluation (all 4,500 positions) at updates 0, 200, 400, 600, 800 with ACTIVE selection at B0/B2/B4/B8:
per family, depth, cell, macro and pooled top-1, correct mass, CE, entropy and chance.

## 7. Selection rule (exact, applied once)

`S_run` = mean over budgets {0,2,4,8} of the mean over the 6 TUNE cells of the ACTIVE policy CE at update 800
(24 equal values). `S_lr` = mean of the two seeds' `S_run`. The lowest `S_lr` wins. An exact tie goes to the
lower LR. No tolerance and no human override. FIXED, oracle, selector rates, B16 and Gates I-III are
diagnostics only. The rule is implemented as a pure tested function and the `select` command refuses to run
twice.

## 8. Update-800 diagnostics (recorded, never gating)

ACTIVE query classification (proof_admissible / refute_admissible / off_target), first-query correct-root
rate, mean and median depth, root-branch coverage, exact proof residual change per query, final residual,
completion fraction, the ideal ceiling (`Q* <= B`), teacher-forced policy metrics at B2/B4/B8 (ACTIVE-vs-teacher
gap), the FIXED diagnostic and selector NLL under the teacher.

## 9. Run identity and resumability

A deterministic recipe digest covers the architecture and its 11 contracts, data and trace digests, sampler and
tags, teacher id, latch semantics, budgets, sequence, layout, updates, warmup, LR, loss weights, optimizer
contract, seed and precision. A strict sidecar (`v3_p5_state_v1`) is written next to every checkpoint; any
other recipe is refused on resume; historical checkpoint identities are unchanged. Resume fast-forwards the
samplers, which are pure functions of (seed, draw index). A CPU test proves the resumed run is bit-exact.

## 10. Commits and stop

A: P4.1 + P5 infrastructure, tests, this plan, decisions. B: the resolved layout and preflight evidence, pushed
before screening. C: the six runs, selection applied once, evidence and results. Then **stop**.
`P6 NOT RUN - awaiting owner review/approval.`

## 11. Out of scope

Anything on the do-not-touch list: the P4 gate and result, `Q*`, `StateQueryV1`, descendant CandidateFacts,
model geometry, B16 training, HOLDOUT_C, P6, ALL-INFO, Gates I-III, DAgger, adaptive STOP, self-play,
conversion, engine or tablebase labels, B16 for tuning, intermediate-checkpoint selection.
