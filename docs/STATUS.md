# Recur64 — Status

## V5 HP counterfactual relational loop (2026-10-03)

- Lineage and single-seed fixed-reader pilot are PRE-REGISTERED from accepted V4
  source `17a782f8ebe68d1519ba3dc808c2473f0fd3a9f5`.
- After owner delegation, the complete-frontier correction passes the retained
  regression: all 19 previously omitted edges are included. Depths 6..16 and
  disjoint structural fields are tested; the amended count is 7,162,896.
  `V5_ROOT_CAUSE.md` and `V5_NUMERICAL_ROOT_CAUSE.md` preserve failure evidence.
- HP hardware, toolchain and CUDA runtime are DETECTED. Historical network
  fixture checks passed CPU/RTX 2050 in FP32 at microbatch 2. They do not satisfy
  the full engineering gate. Corrected current CPU/RTX 2050 functional fixture
  qualifications pass all recorded gates without relaxed assertions at source
  64c4dd4, FP32/microbatch 2. Full release workspace: 578 passed, two ignored.
  CUDA sampled device-wide peak 1,068 MiB; all 50 resident samples 364 MiB.
  Detailed component/end-to-end timing is not inferred from qualifier intervals.
- Synchronized real-path profiling and strict qualification-loader checks are
  implemented (`V5_TIMING_ACCOUNTING.md`). New CPU padding parity tests pass
  for all gradients and AdamW state at R1/R2/R4. Full release suite: 581 passed,
  zero failed, two preserved ignores, native exit 0. Current source d970049
  qualification: CPU PASS, CUDA FAIL at normal/profile exact output-gradient and
  AdamW parity. Other recorded checks pass but do not clear the gate. STOP before
  drill/training; `V5_PROFILING_ROOT_CAUSE.md` records the bounded next diagnostic.
  Standalone graph source provenance also remains an open contract audit.
- The required P25 TRAIN artifact is not present on this HP. Code and fixture-only
  work may proceed under the owner's next-step delegation. Drill/training/
  DEV evaluation require exact custody, the FIT drill and complete accounting.
- The complete bounded drill, update-0/update-800 reader matrix, interventions,
  composition analysis, exact bootstrap gates and conditional R8 commands are
  implemented and tested at their non-data boundaries. They remain NOT RUN.
- Seeds 5302/5303, controller training, self-play and sealed evaluation are NOT
  RUN and not authorized.

## COMPLETED

- Canonical repo identity: project renamed to **Recur64**; specs preserved as
  `docs/RECUR64_RESEARCH_AND_BUILD_PLAN.md` and `docs/RECUR64_PHASE0_KICKOFF.md`
  (labels changed, technical requirements unchanged).
- Minimal two-crate workspace (`recur64-model`, `recur64-cli`), pinned Rust
  toolchain, committed `Cargo.lock`.
- Native Windows build/link proof (Burn 0.21.0 + Flex via `rust-lld`/xwin-splat).
- `recur64 doctor` (read-only, DETECTED vs TESTED).
- `recur64 model-info` (exact parameter counts + executed-block accounting).
- Model-shaped probe graph: bidirectional square-token transformer, pre-RMSNorm,
  MHA with relative-displacement bias, GeLU FFN, residuals.
- Sparse legal-candidate policy with joint masked softmax, promotion deltas,
  terminal bypass.
- Pooled WDL head; policy + WDL cross-entropy.
- Shared recurrent core with R=1/2/4, full backprop, deep-supervision switch.
- AdamW training; bounded overfit proof.
- Training checkpoint (model + optimizer + metadata) with schema versioning.
- `recur64 bench` bounded matrix with JSON/markdown output (CPU and CUDA).
- `recur64 cuda-smoke` GPU proof (feature-gated).
- User-space CUDA 12.9.1 runtime (no admin) and native Windows Burn CUDA build.
- Precision gate (FP32 accepted; BF16/FP16 fail visibly).
- Documentation: `HARDWARE.md`, `ARCHITECTURE.md`, `BENCHMARKS.md`,
  `DECISIONS.md`, `STATUS.md`, `README.md`, `AGENTS.md`.

## VERIFIED (test evidence)

- 26 tests pass (`cargo test`).
- CPU FP32 forward is exactly repeatable (max abs diff = 0).
- CPU FP32 training is deterministic across identical runs (Δloss = Δweight = 0).
- Fixed synthetic fixture overfits: loss 3.30 → ~2e-7 within 100 steps.
- Legal candidate probabilities normalize to 1; padding is exactly 0.
- Terminal rows bypass the softmax with no NaN.
- Promotion path receives a nonzero gradient.
- R=1 parity: recurrent loop equals the explicit control graph.
- Shared-core gradient is nonzero at R=1 and R=4, changes with recurrence, and
  matches a finite-difference check (relative error < 0.25).
- Batch inference equals single-item inference within 1e-5.
- Parameter count is independent of recurrence; F10 == R10 unique params.
- Checkpoint resume on CPU FP32 is **bit-exact** (Δloss = 0, Δweight = 0).
- Schema-mismatched checkpoints are refused visibly.
- Unsupported precision requests fail visibly.
- Native Windows Burn CUDA builds and links against the user-space CUDA 12.9.1
  runtime.
- `recur64 cuda-smoke` **PASS** on the RTX 2000 Ada: FP32 forward at R=1/2/4
  (8/12/20 blocks) finite, backward + AdamW update moves parameters, GPU
  checkpoint restore is exact (delta 0).
- Synchronized GPU benchmark matrix (R10, batch 1–128, R=1/2/4) produced finite
  results; see `BENCHMARKS.md`.

## FAILED

- None. (A resume divergence was found and fixed; see `DECISIONS.md` D6.)

## NOT RUN

- F10/R10 CPU benchmarks.
- BF16 / FP16 full-graph tests.
- Peak VRAM/host-RAM measurement and checkpoint timing under load.
- WSL2 evaluation (not required: native Windows CUDA works).
- GPU bit-exact determinism (not claimed).

## BLOCKED

- Nothing blocks the Phase 0 gate. BF16 remains intentionally unverified.

## Next gate

Phase 0 is **GO**: CPU FP32 correctness and the native Windows CUDA FP32 graph
are both verified. BF16 is the only outstanding precision item and is deferred
until it can be tested as a complete graph. The next phase (chess contracts) can
proceed.

---

# Phase 1 — Chess contracts

## COMPLETED

- New `recur64-core` crate (CPU-only, Burn-free) pinned to `cozy-chess = 0.3.4`.
- `square`: canonical (side-to-move) transform; `action`: `ActionId` V1 +
  `ActionList`; `uci`: `StandardMove` + cozy/standard/UCI conversion.
- `game`: `GameState` with authoritative history and a single `apply` path.
- `rules`: Rules Profile V1 termination/precedence + conservative insufficient
  material.
- `observation`: Observation V1 `[64,119]` encoder.
- `perft`: traversal through Recur64's own conversion.
- `fixtures`: CPW perft fixtures + tactical edge cases with provenance.
- CLI: `perft`, `validate-position`, `encode`, `bench-core`.
- Model re-exports core action constants (single source of truth).
- Docs: `REPRESENTATIONS.md`, `RULES_PROFILE.md`, ADRs D9–D15.

## VERIFIED (test evidence)

- `cargo test --workspace` passes; Phase 0's 26 tests unchanged.
- cozy-API contract test pins square ordering, castling, EP, repetition.
- Action encode/decode round-trips over the full 20,480 space; physical↔canonical
  involution; castling actions symmetric across colors.
- Legal-candidate invariants (unique, exact count, decode-to-legal, bijection) on
  the CPW positions and edge cases; terminal positions yield empty lists.
- Castling round-trips through cozy/internal/UCI/action for all four cases;
  promotions (N/B/R/Q × color × capture) collision-free.
- Observation schema arithmetic (119/7616), startpos and after-1.e4 fixtures,
  unavailable frames zero-not-empty, history frames use the current perspective.
- Termination precedence, threefold by knight shuffle, 50-move at clock 100,
  truncation-not-draw, insufficient-material recognized/non-recognized sets.
- Perft matches published CPW counts at CI depths.
- Seeded random games are self-consistent and reproducible.
- **Independent oracle:** with `--features oracle`, legal move sets and
  mate/stalemate match `shakmaty` over 200 random games.
- Model boundary: real positions feed the Phase 0 model (policy normalizes, WDL
  finite); terminal positions bypass the policy path.
- fmt and clippy clean for default and `oracle` feature sets.

## FAILED

- None.

## NOT RUN

- Deeper perft depths beyond CI (available as `#[ignore]` / CLI).
- BF16/FP16 (still gated, Phase 0).

## BLOCKED

- Nothing blocks the Phase 1 gate.

## Next gate

Phase 1 is **GO**: observation and action V1 are documented and tested, legal
moves map one-to-one to actions, castling/promotions/canonicalization are proven,
history/repetition/termination are correct, CPW perft and the independent oracle
agree, and real chess data feeds the existing model. Phase 2 (Micro vertical
slice: inference + PUCT + self-play + replay + learner) may proceed.

---

# Phase 2 — First complete vertical slice

## COMPLETED

- New crates `recur64-search`, `recur64-runtime`, `recur64-eval` (acyclic graph).
- PUCT with perspective-safe backup, deterministic tie-breaks, exact budget.
- Single-owner batched inference with metrics and always-respond shutdown.
- Independent self-play games; Rules Profile V1 terminations; truncation ≠ draw.
- Replay V1: versioned/checksummed/atomic shards, reader, audit.
- Learner: real positions, policy + WDL CE, truncated games excluded.
- Checkpoint schema v2 with contract versions and `model_id` content hash.
- Paired-color systems arena.
- Bounded `recur64 run` coordinator with run directory and Ctrl+C recovery.
- CLI: `selfplay`, `replay-audit`, `train`, `arena`, `run`, `report`.
- Docs: `SEARCH.md`, `REPLAY.md`, `RUNS.md`, ADRs D16–D22.

## VERIFIED (test evidence)

- `cargo test --workspace` passes; Phase 0/1 tests unchanged.
- PUCT synthetic suite (one move, unequal priors, sign inversion, terminal
  win/loss/draw, zero-visit, ties, budget, no NaN) and real-chess suite
  (mate-in-1, forced move, stalemate, neutral perspective).
- Inference: concurrent requests all answered; batching coalesces; errors
  propagate; shutdown fails further requests visibly; metrics recorded.
- Self-play games are legal, replayable move-for-move, and reproducible by seed.
- Replay round-trip; CRC detects corruption; partial `.tmp` ignored; audit
  rejects truncated-with-outcome, illegal selected action, bad target sum.
- Learner reconstructs correct WDL perspective (Fool's mate) and moves parameters.
- Checkpoint v2 round-trip, `model_id` content hash, contract mismatch refused.
- Arena runs paired colors, reproducible from seed.
- **End-to-end:** CPU and CUDA `recur64 run` complete COLLECT → AUDIT → TRAIN →
  EVALUATE → REPORT. CPU smoke: 8 games, 346 examples, audit clean, loss
  4.01 → 1.42, arena ran. CUDA smoke: 17,227 requests, 0 errors, batching mean
  5.5, training + arena ran.
- Interruption: pre-cancelled run is `interrupted` with a recoverable reference
  checkpoint and no candidate; cancel during collect never corrupts replay.

## FAILED

- None.

## NOT RUN

- BF16/FP16 (still gated).
- Gumbel, recurrent R10 comparisons, diffusion (deferred).
- Long training / strength evaluation (not a Phase 2 goal).

## BLOCKED

- Nothing blocks the Phase 2 gate.

## Next gate

Phase 2 is **GO**: the whole learning system closes the loop truthfully and
reproducibly on real legal chess with real neural outputs. Strength was never the
goal; Micro remains weak by design. Phase 3 (F10 baseline and longer pilots) may
proceed only after review of these artifacts.

---

# Phase 3 — F10 + PUCT control baseline

## COMPLETED

- `RunConfig` Phase 3 fields (cycles, budgets, reuse, warmup/planned, accumulation,
  snapshot policy, opening suite, config hash) and `lineage.jsonl` provenance.
- `bench-runtime`: CUDA warmup + batching/active-game sweep (cold/warm separated).
- Fixed a real concurrency bug: `active_games` now means concurrent games, so
  batches coalesce (mean 20–34; was 1.0).
- Streaming replay sampler + capacity archiving (bounded memory).
- Learner hardening: policy/WDL loss split, grad norm, warmup+cosine schedule,
  gradient accumulation, per-update metrics, health guards.
- F10 checkpoint/resume proof (bit-exact on CPU; schedule + optimizer preserved).
- Raw-policy evaluator; frozen opening suite; arena openings + 95% CI.
- `recur64 pilot`: bounded multi-cycle controller with a conservative snapshot
  policy.
- CLI: `bench-runtime | gen-openings | eval-policy | pilot`.
- Docs: `F10_BASELINE.md`; configs `f10-baseline`, `f10-pilot`, `f10-stage-c`,
  `f10-smoke`, `f10-sweep`, `openings-v1`.

## VERIFIED

- `cargo test --workspace` passes (~180 tests); fmt/clippy clean.
- F10 standard-start self-play runs legally (zero illegal actions); audit passes.
- Batcher coalesces; warmup recorded separately.
- Streaming sampler excludes truncated games; capacity archives oldest shards.
- Learner schedule/accumulation/metrics unit tests pass.
- F10 resume is bit-exact (Δloss = Δweight = 0).
- Bounded pilot completes multiple cycles and writes a report + lineage.

## FAILED / WEAK (honest)

- **Learning health is poor at this scale:** loss unstable within cycles
  (often rising), grad norms high (44–88), replay reuse far below target
  (0.07–0.13 vs 2.0), and raw policy vs random below 0.5.
- **Searched self-play is repetition-dominated** (arena near-all threefold/
  fifty-move draws), so the searched arena is uninformative for untrained models.

## NOT RUN

- The full ~2h Stage C pilot and the ~24h baseline (blocked on the fixes below).
- BF16/FP16 (gated).

## BLOCKED

- Long baseline is **CONDITIONAL GO**: fix reuse/update scaling, training
  stability, and repetition-dominated search, then re-run a bounded pilot.

## Next gate

Phase 3 pilot is **CONDITIONAL GO**. See `docs/F10_BASELINE.md` for the decision
package. No ~24h run without explicit owner approval.

---

# Phase 4 — Mainline harness convergence (in progress)

The generic harness improvements proven on the experimental branch
`experiment/hp-r15` were brought onto main without importing HP scientific
assumptions (P4.0/P4.1; `docs/PHASE4_CONVERGENCE.md`). The GPU phase (P4.2+)
is recorded in **`docs/PHASE4_RESULTS.md`**.

## COMPLETED

- P4.0/P4.1: collection semantics, seed policy, gradient reduction, optimizer
  continuation, conservative-v2 promotion, frozen references, identity/hashes,
  provenance, NVRTC and precision gates (see `PHASE4_CONVERGENCE.md`).
- CUDA runtime proven on the RTX 2000 Ada (cuda-smoke PASS; the NVRTC guard
  accepts `nvrtc64_120_0.dll`).
- Sweep methodology: requested vs effective concurrency, refusal of
  unrealizable cells, and a multi-wave comparison. Shared GPU telemetry.
  `bench-train`, `bench-lifecycle` and `search-gain` probes.
- **P4.3 hardware schedule MEASURED** (`configs/hardware/workstation-main.toml`):
  - self-play: 32 concurrent, cap 32, 500 µs, cpu_workers 32
  - learner: 64 × 4 (effective 256)
- **Root-cause fixes (D40–D43):**
  - head v2: final pre-head RMSNorm, 1/√d policy logits, zero-init WDL
  - checkpoints carry `head_version` and every load path checks contracts
  - root Dirichlet noise and argmax after ply 30 in self-play (arenas
    noise-free)
  - `argmax_after_ply`, previously a dead identity field, is implemented
  - inference-only commands run on the inner backend
- Frozen F10 reference **v2** `d22c78bd…` (head v2) and its T0.

## VERIFIED

- fmt/clippy clean; `cargo test --workspace --release` 196 passed, 0 failed,
  1 ignored, at the smoke binary (`7a8b492`).
- Every load path refuses a checkpoint with a model-config or head-version
  mismatch.
- A fresh R10 at R1/R2/R4 also starts at 1.000 × uniform with value 0.000.
- A fresh F10 prior is 0.999 × uniform entropy with value 0.000 (was 0.502 /
  0.245) — `tests/t0_prior.rs`.
- F10 == R10 == 9,805,672 unique parameters under head v2.
- Changing only the schedule left data aggregates identical in every P4.3
  cell.
- Head v1 checkpoints are refused under head v2.

## FAILED / FOUND

- `eval-policy` on the autodiff backend filled 16 GB of VRAM (fixed, D43).
- Head v1 made self-play distill an arbitrary initial prior. Search gain fell
  with budget, and games were repetition-dominated (fixed, D40/D41). The v1
  reference and the v1 P4.4 curve are superseded evidence.
- The T0 search-gain gate was a design error. It is now a learning-progress
  metric (D42).

## PHASE 4 GPU RESULTS (details in docs/PHASE4_RESULTS.md)

- P4.4 search budget: **64 simulations/move**, frozen by the pre-registered
  rule on reference v2 (curve 8-256; 128 failed both override conditions).
- P4.4L lifecycle: **GO after fix D44.**
  - Inference-owner VRAM grew 453 MiB to 10.5 GB over 32 lifecycles, because
    CubeCL's per-thread stream pools were orphaned.
  - After the fix it plateaus at about 1 GB.
- Science parity was proven for the addendum code (identical data
  aggregates and scientific hash).
- **P4.5 F10 smoke: CONDITIONAL.**
  - Every system, data and training gate passed: 0 inference errors across
    756k requests, audit clean, reuse 2.00/2.01, cap never bound, VRAM
    stable, 44.4 min wall.
  - The WDL head learns (loss 1.10 to 0.83). The policy target is still
    near-uniform.
  - The searched arena is 75% threefold, leaving 3-5 decisive games of 32.
    Both cycles held, so learning does not compound.

## POST-SMOKE (2026-09-25; details in docs/PHASE4_RESULTS.md)

- **D45 arena exploration (adopted by the pre-registered rule):** sample 30
  plies, then root noise 0.25. On the same model pair, decisive games rose
  from 3 to 24 of 32 and threefold fell from 24 to 0.
- **D47 multi-leaf PUCT with virtual loss (adopted at K=2):** +83% trainable
  pos/s with unchanged data health.
- **D48 continuous trainer (adopted):** held candidates keep training.
- **D38 evaluation deadline, D37 crash-safe archival, D46 at most two
  resident models:** implemented and tested.
- **F10 smoke v2: GO for the learning mechanism.**
  - Training compounds: WDL loss 1.10 to 0.63 over 3 cycles.
  - Two promotions.
  - Once a promoted value head generates self-play, search movement over the
    prior rises from 11% to 37% (KL 0.02 to 0.40).
  - Strength over T0 is not yet shown (0.500 vs the frozen reference).
  - Watch item: self-play draw share 0.25 to 0.73 in cycle 3.
- Gate: fmt/clippy clean; 207 tests passed, 0 failed, 1 ignored.

## DRAW-DRIFT ROOT CAUSE (2026-09-29; details in docs/PHASE4_RESULTS.md)

- P4.6 was stopped by the owner after 6 cycles: 3 promotions; vs the frozen
  reference 0.578 then 0.594; self-play draws rose to 0.72-0.78.
- **Draw diagnostic:** 74-90% of draws are failed conversions (a rook-or-more
  lead that still drew).
- **Search depth:** 128 sims did not help (draw share 0.70 to 0.69).
- **Value head:** it evaluates leads of a rook or more as about 84% draw.
- **K+Q vs K:** 1-6 mates in 64 games even at 256 sims.
- **Root cause:** there is no conversion signal at this scale.
- **Owner-approved fix:** MCTS-solver (D50) plus endgame curriculum (D51),
  measured separately.

## D50/D51 STAGE 1 + D52 (2026-09-29; details in docs/PHASE4_RESULTS.md)

- **D52 (fixed):**
  - Builds now pin CUDA 12.9 through `CUDARC_CUDA_VERSION`.
  - Every model build and load runs a known-answer device check. A dead JIT,
    which silently computed zeros, now refuses in 3 s.
- **Stage 1 conversion probes (MEASURED):**
  - Single major piece vs K: 1-2 of 64 conversions for both the trained
    and the untrained network.
  - Two majors vs K: 34-36 of 64.
- **D50 MCTS-solver:** no conversion gain; the pre-registered rule fails.
  Not adopted, and kept off by default.
- **D51 curriculum:** the heavy families qualify. The stage 2 pilot (heavy
  + target, no solver) is pending.
- **Gate:** dependencies are now optimized in test builds, so the gate takes
  about 3 minutes instead of 15.
- **Next:**
  - throughput pass T1 (Burn fusion and autotune, pre-registered)
  - then the stage 2 curriculum pilot

## D51 STAGE 2 + THROUGHPUT + HP COMPARISON (2026-09-29; details in docs/PHASE4_RESULTS.md)

- **Stage 2 curriculum pilot (6 cycles): NO-GO.**
  - P1 failed conversions: 0.448 against the 0.41 required.
  - P2 win estimate at a 5-8 lead: 0.110 against the 0.30 required.
  - P3 K+Q / K+R conversion: 1 / 64 against the 8 required.
  - The trained network converts two-major endgames worse (18 / 64)
    than the untrained one (36).
  - The drift came earlier and reached P4.6's level.
- **Throughput (D53):** self-play 22.8 -> 31.5 trainable pos/s (1.38x)
  and training 1.20x, from two owners, the 48 / 96 schedule and flattened
  linears (bit-exact). Fusion, autotune and TF32 are not adopted (TF32
  kernels never win autotune). T6 candidate buckets are implemented and
  pre-registered, not yet measured.
- **HP branch comparison** (`docs/HP_BRANCH_COMPARISON.md`):
  - **D54 ported:** arena players now search their own trees. Every earlier
    mainline arena mixed both networks and is labelled mixed-tree.
  - Non-finite outputs are no longer hidden, and the metrics race is fixed.
  - The branch independently confirms the conversion weakness.
- **Next: owner decision needed.** Proposed changes to break the draw loop,
  each a new identity:
  1. **Reverse curriculum from near-mate positions.** Curriculum starts
     where our own search can prove a short forced mate: the D50 solver at
     high sims as a filter, with no external labels. The distance then
     grows as conversion succeeds, so the curriculum produces mostly *won*
     labels, not drawn ones.
  2. **Endgame-appropriate exploration.** Curriculum games play argmax from
     ply 0 instead of 30 sampled plies.
  3. **Lower LR** (the HP finding: 7.5e-5 was best at 15M), as a
     pre-registered A/B.
  4. **`root_player_v1` arenas** (D54) for all new runs.

## CURRENT STATE AND NEXT STEPS (2026-09-30)

**Where we are (MEASURED).**

- **The learning loop works mechanically but does not learn to convert.**
  Across P4.6 and the D51 stage 2 pilot:
  - self-play drifts to 70-80% draws, 60-90% of them failed conversions;
  - the value head scores leads of a rook or more as about 83% draw;
  - neither the trained nor the untrained network converts K+Q or K+R vs K
    (1-2 of 64).
  - Training makes conversion worse: two-majors-vs-K goes from 36 / 64
    untrained to 18 / 64 after the curriculum pilot.
- **Tried and not adopted:**
  - 128 sims (the drift persists)
  - the D50 MCTS-solver (no conversion gain)
  - the D51 endgame curriculum (NO-GO: too small a dose, mostly drawn, so
    it taught "lead = draw")
- **Throughput (D53):** self-play 1.38x and training 1.20x over the day's
  start, execution-only or bit-exact.
  - Not adopted: fusion, autotune, TF32, candidate buckets (T1, T5, T6).
  - Run-to-run variance is about 9%, so the smaller gains are
    uncertain.
- **Correctness imported from the HP branch (D54):**
  - arenas now let each player search its own tree (earlier arenas were
    mixed-tree)
  - non-finite model output is refused
  - the metrics race is fixed

**Value-head diagnostics** (`recur64 value-diag`). Stopped by the owner:
DA complete, DB partial, DC and DD NOT RUN.

- **The value head is calibrated:** it predicts close to the
  position-weighted outcome rate. Even in heavy endgames only 23% of
  big-lead positions are in won games.
- **It learns fast:** held-out cross-entropy goes 1.10 -> 0.39.
- **It unlearns fast:** about 50 standard updates undo it.
- **So the bottleneck is conversion in the games** (search and policy),
  not value learning. Next steps should change the data, not the value
  head.

**Candidate next steps.** Each is a new identity needing an owner
decision. The diagnostics favour 1 and 2 over 3:

1. **Reverse curriculum from near-mate positions.** Starts are filtered by
   our own search proving a short forced mate; the distance to mate grows
   as conversion succeeds; games play argmax from ply 0. The goal is mostly
   *won* labels at a meaningful position share.
2. **Dose and exploration fixes** for any curriculum: heavy families only,
   a larger position share, argmax from ply 0.
3. **LR A/B** (the HP finding: 7.5e-5 was best at 15M). This is lower
   priority now: the value head tracks its data at any tested LR.
4. **Evaluation contract:** new runs use `root_player_v1` arenas (D54).
   Consider the HP paired-RNG arenas for less noisy promotions.
5. **Deferred:** the R10 recurrence study. The HP branch found no
   value-learning benefit from recurrence at 15M, and a baseline that
   learns is needed first.

## NOT RUN

- P4.6 bounded qualification and the P4.7 R10 entry decision.
- No 24h run is authorized.

## Historical evidence note

The Phase 3 F10 result above predates the seed and gradient fixes and head v2.
It is not a clean modern baseline, and its checkpoints are head v1. The HP
F15/R15 record lives on `experiment/hp-r15`; it is external evidence, not a
mainline result.


---

# Workstation V2.5 (branch `experiment/workstation-v25`; main is unchanged)

## COMPLETED
- candidate_v25 (27,469,204 params) and legacy_facts_v25 (26,810,584 params) architectures with
  distinct checkpoint identities; CandidateFactsV1 in core; evaluator/inference plumbing.
- ProofTargetsV1: exact mate solver, independent audit, exhaustive pools, sealed splits,
  heavy-family holdouts A/B/C (C unused), P25_DATA_V1 (44,332 positions).
- CUDA qualification (RTX 2000 Ada, FP32): guard, forward, learner layouts, lifecycle, facts cost.
- P1a/P1b LR screens (3e-4, a grid-boundary result), P2 (L/C0/CF x 2 seeds), P2.5-F (2x2 factorial
  with LF), P2.5-D (5x unique heavy data).

## VERIFIED (test evidence)
- fmt clean, clippy 0, 302 workspace release tests pass; Probe/F10 identity and the frozen P4.5
  hash unchanged; 3x3 architecture cross-refusal matrix; exact checkpoint round trips.
- CandidateFacts vs an independent reference on 41,353 positions; solver vs brute force;
  100% independent audit of every proof position used.

## FAILED (pre-registered gates, reported as failures)
- P2 Q3: CF M2 0.689 (floor 0.75). P2.5-D: unique-data scale signal not met (M2+M3 +0.005,
  CI includes 0) and the absolute M2 gate fails (0.662). LF did not meet its M1 >= 0.95 health
  expectation (0.74, underfit).

## NOT RUN
- P3 conversion, M4/M5, self-play, curriculum, any optimization-horizon (800-update) test
  (removed from scope by the owner), a fresh KQ/KR holdout (the exact small pools are fully
  partitioned), P4 scheduling/GPU-feed optimization (utilization ~85% noted for later).

## Next gate
Owner decision after reviewing `docs/WORKSTATION_V25_SUMMARY.md`. P3 is not authorized by the
pre-registered rules.


---

# V3 - Active learned search (branch `experiment/workstation-v3-active-search`)

Separate research line from `experiment/workstation-v25@feb86236`. Main and V2.5 are unchanged.
No science has been run: no HOLDOUT_C access, no P4 feasibility measurement, no TUNE training.
See `docs/V3_RESEARCH_PLAN.md` (pre-registered gates), `docs/V3_ARCHITECTURE.md`,
`docs/V3_BUILD_RESULTS.md`, `docs/V3_EXPERIMENTS.md` (ledger) and `docs/evidence/v3/`.

## COMPLETED (P0 to P3)

- V3-P0: lineage, architecture, research plan, pre-registered gates, decisions V3-D1 to V3-D9.
- V3-P1 / P1.1: `recur64-statequery`, an exact one-edge query tool that depends only on
  `recur64-core`, with an answer-free packet, semantic state identity, and a separate persistent
  query identity.
- V3-P2 / P2.1: the `active_search_v3` model (30,853,790 parameters): V2.5 CF root executed once,
  shared query-state encoder (4 heads, frozen), gated RMS-normalised planner with per-root-branch
  memory and a K=8 workspace, selector with a masked STOP logit, sparse root readout; strict identity
  with 11 versioned contracts and 4x4 architecture cross-refusal.
- V3-P3: real FP32 CUDA qualification (`recur64 v3-qual`).
- `recur64 model-info` reports the real active graph; every historical command refuses
  `active_search_v3` before any work.

## VERIFIED (test and measurement evidence)

- Full workspace release suite: 372 tests passed, 0 failed; fmt clean; clippy 0 warnings (default features). Historical V2.5 and mainline tests and pinned hashes are unchanged.
- StateQueryV1 against the authoritative `GameState` transition: 203,426 random-descent edges,
  105,670 fixture edges and a depth-3 BFS, all exact.
- Root encoder runs exactly once at every budget (measured by in-function counters); query-encoder
  and planner rows executed equal successful queries (compacted); forced budget spent exactly or the
  frontier is genuinely empty; invariants are checked before every `run()` returns.
- Every non-STOP parameter gets a finite non-zero gradient on update one; the masked STOP head gets
  exactly zero.
- CPU checkpoint resume is bit-exact (0e0 over every parameter and two further optimizer steps).
- Real FP32 CUDA: B0/2/4/8/16 with ACTIVE and FIXED, teacher-forced training updates, backward,
  AdamW, checkpoint save/load (difference 0.0), device known-answer guard, VRAM plateau for a
  resident model (150 inference calls flat at 2001 MB; 40 training updates flat at 1489 MB).
- Measured compute: the exact CPU query is at most 2.4% of wall; planner + selector are launch-bound
  at about 6 to 7 ms per round regardless of batch; B16/B0 wall is 5.6x to 16.1x depending on batch.

## FAILED / FOUND

- Found and fixed: inert planner key bias; CLI fall-through of `model-info` to the Probe graph;
  `v3-qual` accepting out-of-range budgets; a flaky gradient test (inherited inert key biases).
- Found, not fixed: repeated model build/drop grows VRAM by about 13 to 16 MB per cycle on this
  stack (reproduced on the V2.5 model; does not occur inside a run that builds its model once).
- Found, not fixed: `GameState::from_fen` accepts adjacent kings, after which `candidate_facts`
  panics (unreachable from legal play or solver-verified data).

## NOT RUN

- HOLDOUT_C: not loaded, hashed or evaluated.
- P4: ProofTraceV1, the P4 feasibility measurement `C_8(KQRvK M3)`, V3 TUNE generation.
- P5 to P11: LR screen, information-sufficiency control, primary training, confirmation, adaptive
  STOP, conversion transfer, self-play.
- BF16/TF32; fusion, graph capture or shape bucketing (separate execution-only experiments).

## BLOCKED

- Nothing blocks P4 technically. P4 and later need owner approval.

## Next gate

P3 engineering qualification is complete. Next is V3-P4 (data and process layer, including the
frozen feasibility rule) on owner approval.


## V3 P3.1 and P4 update (2026-10-01)

- **COMPLETED:** P3.1: an explicit, gated qualification verdict (11 independent gates, non-gating diagnostics,
  non-zero exit on failure, derived reclassification of the P3 evidence, a fresh hardened CUDA run that passes).
  P4: custody of P25_DATA_V1 TRAIN and HOLDOUT_C (sealed, `evaluated = false`), `v3_tune_v1` (4,500 positions,
  audited, disjoint, regenerated), `proof_trace_v1` (exact `Q*`, set-valued `A(S)`, independent audit of all
  44,332 TRAIN traces and 4,500 TUNE traces), and the frozen feasibility measurement.
- **VERIFIED:** `C_8(KQRvK M3) = 2168 / 5000 = 0.4336`, threshold 0.25, classification
  SCIENTIFICALLY QUALIFIED FOR THE B8 PRIMARY EXPERIMENT. Full workspace release suite and clippy clean.
- **NOT RUN:** P5 and later, any model evaluation, HOLDOUT_C evaluation. HP/X1/X2 disjointness is not verified
  (datasets unavailable locally).
- **Next gate:** P5 needs a new owner instruction.

### V3 P4.1 / P5 (pre-measurement, 2026-10-01)

P5 infrastructure (`recur64_runtime::p5`, `recur64 v3-p5`) and the pre-registered plan
(`docs/V3_P5_PLAN.md`, V3-D16 to V3-D19) are in place. No LR-screen result exists. P6 is not authorised.

### V3 P5.1 (2026-10-01)

- **State:** one P5 screen attempt (LR 7.5e-5, seed 5101) trained to update 150 and was externally interrupted. Only its
  update-0 TUNE baseline (`S_run` 3.5533) was evaluated; no trained TUNE checkpoint was evaluated. That attempt is
  voided operationally, excluded from selection, and quarantined (V3-D20, V3-E14). The clean six-run screen has not
  produced results.
- **COMPLETED:** P5.1 integrity patch (strict resume/sidecar consistency, fail-closed run directories, persisted sampler
  exposure and run provenance, strengthened summary validation, same-seed update-0 pairing check, fail-stop launcher).
- **VERIFIED:** contract digest unchanged (`105ac313...009d2`, micro 16 x accum 8); corruption/validation tests, the
  full workspace release suite, fmt and clippy pass; the CUDA release binary builds.
- **NOT RUN:** the six clean screen runs, selection, P6, HOLDOUT_C (sealed, unevaluated).

### V3 P5 complete (2026-10-01)

- **COMPLETED:** clean six-run P5 LR screen (all uninterrupted), validation, same-seed pairing check (bitwise identical),
  and the frozen selection (applied once): peak LR **3.0e-4**, `S_lr` 1.5098 vs 1.5570 and 1.6964.
- **VERIFIED (TUNE diagnostics, not gates):** ACTIVE CE is not below B0 at B2/B4/B8; teacher-forced top-1 about 0.99; ACTIVE proof
  completion lags the ideal ceiling at larger budgets. Details: `docs/V3_P5_RESULTS.md`.
- **NOT RUN:** P6, Gate I/II/III, B16, DAgger, HOLDOUT_C (sealed, unevaluated).
- **Next gate:** owner review. Open items for approval: P6, a DAgger-style rescue, and profiling/optimisation before long runs.

### V3 P6 pre-registered (2026-10-01)

- **COMPLETED:** P5.2 diagnostic code; `all_info_v1` implementation and tests; baseline replication seed 5103; TRAIN-only state
  census and CUDA preflight (micro16 x accum8); `docs/V3_P6_PLAN.md` and V3-D22 to V3-D25 committed before any ALL-INFO TUNE
  evaluation.
- **NOT RUN:** ALL-INFO training and evaluation, Gate I, DAgger, P7, HOLDOUT_C (sealed, unevaluated).

### V3 P6 complete (2026-10-01)

- **COMPLETED:** P5.2 refined selector diagnostic and query-content ablation; three ALL-INFO runs; B0 references (three paired
  seeds); Gate I applied once.
- **VERIFIED (TUNE):** **Gate I PASS** - `Delta = +0.2351` (CI [0.2044, 0.2671], threshold +0.20) on KQRvK M3 top-1; ALL-INFO
  0.692 vs B0 0.457. Raw future-state information is sufficient for this model family on V3_TUNE_V1 (narrow claim). Teacher-forced
  P5 accuracy is explained by query-pattern leakage (ablation). Details: `docs/V3_P6_RESULTS.md`.
- **NOT RUN:** DAgger, P7, Gate II, Gate III, B16, HOLDOUT_C (sealed, unevaluated).
- **Next gate:** owner decision on the one pre-registered DAgger rescue; separately, the throughput/overhead pass (V3-D27).
