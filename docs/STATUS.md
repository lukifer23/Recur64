# Recur64 — Status

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

## NEXT (owner decision)

- P4.6 bounded qualification, with the self-play draw share as an explicit
  monitored health metric.
- Remaining owner-approved item: cross-cycle tail waste.
- The 48-way concurrency lead is unconfirmed; K=2 now gives 64 leaves in
  flight.

## NOT RUN

- P4.6 bounded qualification and the P4.7 R10 entry decision.
- No 24h run is authorized.

## Historical evidence note

The Phase 3 F10 result above predates the seed and gradient fixes and head v2.
It is not a clean modern baseline, and its checkpoints are head v1. The HP
F15/R15 record lives on `experiment/hp-r15`; it is external evidence, not a
mainline result.
