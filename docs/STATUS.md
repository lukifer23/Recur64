# Recur64 — Status (Phase 0)

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
