# Recur64 — Decisions (Phase 0)

Architecture decision records. Status values: **ACCEPTED**, **PENDING**,
**REJECTED**, **DEFERRED**.

## D1 — Burn 0.21.0 stable as the framework

- **Status:** ACCEPTED
- **Decision:** Pin `burn = "=0.21.0"` (and `burn-flex`, `burn-cuda` when
  enabled). Commit `Cargo.lock` and `rust-toolchain.toml` (Rust 1.97.1).
- **Why:** 0.21.0 is the latest stable release with a stable `burn-cuda`; the
  0.22 line is prerelease. Stable pins are more reproducible.
- **Revisit if:** the graph cannot run or performs poorly and a specific 0.22
  prerelease fix is demonstrably required. Re-evaluation is a separate decision.

## D2 — Native Windows first; WSL2 as fallback

- **Status:** ACCEPTED
- **Decision:** Evaluate native Windows + Burn/CUDA first (tree A). Fall back to
  a fresh Ubuntu under WSL2 (tree B) only if native is blocked or materially
  inferior. Do not maintain both.
- **Why:** Rust builds work natively here (validated), avoiding a second
  toolchain. WSL2 remains available but its only current distro is an unfamiliar
  `BendExp`; a fresh Ubuntu would be used.
- **Evidence so far:** native Windows CPU FP32 graph passes all correctness gates.

## D3 — CUDA runtime via user-space redistributables (no admin)

- **Status:** ACCEPTED (implemented and verified)
- **Decision:** Because there are no administrator rights, do not use the CUDA
  Windows installer. Extract the CUDA **12.9.1** redistributable component
  archives (`cuda_cudart`, `cuda_nvrtc`, `cuda_nvcc`, `libnvjitlink`,
  `libcublas`) into `%LOCALAPPDATA%\Recur64\cuda\12.9.1` and set `CUDA_PATH` and
  `PATH` **for the `recur64` process only**. No PATH/registry/system changes.
- **Why:** `burn-cuda` requires CUDA 12.x on `PATH`; the display driver (596.71)
  is already present, so only user-space runtime libraries were missing.
- **Evidence:** `burn-cuda`/`cubecl-cuda`/`cudarc` compile and link against the
  extracted runtime; `recur64 cuda-smoke` passes on the RTX 2000 Ada (FP32
  forward R=1/2/4, backward, AdamW, checkpoint restore). See `BENCHMARKS.md`.
- **Result:** native Windows CUDA works; WSL2 (D2/B) is not required.

## D8 — CUDA FP32 is the accepted GPU backend

- **Status:** ACCEPTED
- **Decision:** Use `burn-cuda` (Burn 0.21.0) on native Windows for the Phase 0
  GPU path. No fallback to tch-rs or Candle was needed.
- **Evidence:** CUDA smoke PASS; synchronized R10 benchmark at batch 1–128 and
  R=1/2/4 with finite outputs; checkpoint restore on GPU.
- **Not claimed:** BF16 support, GPU bit-exact determinism, or full-training-run
  stability. BF16 remains refused until the full graph is tested.

## D4 — CPU FP32 correctness baseline via Flex

- **Status:** ACCEPTED
- **Decision:** All correctness work runs first on `Autodiff<Flex>` (pure-Rust
  CPU). GPU evidence is added only after the CPU graph passes.
- **Why:** No installs required; deterministic; isolates model/contract bugs from
  backend/GPU bugs.

## D5 — Minimal two-crate workspace

- **Status:** ACCEPTED
- **Decision:** Only `recur64-model` and `recur64-cli`. `recur64-core`,
  `recur64-search`, `recur64-runtime`, `recur64-eval` are deferred to the phases
  that need them.
- **Why:** Avoids fake scaffolding for hypothetical future needs.

## D6 — Eager parameter initialization

- **Status:** ACCEPTED
- **Decision:** `ProbeModel::new` force-initializes all parameters.
- **Why:** Burn 0.21 lazily initializes parameters; cloning an uninitialized
  module copies the deferred initializer and re-samples on first access. This
  silently broke value-preserving clones and made a resume test diverge. With
  eager init, CPU FP32 resume is bit-exact.
- **Consequence:** any future module clone in a resume path must ensure
  parameters are materialized first.

## D7 — No Python trainer, no custom autodiff, no custom CUDA kernels

- **Status:** ACCEPTED
- **Decision:** Training is Rust. Any deviation requires its own ADR.

## D9 — cozy-chess for rules and move generation

- **Status:** ACCEPTED (Phase 1)
- **Decision:** Pin `cozy-chess = "=0.3.4"` (MIT) as the sole production chess
  dependency. Do not switch to a higher-perft crate.
- **Why:** correctness, maturity, MIT license, stable API, and the `util`
  UCI converters. Its internal king-captures-rook castling is contained at the
  `uci` boundary; `Board::same_position` is the FIDE repetition authority.
- **Verified:** `crates/recur64-core/tests/cozy_api.rs` pins the exact behaviors
  Recur64 relies on.

## D10 — shakmaty as an optional, dev-only differential oracle

- **Status:** ACCEPTED (Phase 1)
- **Decision:** `shakmaty` 0.30.1 is an **optional** dependency behind the
  non-default `oracle` feature, used only in tests for legal-move-set and
  mate/stalemate comparison. It is never in the production runtime.
- **Why:** it is GPL-3.0-or-later; keeping it optional and off by default avoids
  any distribution obligation while still providing independent validation.
- **Mandatory independent validation** remains published CPW perft counts plus
  hand-verified fixtures.

## D11 — En-passant observation is FEN-style

- **Status:** ACCEPTED (Phase 1)
- **Decision:** Observation V1's EP indicator is set after any double pawn push
  (FEN semantics), regardless of whether a legal EP capture exists. The
  repetition key instead uses the stricter FIDE notion via `same_position`.
- **Why:** literal reading of the observation spec; avoids a per-position
  legality query in the encoder. The two notions are documented separately.

## D12 — Termination precedence and auto-claim draws

- **Status:** ACCEPTED (Phase 1)
- **Decision:** Precedence is checkmate, stalemate, insufficient material,
  threefold, 50-move, truncated. Threefold and 50-move are **auto-claimed on the
  current position**. `Truncated`/`Aborted` are not results.
- **Why:** checkmate must never be overwritten; the training convention must be
  identical everywhere. See `RULES_PROFILE.md`.

## D13 — `recur64-core` boundary

- **Status:** ACCEPTED (Phase 1)
- **Decision:** New CPU-only, Burn-free crate owning the chess contracts.
  Dependency direction is `core ← model ← cli` and `core ← cli`; no cycles.
- **Why:** the chess world must be testable and fast without CUDA/Burn, and
  Phase 2's self-play/replay will consume it directly.

## D14 — Model re-exports core action constants

- **Status:** ACCEPTED (Phase 1)
- **Decision:** `recur64-model::action` delegates to `recur64-core` for
  `PROMO_*`, `SQUARES`, `ACTION_SPACE`, `action_id`, `decode_action_id`, while
  preserving the Phase 0 public API. A drift-guard test checks equality over the
  full 20,480 space.
- **Why:** single source of truth; the Phase 0 test suite is the regression gate.

## D15 — Checkpoint contract metadata deferred to Phase 2

- **Status:** ACCEPTED (Phase 1)
- **Decision:** Phase 1 does not modify `CheckpointMeta`. Contract version
  constants exist in `recur64-core::schema`; Phase 2 will add optional
  `#[serde(default)]` fields and refuse to resume on a mismatch.
- **Why:** avoids Phase 0 checkpoint churn; no chess checkpoints exist yet.

## D16 — PUCT is the only Phase 2 search

- **Status:** ACCEPTED (Phase 2)
- **Decision:** Implement PUCT with an explicit formula, perspective-safe backup,
  deterministic tie-breaks, and an exact traversal budget. Gumbel is deferred.
- **Why:** correctness and system integration before search sophistication. See
  `docs/SEARCH.md`.

## D17 — One GPU inference owner; no direct CUDA from workers

- **Status:** ACCEPTED (Phase 2)
- **Decision:** A single owner thread holds the Burn backend; workers submit
  single-position requests through a bounded channel and receive exactly one
  response each. Batching is bounded by `max_inference_batch` and
  `batch_timeout`.
- **Why:** avoids intra-tree races, fills batches across independent games, and
  guarantees no caller blocks forever.

## D18 — Replay stores moves, not observations

- **Status:** ACCEPTED (Phase 2)
- **Decision:** A game stores its start FEN and selected canonical actions; every
  position and Observation V1 is reconstructed on read. Targets are sparse.
- **Why:** compact, auditable, and impossible to silently misalign: an illegal
  target action is a hard error. See `docs/REPLAY.md`.

## D19 — Truncated/aborted games are excluded from the learner

- **Status:** ACCEPTED (Phase 2)
- **Decision:** Games without a result are stored with their termination reason
  but excluded from training; they are never labelled draws.
- **Why:** the simplest safe policy; no loss change. Policy-only training is
  deferred.

## D20 — Checkpoint schema v2 records chess contracts and model identity

- **Status:** ACCEPTED (Phase 2)
- **Decision:** `SCHEMA_VERSION = 2`; `CheckpointMeta` records observation/action/
  rules/replay versions, `model_id` (SHA-256 of the saved weights), `run_id`, and
  counters. v1 probe checkpoints fail visibly; a contract mismatch is refused.
- **Why:** a checkpoint must know which chess world it belongs to.

## D21 — Self-play game loop lives in `recur64-search`

- **Status:** ACCEPTED (Phase 2)
- **Decision:** `play_game_from`/`play_game_seeded` and the sampling RNG live in
  `recur64-search`, not the runtime.
- **Why:** the arena (`recur64-eval`) must drive games without depending on the
  Burn-backed runtime; this keeps the dependency graph acyclic
  (`core → search → eval → runtime → cli`).

## D22 — Deterministic RNG, no external rand dependency

- **Status:** ACCEPTED (Phase 2)
- **Decision:** a small in-crate SplitMix64 provides reproducible move sampling.
- **Why:** exact reproducibility from a recorded seed without an extra crate.

## Version pins

| Component | Pin |
|---|---|
| Rust | 1.97.1 (`rust-toolchain.toml`) |
| burn | =0.21.0 |
| cubecl (transitive) | 0.10.0 |
| cozy-chess | =0.3.4 (MIT) |
| shakmaty | 0.30, optional `oracle` feature (GPL-3.0, dev-only) |
| proptest | 1 (dev-dependency) |
| bincode | 2 (serde feature) |
| crc32fast | 1 |
| sha2 | 0.10 |
| ctrlc | 3 |
| serde / serde_json / toml / anyhow / clap | caret, locked by `Cargo.lock` |

## Rejected / deferred

- **tch-rs**, **Candle**: deferred fallbacks (see `ARCHITECTURE.md`).
- **0.22.0-pre.x Burn**: deferred until a stable release or a demonstrated need.
- **Docker, Python trainer, custom CUDA kernels**: rejected.
