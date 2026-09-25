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

## D23 — F10 baseline contract (Phase 3)

- **Status:** ACCEPTED
- **Decision:** The control baseline is F10 (9,805,288 params, R=1), PUCT only,
  self-play only, random init, FP32, standard start. Frozen: `c_puct = 1.0`,
  `simulations_per_move = 64`, exploration temperature 1.0 for all plies with no
  noise. No Gumbel / R10 / diffusion / geometric bias / auxiliary heads /
  external labels / BF16.
- **Why:** a boring, trustworthy control for all later research.

## D24 — `active_games` is the concurrency

- **Status:** ACCEPTED
- **Decision:** Self-play runs one game per thread; `active_games` concurrent
  games means `active_games` threads. Phase 2 bounded concurrency by
  `cpu_workers` and produced batch size 1.
- **Why:** each game has one outstanding leaf request; concurrency is what fills
  GPU batches. Measured batch mean rose from 1.0 to 20–34.

## D25 — Streaming replay sampler + capacity archiving

- **Status:** ACCEPTED
- **Decision:** The learner samples on demand from a `ReplayStore` that keeps
  compact `GameRecord`s (no observations) and reconstructs each example by
  replaying the game to the sampled ply. Replay capacity archives oldest shards
  to `replay/archive/` (never deletes).
- **Why:** the Phase 2 learner materialized ~30 KB/example; 100k positions would
  be ~7.5 GB.

## D26 — Gradient accumulation and warmup+cosine schedule

- **Status:** ACCEPTED
- **Decision:** Effective batch = physical batch × accumulation steps
  (default 64 × 4). LR schedule: linear warmup then cosine decay over the whole
  run's `planned_updates`; schedule position is checkpointed and never reset on
  resume.
- **Why:** the master plan's effective-batch and schedule intent, with bounded
  VRAM.

## D27 — Frozen Recur64-generated opening suite

- **Status:** ACCEPTED
- **Decision:** `configs/openings-v1.toml` holds deterministic legal prefixes
  generated by Recur64 from a recorded seed, used only for evaluation (paired
  colors), never for self-play.
- **Why:** self-contained provenance; no external corpus.

## D28 — Conditional GO on the 24h baseline

- **Status:** ACCEPTED
- **Decision:** Phase 3 is **CONDITIONAL GO**. The system works end-to-end, but
  learning health is poor (unstable loss, reuse far below target, repetition-
  dominated search, raw policy ≤ random). Fix reuse/update scaling, training
  stability, and repetition-dominated search, then re-run a bounded pilot before
  any ~24h run. No ~24h run without explicit owner approval.
- **Why:** an honest negative/weak result is a valid research outcome; a
  contaminated or misleading long run is not.

## D29 — Collection semantics: games, concurrency, workers

- **Status:** ACCEPTED (Phase 4)
- **Decision:** One authoritative `collect_parallel` serves `run`, `selfplay`,
  `pilot`, and the runtime sweep. New configs set `games_per_cycle` (total
  games) and `concurrent_games` (simultaneous games); `cpu_workers` caps the
  worker threads and the game count caps concurrency. Legacy configs (neither
  new field present) keep the Phase 3 D24 semantics exactly: `active_games` is
  both the total and the concurrency and `cpu_workers` is not applied. Both
  fields must be set together; invalid FENs and failed games are hard errors.
- **Why:** removes three divergent collectors and a sequential-only
  `collect_only`, while preserving the behavior of the historical
  `configs/f10-*.toml` runs so their identities are not silently mutated.

## D30 — Example-weighted mean gradient reduction

- **Status:** ACCEPTED (Phase 4)
- **Decision:** Each microbatch loss is scaled by its example count before
  backward; the accumulated gradient is divided by the actual number of examples
  in the update; reported loss/entropy are example-weighted aggregates.
- **Why:** the Phase 3 learner summed microbatch gradients without normalizing
  and reported only the final microbatch, which inflated gradient norms and made
  loss readings misleading. The optimizer contract now states
  `grad_reduction=example_weighted_mean_over_effective_batch`.

## D31 — Optimizer continuation and accepted-trajectory promotion

- **Status:** ACCEPTED (Phase 4)
- **Decision:** The pilot loads the parent through `load_training` (weights,
  Adam moments, schedule step) into a freshly built module and continues only if
  the candidate is promoted. A held candidate leaves the accepted optimizer step
  unchanged. The content hash proves the reload preserved `ParamId`s.
- **Why:** a fresh AdamW per cycle loses the optimizer state and advances the
  "accepted" trajectory on rejected candidates; neither is scientifically
  defensible.

## D32 — Conservative-v2 promotion

- **Status:** ACCEPTED (Phase 4)
- **Decision:** `PROMOTION_RULE_VERSION = conservative-v2`. Promotion requires
  audit OK, zero inference errors, finite metrics, achieved reuse >= 0.8x target,
  at least `promotion_min_decisive_games` decisive candidate-vs-parent games, and
  a score strictly above 0.5 (and at least the configured floor). Decisions are
  `promote` or `hold` with hold reasons.
- **Why:** the old rule (score >= 0.35) could promote a draw-only 0.5 candidate.

## D33 — Scientific vs resolved identity

- **Status:** ACCEPTED (Phase 4)
- **Decision:** Every run records a `scientific_config_hash` over the experiment
  (model, recurrence, precision, reference id, seed and seed policy, search,
  games per cycle, optimizer contract, effective batch, LR schedule, reuse
  target, replay capacity and sampler, opening-suite content digest, arena
  games, promotion rule) and a `resolved_config_hash` over the entire resolved
  config. Operational scheduling (device, concurrency, cpu_workers, batch and
  timeout, labels, run id, budgets, the `max_updates` safety cap) changes only
  the resolved hash.
- **Why:** a hardware re-tune must not look like a new experiment, and a science
  change must never be hidden behind the same hash.

## D34 — Frozen reference checkpoints

- **Status:** ACCEPTED (Phase 4)
- **Decision:** `recur64 freeze-reference` writes one seeded, untrained,
  step-0 checkpoint plus `reference.json` (model id, seed, git, both hashes,
  suite digest, optimizer contract). The pilot refuses a mismatched reference id
  or config or a trained checkpoint. Scheduling cells, search cells, the T0
  baseline, smoke and qualification all start from the same checkpoint.
- **Why:** regenerating random weights per benchmark cell makes throughput and
  data incomparable.

## D35 — Build-time git provenance

- **Status:** ACCEPTED (Phase 4)
- **Decision:** `recur64-runtime/build.rs` bakes `RECUR64_GIT_SHA` /
  `RECUR64_GIT_BRANCH`, appends `-dirty` when crates/manifests differ from HEAD,
  and watches refs and sources. Run metadata and lineage record the SHA, branch
  and seed.
- **Why:** Phase 2 always wrote `git_revision: None`; a run must name its source.

## D36 — CUDA NVRTC fail-fast

- **Status:** ACCEPTED (Phase 4)
- **Decision:** Every CUDA run path refuses to start when no NVRTC shared
  library is present on `PATH` or `CUDA_PATH\bin`. The check matches any
  `nvrtc*.dll` / `libnvrtc.so*` name (cudarc tries several versioned names).
- **Why:** without NVRTC the JIT can panic on a worker thread and leave a
  misleading checkpoint behind. Matching any version avoids refusing a valid
  user-space runtime over an exact-name mismatch.

## D37 - Crash-safe replay archival

- **Status:** ACCEPTED (implemented 2026-09-25)
- **Decision:** capacity enforcement runs in this order:
  1. Copy each shard to `archive/` (write `.tmp`, fsync through a writable
     handle, rename). An existing archive copy is reused only if it passes the
     checksum.
  2. Fsync the archive directory (Unix; NTFS journals metadata).
  3. Replace the manifest atomically (temp + fsync + rename), then fsync the
     directory.
  4. Only then remove the old active files.
  - A shard that fails the checksum is never archived.
  - Recovery removes an unreferenced active shard only when a verified archive
    copy exists. Archived data is never deleted.
- **Why:** the previous rename-then-rewrite order could leave the manifest
  pointing at moved shards after a power loss. This is required before any
  long run.
- **Test:** `capacity_archival_is_crash_safe_at_every_step` simulates a crash
  after copying (before the manifest) and a crash after the manifest (before
  removal), and checks that unarchived data is never deleted.
- **Found while testing:** on Windows, `sync_all` requires write access
  (`FlushFileBuffers`), so a file opened read-only failed with "Access is
  denied".

## D38 - Deadline semantics

- **Status:** ACCEPTED (complete for learner and evaluation, 2026-09-25).
- **Decision:**
  - A configured wall budget has a soft deadline: stop starting new work at a
    safe boundary and let in-flight steps finish.
  - The learner checks the deadline at update boundaries. Self-play
    collection checks it between games.
  - Every evaluation game scheduler (`play_indexed_until`: searched arenas,
    raw-vs-random, raw-vs-parent, the pilot's T0) checks it at game
    boundaries. Once it passes, no new evaluation game starts and in-flight
    games finish.
  - An incomplete evaluation is an explicit `EvalError::DeadlineExceeded`,
    never a partial result.
  - The pilot turns it into a held cycle: reason `evaluation_deadline`,
    status `budget_exhausted_during_eval`, a partial cycle report is written,
    and the run stops.
  - A budget overrun is recorded (`budget_overrun_secs`), never hidden.
- **Why:**
  - The Phase 3 pilot checked deadlines only between cycles and could exceed
    its nominal budget during a long evaluation phase.
  - A partial arena must never inform promotion.
- **Tests:**
  - `arena_past_deadline_is_an_explicit_incomplete_error` (sequential and
    threaded schedulers)
  - `evaluation_past_the_deadline_is_a_typed_incomplete_error` (pilot
    evaluation path)

## D39 — Replay freshness control

- **Status:** PROPOSED (Phase 5)
- **Decision:** Replace the single ambiguous `replay_reuse_target` with a
  `new_data_exposure_target` (how often each newly generated position is seen)
  plus a `history_mixture_fraction`. The Phase 4 sampler recency weighting is
  unchanged; only the measurement (`current_cycle_sample_fraction`,
  `mean_sample_age_cycles`) ships now.
- **Why:** `examples_consumed / new_positions inserted` near 2 does not prove
  the new data was seen twice. Measure first, then change the sampler behind its
  own experiment.

## D40 — Readout head v2 (near-uniform policy and neutral value at init)

- **Status:** ACCEPTED (Phase 4, owner-approved 2026-09-24)
- **Decision:**
  - A final RMSNorm sits before the policy and WDL heads.
  - The bilinear policy logits are scaled by `1/sqrt(policy_dim)`.
  - The WDL head is zero-initialized.
  - Versioning: `HEAD_VERSION = 2`, recorded in every checkpoint. Checkpoints
    without the field are head v1. A mismatch is refused on every load path,
    including `model_io::load`.
  - Applies identically to F10 and R10. Unique parameters become 9,805,672
    for both.
- **Why:**
  - A fresh F10 under head v1 had policy entropy 0.50 × uniform and mean
    |value| 0.245 (MEASURED).
  - PUCT with an uninformative value head reproduces the prior, so self-play
    distilled the arbitrary initialization. Search gain fell from 8 to 64 sims
    (MEASURED).
  - The raw residual stream made the head-input scale depend on the block
    layout, a potential F10-vs-R10 confound.
  - Head v2 measures 0.999 × uniform and |value| 0.000
    (`tests/t0_prior.rs`). A fresh R10 at R1/R2/R4 measures 1.000 × uniform
    and |value| 0.000, matching F10.
- **Expected zero-init consequence:** at optimizer step 0 the WDL head
  weights are zero. The first update trains the WDL head itself, while the
  WDL gradient into the trunk is zero on that first backward pass. This is
  intended, and is changed only if measured learning shows a harmful
  value-head delay. The promotion head keeps its default init: zero-init
  there broke the existing promotion-gradient test, and no promotion
  pathology has been measured.
- **Consequence:**
  - The v1 reference (`7d1493b4…`) and all Phase 3 checkpoints are head v1
    and are refused under v2.
  - The v1 P4.4 curve is kept as superseded evidence.
  - The P4.3 hardware schedule is kept, because it is hardware-only.

## D41 — Self-play exploration: root Dirichlet noise and argmax after ply 30

- **Status:** ACCEPTED (Phase 4, owner-approved 2026-09-24). This lifts the
  earlier deferral of Dirichlet exploration for the mainline self-play
  contract.
- **Decision:**
  - Self-play mixes `Dirichlet(alpha)` noise into the **root** priors:
    `prior' = (1-eps)·prior + eps·noise`. The mainline F10 values are the
    AlphaZero chess ones: `root_dirichlet_alpha = 0.3`,
    `root_dirichlet_epsilon = 0.25`.
  - Move selection samples at `temperature` until `argmax_after_ply`
    (mainline: 30), then plays the highest-visit move.
  - Arenas are noise-free and deterministic by contract.
  - All three fields are part of scientific identity (v4).
  - The noise is drawn from the project's deterministic RNG (D22).
- **Why:** nothing broke the prior → target → prior loop (D40), and sampling
  every ply at temperature 1.0 kept games noisy to the end.
  `argmax_after_ply` also existed as a config/identity field that self-play
  never read (a dead field); it is now implemented.

## D42 — Search gain is a learning-progress metric, not a T0 gate

- **Status:** ACCEPTED (Phase 4, owner amendment A2; post-hoc, recorded as
  such)
- **Decision:**
  - `recur64 search-gain` re-evaluates replay positions with the generating
    network and reports KL(target‖prior) and the argmax-change fraction.
  - These are measured per learning cycle against that cycle's snapshot.
  - They do not gate the budget selection on an untrained reference.
- **Why:**
  - On an untrained network, search cannot improve on the prior, so no budget
    can pass such a gate.
  - Visit-count quantization also inflates KL at low budgets.
  - The metric remains the right signal once the value head learns.
- **Amendment (Phase 4 addendum A1):** under D41 root noise, the offline
  statistic compares the visit target with the **raw** network policy, so it
  includes exploration noise as well as tree-search movement.
  - It is named **`network_to_target_divergence`**. Historical JSON field
    names are kept for comparability.
  - The exact split is measured during self-play from the priors PUCT
    actually used (`root_search`): network → noisy root prior (noise alone),
    and noisy root prior → visit target (search movement after noise). It is
    aggregated before serialization, so Replay V1 is unchanged.
  - Noisy → target is the cleaner search-contribution signal. Neither metric
    is monotone by construction, and neither is a smoke gate yet.

## D43 — Inference-only commands evaluate on the inner backend

- **Status:** ACCEPTED (Phase 4)
- **Decision:** Inference-only paths (`eval-policy`, arena, pilot owners,
  sweep, search-gain) load models on `B::InnerBackend`, never on the
  autodiff backend.
- **Why:** `eval-policy` on `Autodiff<Cuda>` drove the 16 GB device to
  16,005 MiB and grew host RSS by 1.5 GB in about a minute, because every
  forward recorded graph state that no backward pass consumed (MEASURED).
  After the fix it held a flat 485 MiB.

## D44 - Inference owners release their thread's device memory on shutdown

- **Status:** ACCEPTED (Phase 4, P4.4L)
- **Decision:** `BatchedModel` calls `B::memory_cleanup(device)` in `Drop`.
  `Drop` runs on the owner thread, so it releases that thread's CubeCL stream
  memory pool.
- **Why:**
  - CubeCL 0.10 keys device streams and their memory pools by OS thread (up
    to 128 streams). Each `InferenceOwner` runs on a fresh thread that exits
    at shutdown, so its pool was orphaned.
  - MEASURED: VRAM grew 453 -> 10,459 MiB over 32 owner lifecycles, with
    stable latency and no errors.
  - After the fix, the same probe plateaus at about 0.9-1.0 GB.
- **Consequence:**
  - Long pilots no longer accumulate device memory per cycle.
  - Reusing persistent owner threads remains a possible later optimization,
    to avoid re-warming pools; it is not needed for correctness.
  - Owner residency (at most 2 resident owners) is a separate peak-memory
    item, scheduled after the smoke.

## D46 - At most two resident models during candidate evaluation

- **Status:** ACCEPTED (2026-09-25, owner-approved post-smoke item)
- **Decision:** `evaluate_candidate` keeps the candidate resident and runs
  two phases.
  - **Phase A:** parent + candidate play the searched parent arena, raw vs
    random and raw vs parent. The parent is then shut down.
  - **Phase B:** only when reference != parent, the reference is loaded for
    the longitudinal arena.
  - When parent == reference, no third model is loaded and the parent arena
    is reused (labelled).
- **Why:**
  - Three models were resident even when the third was identical to the
    parent. After D44 this no longer leaked memory; it only wasted it.
  - Each match is independent and deterministic given its seed and inputs, so
    the schedule does not change any result.
- **Test:** `two_owner_evaluation_matches_the_three_owner_sequence` checks
  identical results against the original three-owner sequence, both with
  parent == reference and with parent != reference.

## Rejected / deferred

- **tch-rs**, **Candle**: deferred fallbacks (see `ARCHITECTURE.md`).
- **0.22.0-pre.x Burn**: deferred until a stable release or a demonstrated need.
- **Docker, Python trainer, custom CUDA kernels**: rejected.
