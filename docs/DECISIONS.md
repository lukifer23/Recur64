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

## D45 - Searched-arena exploration contract

- **Status:** ACCEPTED (2026-09-25; owner-approved Option A, selected by the
  pre-registered rule)
- **Decision:**
  - Searched arenas sample from visit counts for the first 30 plies after
    the opening (`arena_sample_plies = 30`).
  - They apply root Dirichlet noise to every move (`arena_root_dirichlet_epsilon
    = 0.25`, alpha as in self-play), with paired colours and seeded games.
  - The defaults (none / 0) reproduce the original deterministic arena, and
    the fields enter the scientific identity only when set, so earlier
    hashes are unchanged.
- **Why:**
  - The deterministic, noise-free arena was 75% threefold between
    near-identical networks, which starved conservative-v2 (3-5 decisive of
    32; both smoke cycles held).
  - Measured on the same model pair: V0 had 3 decisive and 24 threefold, V1
    (sampling only) had 8 and 19, V2 had 24 and 0.
- **Consequence:**
  - Arena scores have wider intervals (±0.15 at 32 games), so promotion needs
    a clear margin.
  - Arena games no longer measure noise-free argmax play. Both sides get
    identical exploration, so the comparison stays symmetric.

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

## D47 - Multi-leaf PUCT with virtual loss (throughput)

- **Status:** ACCEPTED at K = 2 for the mainline F10 contract (owner
  decision 2026-09-25, after measurement).
  - The code default stays K = 1, which reproduces every earlier identity.
  - Measured on the same seeds: K = 2 gave +83% trainable positions/s
    (12.4 to 22.8), and K = 4 gave +115%, with unchanged data health.
  - K = 2 is the pre-registered pick (the smallest K clearing +10% with
    healthy data).
- **Decision:**
  - Each search round selects up to K leaves. Every edge on a selected path
    gets a provisional visit and a virtual loss (w -= 1), so later selections
    in the round prefer other lines.
  - All K leaves go to the evaluator as one `evaluate_many` submission. The
    batcher queues all of them before waiting, and the arena router splits
    them by side. They are then expanded and backed up, and each virtual visit
    becomes the real visit with the virtual loss removed.
  - An edge that is visited but has no child can only be a leaf pending in the
    same round; hitting one ends the round (collision).
  - The traversal budget is exact.
  - K = 1 runs the original recursive search unchanged.
  - K > 1 enters the scientific identity (`search_execution`) only when
    enabled, so every earlier identity stays reproducible.
- **Why:**
  - Each search thread had one evaluation in flight, so batches were capped
    by the number of games.
  - More OS threads oversubscribe the 24 cores (P4.3).
  - The forward is launch-bound (~13-15 ms almost independent of batch
    size), so larger batches should raise eval/s substantially.
- **Tests:**
  - `multi_leaf_search_keeps_budget_batches_and_clears_virtual_loss`: exact
    budget; batched submissions of at most K; all virtual loss removed; one
    evaluation per non-terminal node.
  - `multi_leaf_search_still_prefers_mate_in_one`.
  - `evaluate_many_shares_a_batch_and_answers_in_order`.
  - The pre-D47 smoke hash is still reproduced.

## D48 - Continuous trainer across held cycles

- **Status:** ACCEPTED (owner decision 2026-09-25).
  - Config: `trainer_policy = "continuous"`.
  - The default `discard_held` keeps D31, and the field enters the scientific
    identity only when set to continuous.
- **Decision:**
  - The learner trains each cycle from its own previous state
    (`checkpoints/trainer`: weights plus Adam state), whether or not the
    previous candidate was promoted.
  - The optimizer step and the LR schedule advance continuously, and the
    trajectory check follows the trainer's step.
  - Conservative-v2 promotion still decides which network generates
    self-play and serves as the arena parent. On promotion, the accepted
    step becomes the trainer's step.
- **Why:**
  - Under D31 a held candidate was discarded, and every cycle retrained from
    the last accepted model.
  - With about 40-90 updates per cycle, a single cycle rarely beats its
    parent. The D45 arena showed the 39-update candidate at exactly 0.500
    over 24 decisive games, so nothing could ever accumulate.
  - AlphaGo Zero separates the continuously trained optimizer from the gated
    self-play generator in the same way.
- **Test:** `continuous_trainer_carries_learning_across_held_cycles`. With
  every cycle forced to hold, cycle 1 starts at cycle 0's step, the parent
  and the accepted step stay at the reference, and the trainer state
  persists.

## D49 - Pilot health stops

- **Status:** ACCEPTED (2026-09-25)
- **Decision:** the pilot checks `[health_stops]` at every cycle boundary,
  after the cycle report has been written. It stops, with status
  `stopped_health: <reason>`, when any of these hold:
  - the self-play draw share reaches `draw_share_two_cycles` in two
    consecutive cycles;
  - (threefold + fifty-move) / games reaches `threefold_fifty` in any cycle;
  - truncated / games reaches `truncation` in any cycle.
  - Every check is off by default.
  - These are execution bounds, excluded from the scientific identity
    (tested).
- **Why:**
  - Smoke v2's third cycle raised the self-play draw share from 0.25 to 0.73
    once a trained value head guided search.
  - A longer run must not silently spend hours inside a draw attractor. It
    has to stop with the evidence recorded.
- **Test:** `health_stops_trigger_on_the_preregistered_conditions`.

## D50 - MCTS-solver (proven-result propagation)

- **Status:** IMPLEMENTED, NOT ADOPTED (2026-09-29). Stage 1: no conversion
  gain (every probe pair within one game; the pre-registered rule fails on both
  heavy pairs). It stays available, off by default.
- **Decision:** `search_solver = true` enables an MCTS-solver in self-play
  and arena search (`mcts_solver_v1`):
  - Proofs come only from the rules profile's terminals, never from the
    network. The tree is path-dependent, so repetition draws are exact.
  - A node is a proven win when some move reaches a proven loss for the
    opponent. It is proven once every move is proven: a draw if any move
    draws, otherwise a loss.
  - A proven node is not expanded further. Traversals reaching it back up
    its exact value, as for a terminal.
  - Selection always takes the shortest proven win and never a proven loss
    while an alternative exists.
  - Self-play and arena move choice play a proven root win (the shortest)
    regardless of temperature, and never sample a proven losing move while an
    alternative exists.
  - The default `false` is the original search, unchanged (tested). The
    identity records the solver only when it is enabled.
- **Why:** checkmate is the only signal in the system. At 64 sims a found
  mate is still averaged with network values and diluted by root noise and
  sampling, so near-horizon mates are missed. The solver makes every found
  forced result exact and decisive.
- **Limit (INFERRED):** losing the queen in K+Q vs K reaches K vs K, a
  terminal *draw*. The solver values that exactly at 0, which is what the
  network already believes of the position with the queen. The solver
  therefore cannot by itself stop material being thrown away; only a value
  head that knows material wins can. That is D51's job.
- **Tests:**
  - puct: `solver_proves_a_losing_move_and_stops_choosing_it`,
    `solver_proves_a_win_and_concentrates_visits_on_it`,
    `solver_proves_draws_and_losses_when_every_move_is_proven`
  - game_tree: `solver_proves_mate_in_one` (K = 1 and 4),
    `solver_proves_kqk_mate_in_two`
  - play: `solver_always_plays_a_found_mate`,
    `solver_avoids_proven_losses_and_is_inert_when_off`
  - config: `search_solver_is_a_new_identity_only_when_enabled`

## D51 - Endgame curriculum (generated won-material starts)

- **Status:** IMPLEMENTED; stage 2 **NO-GO** (2026-09-29). All of P1-P3
  fail, and the trained network converts heavy endgames worse (18 of 64,
  against 36 untrained). The curriculum dose was 6-10% of positions and
  mostly drawn. It stays available, off by default; see PHASE4_RESULTS for
  the analysis.
- **Decision:** `[endgame_curriculum] fraction, families` starts that share
  of self-play games from a generated endgame (`endgame_curriculum_v1`):
  - **Families:** `K<pieces>vK<pieces>`, stronger side first (e.g. `KQvK`,
    `KRRvK`, `KRPvKP`). The stronger side needs a queen, rook or pawn, and
    more material.
  - **Selection:** exactly `floor(n * fraction)` of the first n games by
    global game id, spread evenly.
  - **Generation:** positions are uniform random placements, deterministic
    from the run seed and game id. Both colours and both sides to move occur.
  - **Rejected placements:**
    - adjacent kings
    - pawns on the first or last rank
    - illegal positions (the side not to move in check)
    - terminal positions
    - the side to move in check
    - any capture available
  - **Unchanged:** moves, search targets and outcome labels all come from
    the network's own play. There is no tablebase, engine label, material
    reward, contempt or adjudication, and a drawn won-material game is
    labelled a draw.
  - **Scope:** arenas are unaffected, and `start_fen` cannot be combined
    with the curriculum.
  - **Metrics:** self-play reports `standard_start` and `curriculum` subsets
    (with stronger- and weaker-side wins). D49 health stops read the
    standard-start subset, and `draw-report` reports other starts
    separately.
- **Why:** P4.6 self-play almost never mates (1-6 of 64 K+Q vs K games), so
  won endgames are labelled draws. Games that start in simplified won
  positions put mates within reach of 64-sim search, especially with D50,
  and give the value head outcome-labelled evidence that material converts.
- **Risk (pre-registered):** if the network cannot convert even the
  simplified starts, the curriculum only adds draws. Stage 1 measures
  conversion per family before any training run.
- **Tests:**
  - curriculum: `families_parse_and_invalid_ones_are_refused`,
    `curriculum_games_are_an_exact_even_share`,
    `generated_positions_are_legal_quiet_decisive_and_reproducible`
  - coordinator:
    `curriculum_games_start_from_generated_endgames_and_are_split`
  - config:
    `endgame_curriculum_is_a_validated_new_identity_only_when_enabled`

## D52 - Pinned build-time CUDA version and a behavioral device check

- **Status:** ACCEPTED (2026-09-29; root-cause fix for an aborted stage 1
  run)
- **Found:**
  - A release build from a shell without the CUDA toolkit on PATH made
    cudarc (0.19.9, `cuda-version-from-build-system` via CubeCL) fall back
    to "latest" (CUDA 13.3).
  - The binary then searched for CUDA 13 NVRTC names. It could not load the
    pinned user-space 12.9.1 runtime (`nvrtc64_120_0.dll`) and panicked on
    the device thread in a loop.
  - The existing guard only checked that *some* `nvrtc*` file is on PATH,
    so it passed.
  - The run was stopped at once and produced no results. No earlier log
    shows this failure: every earlier GPU run was built with the CUDA
    environment set.
- **Decision:**
  1. `.cargo/config.toml` sets `CUDARC_CUDA_VERSION = "12090"`, so every
     build targets the pinned CUDA 12.9.1 runtime (D3) regardless of the
     shell.
  2. `model_io::verify_device` runs elementwise, reduction and matmul
     kernels with known results on the device, on a helper thread with a
     timeout. Every model build and load goes through it. A panic, a hang
     or a wrong result is a visible error, never a silent run.
- **Why:** AGENTS.md forbids a device claim unless the graph actually ran,
  and forbids silent failure. A file-name check cannot prove that kernels
  run; a computation with a known answer can.
- **Tests:** `device_check_passes_on_a_working_backend` (CPU). GPU: a build
  deliberately pinned to CUDA 13.3 must refuse, and the 12.9 build must
  pass (recorded in PHASE4_RESULTS).

## D53 - Throughput pass: owner pool, 48/96 schedule, flattened linears

- **Status:** ACCEPTED (2026-09-29). Execution only: no change to the
  scientific identity, and results are the same.
- **Decision:**
  - **T2:** self-play uses `inference_owners = 2`, two owner threads on one
    queue, each with its own weights copy and device stream.
  - **T3:** the self-play schedule is 48 concurrent games with batch cap 96.
  - **T4:** every rank-3 `Linear` runs as one 2-D GEMM over `b * 64` rows
    (`linear_rows`). This is bit-exact with the old code.
- **Evidence:**
  - T2 1.10x, T3 1.10x, T4 1.094x self-play and 1.20x training.
  - Cumulative self-play: 22.8 -> 31.5 trainable pos/s. Games identical.
  - T4 missed its pre-registered self-play bar by 0.6% and is kept by owner
    decision.
- **Not adopted:**
  - Burn fusion (T1, 20-40% slower).
  - Autotune (T1, below the bar, and it cannot guarantee FP32).
  - TF32 (T5): the tensor-core kernels never win autotune on this model, so
    TF32 is not achieved. The `tf32` mode and its refusal contract remain,
    tested.

## D54 - Arena players search their own trees; no hidden non-finite output

- **Status:** ACCEPTED (2026-09-29). Ported from the HP integration branch
  (its D54 and core review; `docs/HP_BRANCH_COMPARISON.md`) and verified on
  mainline code.
- **Found (mainline has the same code):**
  - `run_arena` passed a `SideRouter` into `play_game_from` as the search
    evaluator, so every tree node went to the network of that node's side to
    move.
  - Each player's search therefore evaluated half its nodes with the
    opponent's network.
  - This pulls arena scores toward 0.5. It affects every mainline searched
    arena since Phase 3, D45 included. Self-vs-self results are unaffected.
- **Decision:**
  1. `arena_tree_policy = "root_player_v1"`: each player searches its own
     tree with its own network (`play_game_per_side`), the AlphaZero
     evaluation contract.
     - The default `per_node_side_v1` reproduces every earlier identity.
     - Any other value enters the evaluation identity.
     - New runs use `root_player_v1`.
     - Earlier arena scores and promotions are **mixed-tree measurements**.
       They are labelled as such, not invalidated.
  2. `evaluate_batch` refuses non-finite or zero policy mass and non-finite
     WDL. Before, it fell back to a uniform policy and a value of 0: a
     silent fallback.
  3. The inference owner counts `completed`/`errors` before replying (a
     metrics race).
- **Tests:**
  - `per_side_play_gives_each_player_its_own_tree`: the old routing sends a
    network only its own colour's nodes; per-side play gives each network
    both parities of its own tree. With identical networks the games are
    identical.
  - `arena_tree_policy_enters_the_identity_only_when_changed`.

## D55 - candidate_v25 architecture identity and CandidateFactsV1

- **Status:** IMPLEMENTED (2026-09-30), branch `experiment/workstation-v25` only.
- **Decision:**
  1. `ModelConfig.architecture` (`probe_v1` default) plus optional `candidate` /
     `legacy_facts` geometries. The defaults are never serialized, so every historical
     scientific hash and `check_model` value is unchanged (the frozen P4.5 hash test still
     passes).
  2. The runtime dispatches over a `NeuralModel` trait and monomorphizes per architecture.
     A Burn `Module` enum would change the recorded checkpoint layout and break every
     historical Probe checkpoint, so it was rejected.
  3. Checkpoints record architecture, head version, CandidateFacts version and the
     token/block/fact-delta contracts. Loading across architectures is refused by id, in both
     directions, for every ordered pair of {probe_v1, candidate_v25, legacy_facts_v25}.
  4. `CandidateFactsV1` (8 exact one-ply facts per legal move, legal-action order) lives in
     `recur64-core` and is computed from the authoritative `GameState`, only for evaluators that
     ask (`needs_candidate_facts`); legacy evaluators pay nothing. `SyncEvaluator` now also refuses
     non-finite/zero policy mass (it previously fell back to a uniform policy).
- **Why:** the experiment needed a second, distinguishable architecture without touching the
  control line.
- **Tests:** `probe_identity_is_unchanged_by_the_architecture_field`,
  `arena_exploration_is_a_new_identity_and_default_is_unchanged`, `candidate_facts_diff`
  (41,353 positions vs an independent reference), `every_ordered_pair_of_architectures_is_refused_explicitly`,
  `lf_contract_one_checkpoints_are_refused_under_contract_two`.

## D56 - ProofTargetsV1: exact mate proofs, independent audit, pool-limited scale

- **Status:** IMPLEMENTED (2026-09-30), branch `experiment/workstation-v25`.
- **Decision:** exact near-mate policy targets (M1/M2/M3 over KQvK, KRvK, KQQvK, KQRvK, KRRvK,
  white to move, `fresh_no_history_v1`) from an exhaustive memoized adversarial search. No
  network, PUCT, engine or human data. Every position of every dataset is re-derived by an
  independent implementation (`GameState::apply`, full termination classification, own memo).
  Splits are hard-disjoint by exact FEN and symmetry-canonical class. Datasets carry a content
  digest.
- **Found:** exact eligible pools are small for KQvK/KRvK (e.g. KRvK M1 189) and KQQvK M3 (4,409),
  so the requested 1000/100/100 was infeasible in those cells. Rule (pre-registered before any
  dataset existed): pools >= 1200 use the targets, smaller pools split 80/10/10; holdouts take
  priority over a training extension; no filter is ever relaxed.
- **Found:** a repeat run caught a determinism bug (the stored FEN was whichever symmetric
  representative a worker saw first); fixed by storing the canonical representative.
- **Tests:** solver vs brute-force enumeration (600 positions), audit rejects a corrupted label,
  thread-count independence, overlap refusal.

## D57 - Evaluation hygiene: cell_balanced_v1, macro metrics, sealed evaluation sets

- **Status:** IMPLEMENTED (2026-09-30), branch `experiment/workstation-v25`.
- **Decision:**
  1. `cell_balanced_v1`: a rotor over the 15 (family, depth) cells with independent seeded
     per-cell shuffles, equal long-run exposure (within one example); oversampling is sampling,
     not extra data.
  2. Macro-cell/family/depth metrics and a full-TRAIN evaluation are reported beside the pooled
     metrics (an earlier "train slice" was a single family and was removed).
  3. Evaluation files carry `model_seed`; `proof compare --per-seed` pairs by seed identity and
     refuses mismatches.
  4. Every CONFIRM/holdout evaluation prints an exposure guard and appends to an exposure log.
  5. A split assignment whose CONFIRM had been touched by a toy smoke was retired and replaced.
- **Tests:** sampler determinism/exposure/wrap tests, macro-metric test, seed-pairing tests.

## D58 - legacy_facts_v25 (LF) and fact-delta contract 2

- **Status:** IMPLEMENTED (2026-09-30), branch `experiment/workstation-v25`.
- **Decision:** LF wraps the unmodified `ProbeModel` and adds CandidateFacts as a candidate-local
  policy delta (`Linear(64->1, no bias)(GELU(Linear(8->64)(facts)))`) added to the legacy logit.
- **Found:** contract 1 had a final bias that adds one constant to every candidate of a row and is
  cancelled by the softmax: an inert parameter with zero gradient. Removed; contract 2; size
  26,810,584 (= L + 640). Gradient coverage is now checked per parameter.
- **Found:** a same-seed wrapped-Probe-vs-independent-Probe test is only deterministic in its own
  process, because Burn's backend RNG is global state shared by parallel test threads.
- **Tests:** `every_facts_parameter_gets_a_finite_nonzero_gradient_on_update_one`,
  `wrapped_probe_is_the_historical_legacy_model_under_the_same_seed`.

## D59 - V2.5 outcome: P2 NO-GO, P2.5 answers, lineage stops

- **Status:** RECORDED (2026-09-30). Measured results, not a code change.
- **Found (P2):** CandidateFacts solve M1 (CF 1.000) and help M2/M3; candidate tokens without
  facts do not beat the matched-capacity legacy head; CF M2 0.689 missed the 0.75 gate (Q3 NO-GO).
- **Found (P2.5, heavy holdouts):** facts interact strongly with the candidate-token architecture
  (interaction +0.085 on M2+M3); the legacy-plus-facts variant LF learned facts far too slowly
  (M1 0.74, underfit); 5x unique heavy data closed the train/held-out gap without raising
  held-out accuracy (M2+M3 +0.005, CI includes 0).
- **Decision:** P3 (conversion) is not authorized by these results; the optimization-horizon test
  was removed from scope by the owner; the lineage stops for review and V3 design.
- **Evidence:** `docs/WORKSTATION_V25_SUMMARY.md`, `docs/WORKSTATION_V25_EXPERIMENTS.md`,
  `docs/WORKSTATION_V25_P25_RESULTS.md`, `docs/evidence/v25/`.

## Rejected / deferred

- **tch-rs**, **Candle**: deferred fallbacks (see `ARCHITECTURE.md`).
- **0.22.0-pre.x Burn**: deferred until a stable release or a demonstrated need.
- **Docker, Python trainer, custom CUDA kernels**: rejected.

---

# V3 decisions (`experiment/workstation-v3-active-search`)

V3 uses its own `V3-D<n>` numbering because HP D50-D63 collide with mainline D50-D54 and V2.5
D55-D59. HP decisions are cited as `HP D<n>`.

## V3-D1 - V3 lineage and scope

- **Status:** RECORDED (2026-09-30).
- **Decision:** branch `experiment/workstation-v3-active-search` from exactly
  `feb86236f24eeaca2a1dc16f7c9e45bca4dc51de`; HP branches are donors/reference only and are not
  merged; V2.5 is closed and gets no P3 or one-pass rescue. See `docs/V3_LINEAGE.md`.

## V3-D2 - Architecture identity `active_search_v3` and query accounting

- **Status:** RECORDED (2026-09-30). Implementation pending (P1-P2).
- **Decision:** one exact transition per query unit; CandidateFacts are root-only and never
  computed on queried descendants; the state-query crate depends only on `recur64-core`; tool
  outputs carry no solver-derived fields; STOP is masked in the primary experiment. See
  `docs/V3_ARCHITECTURE.md`.

## V3-D3 - Pre-registered gates, with a flagged feasibility dependency

- **Status:** RECORDED (2026-09-30).
- **Decision:** gates I-VI and outcome classes are frozen in `docs/V3_RESEARCH_PLAN.md`. The
  Gate II/III magnitudes depend on the P4-measured fraction of ideal proof certificates that fit
  in 8 queries; any restatement must happen before CONFIRM and be reported to the owner.
- **Decision:** at the end of P3 the report and any suggested contract changes are committed and
  pushed to the V3 branch so the record exists off the workstation.
