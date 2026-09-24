# Recur64 — Phase 4 measured results (main workstation)

Durable record of the Phase 4 GPU work (P4.2–P4.5). Every item is labelled
**MEASURED** (produced by a run whose artifacts are listed), **INFERRED**
(derived from measured numbers, not itself measured) or **NOT RUN**. Phase 3
evidence (`docs/F10_BASELINE.md`, `docs/BENCHMARKS.md`) is historical and not
overwritten here.

Compact copies of the machine-readable artifacts live in
`docs/evidence/phase4/`; the full run directories are under `runs/phase4-*`
(gitignored, local to the workstation).

## Hardware and software

| item | value |
|---|---|
| machine | Dell Pro Max Slim FCS1250, Windows 11 x64 |
| CPU / RAM | Intel Core Ultra 9 285K (24c/24t) / 64 GB |
| GPU | NVIDIA RTX 2000 Ada, 16 GB, CC 8.9, driver 596.71 (WDDM) |
| CUDA | user-space 12.9.1 (`%LOCALAPPDATA%\Recur64\cuda\12.9.1`), process-local `CUDA_PATH`/`PATH` |
| toolchain | Rust 1.97.1, Burn 0.21.0 (Cuda backend), FP32 |
| lineage | `main`; harness commit `3cf7408` (Phase 4 sweep methodology, telemetry, probes) |

## P4.2-pre — CUDA runtime proof (MEASURED)

Run on `main@f7770fc` before any code change.

- `nvidia-smi`: RTX 2000 Ada visible; 310 MiB used, no compute processes.
- `recur64 doctor`: CUDA_PATH set, nvcc 12.9, cuda feature compiled.
- `recur64 cuda-smoke --config configs/micro.toml`: **PASS** (forward finite,
  backward + AdamW moves parameters, checkpoint round-trip delta 0).
- NVRTC guard (`RunConfig::ensure_supported`, D36): the run path accepted this
  workstation's actual library `nvrtc64_120_0.dll` (also present:
  `nvrtc64_120_0.alt.dll`, `nvrtc-builtins64_129.dll`) and executed a real F10
  CUDA workload: `bench-runtime --config configs/phase4/f10-reference.toml
  --active 4 --max-batch 4 --simulations 4 --games-per-cell 4` (fresh seeded
  weights; a runtime proof, not a measurement) completed 4/4 games, 0 inference
  errors, peak 485 MiB. Mean forward latency was 17.9 ms at batch ≈ 2.3, i.e.
  small batches are launch/latency bound on this GPU.

## Harness changes before measurement (commit `3cf7408`)

- **Sweep concurrency methodology.** Collection concurrency is
  `min(concurrent_games, cpu_workers, games)`. With the default 32
  games/cell, the old `workstation` grid's 48/64/96 cells could only ever run
  32 games at once while being labelled 48/64/96. Now: `--grid workstation` is
  16/24/32 (realizable with 32 games); `--grid workstation-high` is 48/64 and
  needs ≥ 64 games/cell; 96 only as an explicit single cell. `bench-runtime`
  refuses any cell whose games cannot realize its requested concurrency, and
  every row reports `requested_concurrency`, `effective_concurrency` and
  `peak_in_flight`. Regression test:
  `sweep::tests::workstation_cells_realize_requested_concurrency_or_are_refused`.
  `small`/`full` grids unchanged.
- Shared `gpu_telemetry` (nvidia-smi, 500 ms); the pilot now records per-phase
  (collect/train/eval) VRAM, utilization and temperature.
- Pilot cycle reports now carry `completed_updates`, `max_updates`,
  `max_updates_cap_bound` (from `RunConfig::update_plan`), eval owner counts,
  per-owner eval inference metrics, and the wall-budget overrun.
- Pilot evaluation extracted verbatim into `evaluate_candidate` (identical
  matches, order, seeds) so the lifecycle probe drives the real path.
- New probes: `bench-train` and `bench-lifecycle`.
- Gate: `cargo fmt --check` clean, `cargo clippy --workspace --all-targets`
  clean, `cargo test --workspace --release` 184 passed / 0 failed / 1 ignored
  (pre-existing deep perft). CPU plumbing run of `bench-runtime
  --replay-output`, `bench-train`, `bench-lifecycle` and `replay-audit`
  succeeded.

## P4.2 — frozen F10 reference (MEASURED)

`recur64 freeze-reference --config configs/phase4/f10-reference.toml --output
runs/phase4-f10-reference` on CUDA FP32 (config unchanged; the weights depend
only on seed + geometry, and the recorded hashes describe the freezing config).

| field | value |
|---|---|
| model_id | `7d1493b47f9e2159f4b8d419ee7d237129d8405b0b551f66c4e61eaf18574263` |
| seed | 1 |
| git | `3cf7408f86d54091cc2aaeb54ff01d1fed561330`, branch `main`, clean (no `-dirty`) |
| geometry | F10: width 384, heads 12, FFN 768, blocks 0/8/0, R=1, 9,805,288 params |
| device / precision | cuda / fp32 |
| scientific_config_hash | `ca4f5ce82f66002a157ac217ea6bd8496e4ff238a75577f8d12860d474dd8d01` |
| resolved_config_hash | `de431b0ed15a326da679d0abe9e4de67e9f77756819fd82c58a16f1daf16d5ab` |
| opening suite / digest | `configs/openings-v1.toml` / `66d6dcf5cd805f3283c2d7a6d38305bdbdb3d3ae4b87585b46b596eaa78e327c` |
| optimizer contract | `adamw-v1:burn=0.21.0,beta1=0.9,beta2=0.999,eps=1e-5,weight_decay=1e-4,cautious_wd=false,amsgrad=false,clip=per_parameter_l2_norm@1.0,grad_reduction=example_weighted_mean_over_effective_batch` |
| update_counter / lr_schedule_step | 0 / 0 (required) |

This single checkpoint is used by every scheduling cell, search-budget cell,
T0 evaluation, lifecycle probe and the smoke. No cell regenerates weights.

## P4.2A — T0 reference diagnostics (MEASURED)

**Bug found first (MEASURED, fixed in `22e4ec4`).** The first T0 attempt
(`eval-policy`, 100 games) drove device memory to **16,005 / 16,380 MiB** and
host RSS from 2.4 to 3.9 GB within about a minute at ~12% GPU utilization. The
command evaluated on the *autodiff* backend, so every inference-only forward
recorded graph state that no backward pass consumed. The run was killed (VRAM
returned to 310 MiB) and discarded. `eval-policy` now evaluates on
`B::InnerBackend`, as the arena and pilot already did. With the fix, the same
T0 held a flat **485 MiB** for the whole 2m09s run (59 of 65 samples at 485 MiB;
`docs/evidence/phase4/t0/gpu.csv`).

**T0 result** (`recur64 eval-policy --config configs/phase4/f10-reference.toml
--checkpoint runs/phase4-f10-reference --games 100`, binary `22e4ec4`, frozen
reference `7d1493b4…`, opening digest `66d6dcf5…`). This is untrained
initialization and makes **no strength or Elo claim**.

| metric | value |
|---|---|
| games | 100 (raw policy sampled at temperature 1.0 vs uniform random, alternating colours, openings-v1) |
| policy W / D / L | 15 / 77 / 8 (score 0.535) |
| decisive / informative | 23 / true |
| truncated | 0 |
| terminations | insufficient_material 52, checkmate 23, fifty_move 14, stalemate 8, threefold 3 |
| positions (policy diagnostics) | 12 (the openings-v1 suite) |
| mean policy entropy | 1.790 nats |
| mean uniform entropy (reference) | 3.291 nats |
| mean top-1 probability | 0.436 |
| top moves | a2a4 ×4, h2h3 ×3, g2g3, e4f5, g4f5, e5d6, b5a7 ×1 |

INFERRED: the seeded initialization is already far from uniform (1.79 vs
3.29 nats; top-1 0.44), i.e. the untrained logits carry arbitrary but
confident preferences. A score of 0.535 over 23 decisive games is consistent
with chance; it is recorded only as the longitudinal T0 anchor.

## Pre-registered rules (written before any P4.3/P4.4/P4.4L measurement)

### P4.3 scheduling selection

- Science fixed: frozen reference, F10 FP32, seed 1, standard start,
  c_puct 1.0, temperature 1.0, ply cap 512, **16 simulations/move**, same
  seed sequence (`first_game_id` 0) in every cell. Only concurrency, batch
  cap and timeout change.
- Coarse: 16/24/32 at non-binding caps, 500 µs, 32 games/cell.
- 48/64 run only if trainable pos/s rises from 24 → 32; 96 only if 64 beats
  48 by ≥ 10% trainable pos/s and queue p95 is not pathological.
- Cap ring at the leader: one binding cap (≈ half the concurrency) vs the
  non-binding cap. Timeout ring: 250 / 1000 / 2000 µs vs 500 µs.
- Confirmation: top two schedules rerun with ≥ 2 × effective concurrency
  games.
- **Primary:** trainable positions / wall second. Secondary: eval/s, stable
  batching, queue p95, forward latency. Constraints: 0 inference errors,
  stable VRAM, stable temperature, no pathological CPU oversubscription.
  Differences < 5% are treated as ties, broken toward lower concurrency (less
  CPU oversubscription, more headroom). Never select by GPU %, largest batch
  or largest concurrency.
- Training layout: `bench-train` at effective batch 256 (32×8, 64×4, 128×2)
  on real replay from the frozen reference; pick highest examples/s with
  finite loss/gradients and peak VRAM ≤ 12 GB; < 5% differences are ties
  broken toward the historical prior 64×4.

### P4.4 search-budget selection

Budgets 8 / 16 / 32 / 64 at the frozen schedule, same seeds, same frozen
reference; only `simulations_per_move` changes. 256 is never run.

A budget is **degenerate** if ANY of:

- mean trainable-target top-1 visit share > 0.90
- (threefold + fifty-move) / games > 0.80
- mean trainable-target entropy < 0.10 nats
- truncated / games > 0.50

These are the HP priors adopted unchanged (not mainline results).

**Primary:** the highest trainable positions/s among non-degenerate budgets
(the leader).

**Override:** a deeper budget replaces the leader only if its combined
(threefold + fifty-move + truncated) / games rate is ≥ 0.15 lower (absolute)
than the leader's AND it retains ≥ 50% of the leader's trainable positions/s.
If several deeper budgets qualify, take the lowest combined rate; ties broken
toward higher throughput.

**128 rule:** run 128 only if 64 is non-degenerate AND the 32 → 64 step
improved the combined rate by ≥ 0.05 absolute or raised trainable entropy by
≥ 10% relative AND the projected 128 cell runtime is ≤ 30 min. Otherwise 128
is NOT RUN.

If every tested budget is degenerate: STOP; no F10 learning; no reward
shaping, contempt, external labels or Gumbel.

### P4.4L lifecycle GO

Modes: `one` (A), `two` (B), `pilot` (C, parent == reference) and
`pilot-promoted` (C′, forced longitudinal arena), 8 repetitions each.

**GO** if, for every mode, post-shutdown VRAM plateaus (after rep 2, growth
≤ 64 MiB per rep and no monotonic rise across the remaining reps) AND mean
forward latency per rep stays within ±15% of the median of reps 1–2, with
zero inference errors. Otherwise diagnose, and fix only with measured cause.
