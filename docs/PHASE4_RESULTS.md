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

#### Owner amendment A1 (2026-09-24, before any P4.4 measurement)

The owner judged the rule above structurally biased toward very low budgets.
Trainable positions/s scales roughly with 1/sims, and nothing in the rule
measured whether search improves on the network prior. For reference,
AlphaZero/MuZero self-play used 800 simulations/move and Leela Chess Zero
roughly ~800 visits/move. A 16-sim PUCT target sits close to the prior.
Disclosure: the P4.3 scheduling cells (16 sims) had already shown one budget's
data health (trainable top-1 0.678, entropy 0.80 nats, threefold+fifty 0.59,
truncation 0) before this amendment. No 8/32/64/128/256 data existed.

Amended P4.4 (supersedes the corresponding parts above):

- **Curve:** 8 / 16 / 32 / 64 / 128 / 256. 128 is now unconditional (the 128
  rule above is withdrawn). 256 runs with ~16 games as a **measurement-only**
  cell: it is recorded, but it is not eligible for selection.
- **Minimum eligible budget: 64.** 8/16/32 are measured for the curve only.
- **Search-gain gate (new).** A budget is eligible only if, over trainable
  plies, mean KL(visit target ‖ network prior) ≥ **0.10 nats** AND the target
  argmax differs from the prior argmax on ≥ **15%** of positions.
  - The prior is recomputed offline with `recur64 search-gain`: the replay
    positions are re-evaluated with the frozen reference. Self-play applies no
    root noise, so the root prior is exactly that network's policy.
  - Replay V1 is unchanged.
  - **Caveat, found before any P4.4 data (CPU plumbing replay at 4 sims):**
    KL was 0.535 nats while the argmax changed on only 10% of positions.
    Visit targets are discrete (N visits spread over few moves), so KL is
    inflated by coarse-graining at low budgets. That is why both conditions
    are required: argmax change is the guard against quantization-only
    "gain". The thresholds are unchanged.
- **Selection:** among eligible budgets (≥ 64, not degenerate, and passing the
  search-gain gate), take the highest trainable positions/s. The 0.15 / 50%
  data-health override above still applies among eligible budgets.
- **Smoke:** uses the budget this rule selects. Its wall bound may extend to
  ~45 min so each cycle has enough trainable positions to train on.

### P4.4L lifecycle GO (pre-registered)

Modes: `one` (A), `two` (B), `pilot` (C, parent == reference) and
`pilot-promoted` (C′, forced longitudinal arena), 8 repetitions each.

**GO** if, for every mode, post-shutdown VRAM plateaus (after rep 2, growth
≤ 64 MiB per rep and no monotonic rise across the remaining reps) AND mean
forward latency per rep stays within ±15% of the median of reps 1–2, with
zero inference errors. Otherwise diagnose, and fix only with measured cause.

## P4.3 — main-workstation scheduling (MEASURED)

All cells: frozen reference `7d1493b4…`, `configs/phase4/f10-reference.toml`
science (seed 1, standard start, c_puct 1.0, temperature 1.0, ply cap 512),
**16 simulations/move**, `first_game_id` 0 in every cell, binary `22e4ec4`.
Every cell had 0 inference errors.

### P4.3A coarse pass (32 games/cell)

`bench-runtime --grid workstation --games-per-cell 32` → `runs/phase4-sched-coarse`

| conc (req = eff = peak) | cap | trainable pos/s | ev/s | batch mean/p50/p95/max | queue p50/p95 µs | fwd ms | wall s | VRAM MiB | util % | temp °C |
|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| 16 | 16 | 38.4 | 598 | 10.6/14/16/16 | 3978/15106 | 13.2 | 142.7 | 645 | 44 | 57 |
| 24 | 24 | 45.1 | 704 | 12.7/14/24/24 | 4348/12419 | 13.2 | 121.3 | 709 | 44 | 59 |
| 32 | 32 | 45.8 | 714 | 13.8/15/29/32 | 4455/15215 | 13.6 | 119.6 | 805 | 42 | 61 |

Data aggregates were **identical** in all three cells: 5,478 positions (all
trainable), W/D/B 3/28/1, terminations threefold 19 / insufficient 8 /
checkmate 4 / stalemate 1, mean 171.2 plies, target entropy 0.798, top-1
0.678, scientific hash `3565a535…`.

**Finding.** With concurrency ≥ games / 2, the cell is one or two waves, so
its wall time is roughly the longest game (all games start together).
24 → 32 moved +1.5%, which is inside the pre-registered 5% tie band. The
coarse pass therefore cannot rank ≥ 24. The literal 48/64 condition ("rises")
was met, so the ranking was re-measured with 128 games per cell (≥ 2 waves
even at 64). 16 was pruned: it was 15% behind with two waves.

### P4.3B multi-wave concurrency (128 games/cell)

`runs/phase4-sched-multiwave-{24,32,high}`

| conc (req = eff = peak) | cap | trainable pos/s | ev/s | batch mean/p50/p95/max | queue p50/p95 µs | fwd ms | wall s | VRAM MiB | util % | temp °C |
|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| 24 | 24 | 85.9 | 1347 | 21.0/24/24/24 | 336/8894 | 13.9 | 263.6 | 666 | 56 | 77 |
| 32 | 32 | 88.7 | 1391 | 25.0/32/32/32 | 506/14861 | 15.4 | 255.3 | 922 | 58 | 80 |
| 48 | 48 | 81.7 | 1281 | 28.6/35/44/48 | 19497/23466 | 18.4 | 277.1 | 1178 | 55 | 80 |
| 64 | 64 | 93.2 | 1461 | 28.4/34/42/49 | 18936/22755 | 16.6 | 243.1 | 1498 | 58 | 80 |

Data aggregates were **identical** in all four cells: 22,649 positions (all
trainable), W/D/B 15/100/13, 0 truncated, terminations threefold 69 /
checkmate 28 / insufficient 26 / stalemate 3 / fifty 2, mean 176.9 plies,
target entropy 0.787, top-1 0.682, scientific hash `e58a9bc2…`. Changing only
the schedule did not change the data.

**Findings (INFERRED from the table).**

- Multi-wave throughput is about 2× the single-wave numbers: the coarse pass
  was tail-bound.
- Above 32 the queue becomes pathological. Queue p50 rises from 0.5 ms to
  ~19 ms, longer than a whole forward pass, and batch mean plateaus at ~28
  even with 48–64 games. That is the CPU-oversubscription signature: 48–64
  search threads on 24 cores.
- 48 < 24 shows run-to-run variation of roughly ±5–8%. 64's +5.0% over 32 is
  at that noise floor.
- Under the pre-registered constraint (no pathological CPU
  oversubscription), **32 is the leader** and 64 goes to confirmation as the
  runner-up.
- **96: NOT RUN.** Its condition required 64 to beat 48 by ≥ 10% (met) *and*
  a non-pathological queue (not met).
- Sustained load reaches 80 °C. From the ring pass onward, an independent
  `nvidia-smi` log records SM clock and throttle reasons.

### P4.3C/D batch-cap and timeout rings at concurrency 32 (128 games/cell)

`runs/phase4-sched-ring-b{cap}-t{timeout}`. Independent `nvidia-smi` log:
`runs/phase4-sched-rings-gpu.csv`.

| cap | timeout µs | trainable pos/s | ev/s | batch mean/p50/p95 | queue p50/p95 µs | fwd ms | VRAM MiB | temp °C |
|---:|---:|---:|---:|---|---|---:|---:|---:|
| 16 (binding) | 500 | 78.0 | 1223 | 14.6/16/16 | 11017/12373 | 11.1 | 666 | 74 |
| 32 | 250 | 97.4 | 1527 | 24.6/32/32 | 430/15682 | 14.5 | 937 | 80 |
| 32 | 500 (P4.3B) | 88.7 | 1391 | 25.0/32/32 | 506/14861 | 15.4 | 922 | 80 |
| 32 | 1000 | 95.7 | 1500 | 25.1/32/32 | 399/14488 | 14.4 | 937 | 80 |
| 32 | 2000 | 77.9 | 1221 | 25.2/32/32 | 415/14507 | 16.9 | 937 | 80 |

Data aggregates were identical to P4.3B in every cell. Every cell had 0
errors.

**Findings.**

- The binding cap (16) costs about 12%.
- 250 / 500 / 1000 µs are non-monotone (97 / 89 / 96). Batches fill to the
  cap at p50 before the timeout fires, so the timeout should matter little
  here. INFERRED: the spread reflects run-to-run noise of roughly ±8%. That
  is why the confirmation includes 500 µs as a declared noise control.
- 2000 µs is pruned.

**Thermals (MEASURED).** SM clock 2400–2460 MHz under load, maximum 80 °C.
Throttle reasons were only `0x1` (idle) and `0x4` (software power cap, the
card's 70 W limit). No thermal slowdown was observed.

## Future optimization backlog (INFERRED, NOT RUN)

Recorded at the owner's request so Phase 4 remains the measured baseline that
each optimization is compared against. None of this is implemented.

Measured basis:

- Mean forward is ~13–15 ms at batch 25–32, and GPU utilization is 55–60%.
- F10 costs ≈ 1.25 GFLOP per position, so batch 32 ≈ 40 GFLOP. At roughly
  12 TFLOPS FP32 that would take ≈ 3–4 ms.
- INFERRED: the forward runs at roughly 20–25% of peak and is launch- or
  dispatch-bound. Latency is nearly flat in batch size (13.2 → 13.6 ms for
  batch 10.6 → 13.8 in P4.3A).

Engineering (no science change):

1. Larger batches without more OS threads: several leaves in flight per game
   (virtual loss), or async multi-game workers. This matters because
   48/64 threads already show CPU oversubscription on 24 cores.
2. Pipelined inference owner: double-buffer, so the next batch is assembled
   while the GPU runs the current one.
3. Kernel fusion and autotune coverage for RMSNorm, attention with relative
   bias, and the FFN. Verify which Burn fusion paths are active.
4. Profile CPU-side costs: observation encoding, legal-move generation,
   tensor construction, host↔device copies.

Search efficiency (changes search semantics; each needs its own ADR):

- subtree reuse between moves
- a neural-network evaluation cache keyed by the full observation, including
  history (relevant because self-play is repetition-heavy)
- KataGo-style playout-cap randomization
- Gumbel root search (explicitly deferred)
- TF32 / BF16 (precision-gated)

Scale context:

- AlphaZero / MuZero self-play used 800 simulations per move.
- INFERRED: at the ~1,400–1,500 ev/s measured here, 800 simulations would
  give ≈ 1.8 positions/s.

### P4.3E confirmation (192 games = 6 waves at concurrency 32, cap 32)

`runs/phase4-sched-confirm-t{250,500,1000}`; GPU log
`runs/phase4-sched-confirm-gpu.csv`. The top two schedules from P4.3C/D were
32/32/250 and 32/32/1000. 32/32/500 was added as a declared noise control.

| timeout µs | trainable pos/s | ev/s | games/h | batch mean/p50/p95/max | queue p50/p95 µs | fwd ms | wall s | VRAM MiB | util % | temp °C |
|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| 250 | 105.8 | 1660 | 2111 | 26.3/32/32/32 | 435/15730 | 14.4 | 327.4 | 937 | 60 | 79 |
| 500 | **109.9** | **1723** | 2192 | 27.4/32/32/32 | 389/**5296** | 14.5 | 315.3 | 937 | 63 | 80 |
| 1000 | 85.7 | 1343 | 1709 | 27.4/32/32/32 | 400/14385 | 17.3 | 404.5 | 937 | 57 | 80 |

**Data aggregates were identical in all three cells:**

- 34,646 positions, all trainable
- W/D/B 21/153/18, 0 truncated
- terminations: threefold 98, insufficient 46, checkmate 39, stalemate 5, fifty 4
- mean 180.4 plies
- target entropy 0.788, top-1 0.682
- scientific hash `a7ac8cf0…`

Every cell had 0 errors. The throttle reasons were only idle and power cap.

**Findings.**

- The timeout ranking flipped between the 128-game and 192-game runs, so
  timeouts from 250 to 1000 µs are not distinguishable above run-to-run noise.
- That noise is dominated by per-process forward latency: 14.4 vs 17.3 ms for
  the same batch shape.
- INFERRED cause: Burn/CubeCL autotune selects kernels by timing at process
  start, so separate processes can settle on different kernels. This is added
  to the optimization backlog.

**Frozen self-play schedule: concurrent_games 32, max_inference_batch 32,
batch_timeout_us 500, cpu_workers ≥ 32.**

- Primary: it had the highest trainable positions/s in the confirmation set.
- Secondary: it had the best queue p95, and it is the historical prior.
- Constraints met: 0 errors, VRAM stable at 937 MiB, no thermal slowdown, and
  no oversubscription pathology (unlike 48/64).
- Replay written by this cell (`runs/phase4-sched-confirm-t500/replay`) is the
  real data used for P4.3F.

### P4.3F training physical batch (MEASURED)

`recur64 bench-train --config configs/phase4/f10-reference.toml --checkpoint
runs/phase4-f10-reference --replay runs/phase4-sched-confirm-t500/replay
--layouts L --updates 20 --warmup-updates 2`. Each layout ran in its own
process so peak VRAM is not shared through the device pool. Effective batch
was 256 for every layout. The replay was the 34,646-position confirmation
replay.

| layout | examples/s | ms/update | peak VRAM MiB | max temp °C | loss first → last | max pre-clip grad norm | finite |
|---|---:|---:|---:|---:|---|---:|---|
| 32×8 | 432.7 | 591.6 | 1481 | 65 | 1.772 → 2.291 | 102.3 | yes |
| **64×4** | **457.9** | **559.1** | 3049 | 67 | 1.772 → 2.429 | 102.4 | yes |
| 128×2 | 410.6 | 623.5 | 4073 | 69 | 1.772 → 2.431 | 102.3 | yes |

**Frozen: 64 × 4 (effective 256).** It is 5.8% faster than 32×8 (outside the
tie band) and 11.5% faster than 128×2, with ample VRAM headroom. It is also
the historical prior.

**Health flag (for P4.5, not a P4.3 gate).** Loss rises over these first 20
warmup updates (1.77 → 2.3–2.4) with pre-clip global gradient norms of about
100. The per-parameter clip is at 1.0.

## P4.3 GO (MEASURED)

`configs/hardware/workstation-main.toml` is now **MEASURED**:

- self-play: concurrent_games 32, cpu_workers 32, max_inference_batch 32,
  batch_timeout_us 500
- learner: train_batch 64, accumulation_steps 4

Machine-readable evidence is in `docs/evidence/phase4/scheduling/`.

**INFERRED transfer check for P4.4.** Every P4.4 cell runs this schedule and
records eval/s. If eval/s at higher budgets departs materially from about
1,400–1,700, the schedule is re-checked rather than assumed to transfer.

## P4.4 — F10 search-budget requalification

**Sample sizes (decided before any P4.4 run, from the P4.3 finding that
one-wave cells are tail-bound).**

- 8 / 16 / 32 / 64 / 128: 64 games each (2 waves at the frozen concurrency
  32), so throughput is not dominated by the longest game.
- 256 (measurement-only): 32 games, the fewest that realize the frozen
  concurrency 32. The amendment's "~16 games" would need a different
  schedule. Its throughput is therefore single-wave and tail-biased; it is
  used for search gain and data health only.
- Every budget uses the same game seeds (`first_game_id` 0) and the frozen
  reference and schedule (32 / 32 / 500 µs). Only `simulations_per_move`
  changes.
