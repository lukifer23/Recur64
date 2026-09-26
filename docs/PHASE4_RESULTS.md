# Recur64 — Phase 4 measured results (main workstation)

Durable record of the Phase 4 GPU work (P4.2–P4.5). Every item is labelled
**MEASURED** (produced by a run whose artifacts are listed), **INFERRED**
(derived from measured numbers, not itself measured) or **NOT RUN**. Phase 3
evidence (`docs/F10_BASELINE.md`, `docs/BENCHMARKS.md`) is historical and not
overwritten here.

Compact copies of the machine-readable artifacts live in
`docs/evidence/phase4/`; the full run directories are under `runs/phase4-*`
(gitignored, local to the workstation).

## Current status (2026-09-25): P4.2–P4.5, post-smoke fixes and F10 smoke v2 complete; STOP before P4.6

| step | status |
|---|---|
| CUDA runtime proof | **GO** (MEASURED) |
| P4.2 frozen reference | v1 superseded; **v2** `d22c78bd…` frozen (head v2); T0 recorded |
| P4.3 hardware schedule | **GO**: `configs/hardware/workstation-main.toml` MEASURED (32 / 32 / 500 µs, cpu_workers 32; learner 64×4). The 64-sim transfer check shows a 48-way lead, not yet confirmed |
| P4.4 search budget | **GO: 64 sims** (pre-registered rule; v2 curve 8–256) |
| P4.4L lifecycle probe | **GO after fix D44**: an owner-memory defect was found and fixed |
| P4.5 F10 smoke | **CONDITIONAL**: every system, data and training gate passes and the value head learns, but the searched arena is repetition-dominated, so no promotion occurs and learning cannot compound (see P4.5) |
| Post-smoke fixes | D45 arena exploration (decisive 3 → 24 of 32), D47 multi-leaf search (K=2, +83% trainable pos/s), D48 continuous trainer, D38 evaluation deadline, D37 crash-safe archival, D46 at most two resident models |
| F10 smoke v2 | **GO for the learning mechanism**: training compounds (WDL 1.10 → 0.63), two promotions, and search movement over the prior rises 11% → 37% once the promoted value head guides self-play. Strength over T0 not yet shown (0.500 vs reference). Watch item: self-play draw share 0.25 → 0.73 in cycle 3 |

Major findings (details below):

1. `eval-policy` leaked device memory on the autodiff backend: 16 GB VRAM
   within a minute. Fixed (D43).
2. One-wave sweep cells are tail-bound. The methodology was fixed, and
   schedules are compared at ≥ 2 waves.
3. **Root cause of repetition-dominated self-play (D40/D41).** The initial
   network had an arbitrary, confident policy and a non-neutral value. PUCT
   reproduced that prior, so self-play distilled it, and nothing broke the
   loop.
   - Fixed by head v2 (final norm, scaled logits, zero-init WDL), root
     Dirichlet noise, and argmax after ply 30.
   - `argmax_after_ply` had been a dead identity field.
   - At 16–32 sims, decisive games went from ~12 to ~45 of 64, and
     threefold + fifty from ~0.5 to 0.08.
4. The T0 search-gain gate was a design error. It is now a per-cycle
   learning-progress metric (D42).
   - Under root noise, the offline metric includes the Dirichlet noise.
   - The exact noise vs search split is now measured in self-play. At T0,
     search moves the argmax on about 11% of positions.
5. **GPU memory grew by hundreds of MiB per inference-owner lifecycle.**
   - VRAM went from 453 MiB to 10.5 GB over 32 lifecycles; this very likely
     caused the HP branch's late-run exhaustion.
   - Root cause: CubeCL's per-thread stream memory pools were orphaned by
     short-lived owner threads.
   - Fixed by releasing the pool on owner shutdown (D44). Usage now plateaus
     at about 1 GB, and stayed stable in the smoke.
6. **Smoke: the loop is interpretable and the value head learns** (WDL loss
   1.10 → 0.83). But the searched arena, which is deterministic and
   noise-free, is 75% threefold repetition. The promotion gate is starved
   (3–5 decisive games), so no promotion happens and learning cannot
   compound.
   - This evaluation-contract question is the first scientific decision
     before P4.6.

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

### P4.4 curve on reference v1 (MEASURED; SUPERSEDED, not used for selection)

Binary `39d744f`, frozen reference v1 `7d1493b4…` (head v1), schedule
32 / 32 / 500 µs, 64 games per budget, same seeds. Search gain is from
`recur64 search-gain` over trainable plies. Artifacts:
`docs/evidence/phase4/search-v1/`.

| sims | games | positions | trainable | trainable pos/s | ev/s | wall s | W/D/B/T | mean plies | threefold+fifty | target H (tr) | target top-1 (tr) | prior H | KL(target‖prior) | argmax changed |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | 64 | 12236 | 12236 | 110.1 | 869 | 111 | 3/57/4/0 | 191.2 | 0.516 | 0.589 | 0.729 | 1.209 | 0.217 | 9.0% |
| 16 | 64 | 11113 | 11113 | 84.7 | 1325 | 131 | 6/52/6/0 | 173.6 | 0.562 | 0.786 | 0.683 | 1.229 | 0.136 | 6.2% |
| 32 | 64 | 12398 | 12398 | 32.8 | 1029 | 378 | 4/53/7/0 | 193.7 | 0.422 | 0.946 | 0.650 | 1.235 | 0.079 | 4.1% |
| 64 | 64 | 12865 | 12353 | 16.3 | 1064 | 756 | 9/41/13/1 | 201.0 | 0.281 | 1.041 | 0.633 | 1.209 | 0.057 | 4.0% |

Checkmates were 7 / 12 / 11 / 22 and threefold repetitions 33 / 35 / 27 / 15
across 8 / 16 / 32 / 64 sims. 128 and 256 were stopped by the owner
(**NOT RUN** on v1) once the root cause below was found. Every cell had 0
inference errors.

**What the curve showed (MEASURED).** Search gain *fell* as the budget rose.
The visit target converged back toward the network prior: the argmax changed
on 9.0% → 4.0% of positions and KL fell 0.22 → 0.06. Deeper search still
improved data health through terminal discovery: at 64 sims, checkmates
doubled and threefold + fifty fell from ~0.52 to 0.28.

**Schedule-transfer check (MEASURED).** ev/s was 869–1325 across budgets,
against 1343–1723 in P4.3. At 8 sims, per-position CPU work dominates; at
32–64, eval/s was ~1030–1065. INFERRED: part of the drop comes from
terminal-node traversals that need no evaluation, and part from the
per-process latency variance measured in P4.3E. Re-checked on reference v2.

## Root-cause audit (MEASURED / code-reviewed)

The owner asked for a deeper check of the model and configuration rather
than further harness tuning.

**Mechanism (INFERRED from the code and the measurements above):**

1. The fresh network had an arbitrary, confident policy and a non-neutral
   value.
2. PUCT with an uninformative value head allocates visits roughly in
   proportion to the prior (`q + c·P·√N/(1+n)` with near-equal q). More
   simulations therefore reproduce that arbitrary prior more faithfully.
3. The policy target becomes self-distillation of the initialization.
4. Self-play follows those arbitrary preferences into repetition draws, so
   value targets are mostly "draw" and the value head stays uninformative.
5. With no root exploration noise, nothing breaks the loop.

INFERRED: this very likely also underlies Phase 3's "repetition-dominated
search, poor learning health".

**T0 head measurement.** A fresh F10, 3 seeds × 13 positions (openings-v1
plus startpos). Test: `crates/recur64-runtime/tests/t0_prior.rs`.

| head | policy entropy / uniform | mean \|value\| | max \|value\| |
|---|---:|---:|---:|
| v1 (before) | **0.502** | **0.245** | 0.548 |
| v2 (after) | **0.999** | **0.000** | 0.000 |

| # | finding | evidence | action |
|---|---|---|---|
| M1 | No final normalization before the policy/WDL heads. The head-input scale depended on the block layout (F10: after 8 core blocks; R10: after 2 output blocks following R recurrent passes), which is also a potential F10-vs-R10 confound. | code (`model.rs` readout) | **Fixed (head v2):** final RMSNorm before both heads |
| M2 | Unscaled bilinear policy logits: a dot product over `policy_dim` = 128 with no 1/√d, whereas attention in the same file is scaled | code + T0 entropy 0.50× uniform | **Fixed (head v2):** logits × 1/√policy_dim |
| M3 | The value head was not neutral at init | mean \|value\| 0.245 | **Fixed (head v2):** zero-initialized WDL head |
| M4 | No depth-scaled residual init | inferred only | Not changed. T0 passes without it. |
| S1 | **`argmax_after_ply` was a dead field.** It was in the config and the scientific hash, but never passed to self-play, so setting it had no effect. | code (`coordinator.rs` built `SelfPlayConfig` without it) | **Fixed:** implemented (temperature 0 from that ply on, counted from the game's start) |
| S2 | Temperature 1.0 on every ply; AlphaZero sampled only in the opening | config | **Owner decision:** sample the first 30 plies, then argmax |
| S3 | No root exploration noise | code | **Owner decision:** root Dirichlet noise in self-play (α 0.3, ε 0.25, AlphaZero chess); arenas stay noise-free by contract |
| T1 | Gradient clipping is per parameter tensor at 1.0, not global-norm. The pre-clip global norm was ~100 in P4.3F, so every tensor was clipped on every step. | code + P4.3F | **Owner-approved, conditional:** re-measure on v2 first; switch to global-norm clipping if norms stay large |
| — | The observation encoding is normalized: binary planes, halfmove /150, repetition /5 | code | No change |

**Contract changes that make drift impossible to miss:**

- `recur64_model::model::HEAD_VERSION = 2`.
- `CheckpointMeta.head_version`: metadata written before the field existed
  reads as 1.
- A mismatch is refused by `check_contracts`. `model_io::load` (used by
  bench-runtime, eval-policy, search-gain and the pilot's inference owners)
  now checks the checkpoint contracts. Previously it checked none.
- `scientific_identity` is v4 and adds `model_head_version` plus
  `root_dirichlet_alpha` / `root_dirichlet_epsilon`.
- F10 and R10 unique parameters: 9,805,288 → **9,805,672**, both
  architectures, from the 384 final-norm scale parameters. Parity holds.

**One change reverted during the gate (MEASURED).** The promotion-delta head
was first zero-initialized as well. That made the first-step gradient into
`promo1` exactly zero, which the existing test
`promotion_path_receives_gradient` caught. No measurement showed a promotion
problem, so that change was reverted rather than editing the test.

**Owner amendment A2 (post-hoc; the v1 data had been seen).** The search-gain
gate from A1 cannot be passed by any budget on an untrained network. Search
cannot improve on a prior when the value head carries no information, so
this was a measurement-design error.

- Search gain is removed as a T0 selection gate.
- It is kept as a **per-cycle learning-progress metric**: measured on each
  smoke cycle's replay against that cycle's generating snapshot. It should
  rise once the value head learns.
- Budget selection on reference v2: minimum eligible 64, degeneracy gates,
  and highest trainable positions/s, with the 0.15 / 50% data-health
  override.

## P4.2 (re-freeze) — frozen F10 reference v2 (MEASURED)

`recur64 freeze-reference --config configs/phase4/f10-reference.toml --output
runs/phase4-f10-reference-v2` on CUDA FP32, head v2. **This reference replaces
v1 for every later Phase 4 step.**

| field | value |
|---|---|
| model_id | `d22c78bda8fb8fa1214c6868de18b8d07f5c50e6fa00aff43fc74688f7726448` |
| head_version | 2 |
| seed | 1 |
| git | `bbe4a816d450b51a0d0d4f93a99f936c46c82b40`, `main`, clean |
| geometry | F10 384 / 12 / 768, blocks 0/8/0, R=1, 9,805,672 params |
| scientific_config_hash | `58a15f88a8881089675a2635eb91150ccc63ab1216dcc2ef0f4f9f47d885a0fc` (identity v4) |
| resolved_config_hash | `3cd240b7e163a46644810bb2e8172c97204f4e19cc458dba9e94e32e27aed923` |
| opening digest | `66d6dcf5cd805f3283c2d7a6d38305bdbdb3d3ae4b87585b46b596eaa78e327c` |
| optimizer contract | unchanged (`adamw-v1 … clip=per_parameter_l2_norm@1.0 … example_weighted_mean_over_effective_batch`) |
| update_counter / lr_schedule_step | 0 / 0 |

Loading v1 is now refused, as intended: `checkpoint head version 1 is not the
current head version 2 … refused` (MEASURED).

### T0 on reference v2 (MEASURED)

`eval-policy`, 100 games, binary `159eb78`. VRAM stayed flat at 498–542 MiB.
No strength claim is made.

| metric | v1 (superseded) | **v2** |
|---|---|---|
| policy W / D / L (score) | 15 / 77 / 8 (0.535) | **7 / 85 / 7 (0.500)** |
| decisive / informative | 23 / true | 14 / true |
| truncated | 0 | 1 |
| terminations | insuff 52, mate 23, fifty 14, stalemate 8, threefold 3 | insuff 59, fifty 21, mate 14, threefold 4, stalemate 1, truncated 1 |
| mean policy entropy (uniform 3.291) | 1.790 | **3.285** |
| mean top-1 probability | 0.436 | **0.046** |

INFERRED:

- The v2 raw policy is indistinguishable from random play, which is what a
  near-uniform prior should give.
- The per-opening argmax "top moves" match v1. The same seed produces the
  same projection weights, and scaling plus normalization preserve logit
  order; the preferences are about 100× weaker.

### T1 — gradient clipping re-measured on v2 (MEASURED; decision: not applied)

`bench-train --layouts 64x4 --updates 20` on the same real replay as P4.3F.

| head | loss first → last | max pre-clip global grad norm | examples/s |
|---|---|---:|---:|
| v1 | 1.772 → 2.429 (rising) | 102.4 | 457.9 |
| v2 | 3.569 → 1.979 (falling) | 12.0 | 440.5 |

**Decision: keep per-parameter clipping (the optimizer contract is
unchanged).** Head v2 removed the pathology that motivated T1: gradient norms
fell about 8.5× and the loss now decreases. Changing clipping in the same step
as the head would also confound attribution. Per-update gradient norms are
reported by the smoke, and T1 is revisited with that evidence before P4.6.

### P4.4 curve on reference v2 (MEASURED so far; interim)

Binary `159eb78`, reference v2 `d22c78bd…`, schedule 32 / 32 / 500 µs, same
seeds. Self-play contract (D41): temperature 1.0 for the first 30 plies then
argmax, root Dirichlet α 0.3 / ε 0.25. 64 games per budget. Artifacts:
`runs/phase4-search-v2-s{sims}`.

| sims | trainable pos/s | ev/s | W/D/B/T | mean plies | threefold+fifty | truncated | target H (tr) | target top-1 (tr) | terminations |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| 8 | 110.7 | 964 | 16/31/14/3 | 287 | 0.141 | 0.047 | 1.729 | 0.263 | mate 30, insuff 20, fifty 9, stalemate 2, trunc 3 |
| 16 | 59.5 | 948 | 16/22/26/0 | 205 | 0.078 | 0.000 | 2.352 | 0.187 | mate 42, insuff 14, fifty 4, stalemate 3, threefold 1 |
| 32 | 33.3 | 1061 | 23/17/24/0 | 200 | 0.078 | 0.000 | 2.838 | 0.148 | mate 47, insuff 11, fifty 5, stalemate 1 |
| 64 | 13.5 | 898 | 21/18/24/1 | 186 | 0.062 | 0.016 | 2.965 | 0.139 | mate 45, insuff 14, fifty 2, threefold 2, trunc 1 |
| 128 | 6.7 | 850 | 29/12/23/0 | 138 | 0.062 | 0.000 | 3.007 | 0.134 | mate 52, insuff 8, fifty 4 |
| 256 (32 games, measurement-only, single wave) | 2.49 | 633 | 9/12/11/0 | 211 | 0.031 | 0.000 | 2.840 | 0.148 | mate 20, insuff 10, fifty 1, stalemate 1 |

**v1 → v2 at equal budgets (MEASURED).**

| sims | decisive games (of 64) | threefold+fifty | trainable target entropy | trainable top-1 |
|---:|---|---|---|---|
| 16 | 12 → 42 | 0.56 → 0.08 | 0.79 → 2.35 | 0.68 → 0.19 |
| 32 | 11 → 47 | 0.42 → 0.08 | 0.95 → 2.84 | 0.65 → 0.15 |

**Reading (INFERRED).**

- The repetition loop is broken.
- Most games now end in checkmate: a near-uniform prior lets search spread,
  find forced mates, and convert them after ply 30.
- About 70% of games now carry win/loss value targets, versus ~17% before.
- Search-gain columns are not meaningful at T0 with a near-flat prior (D42):
  the argmax changed on ~91% of positions because the prior's argmax is
  arbitrary.

**Open check (pre-registered transfer check).** ev/s on v2 is 948–1061,
against 1325 on v1 at 16 sims. Evaluations per position are unchanged
(~16), so the evaluation rate itself dropped. It is examined after the
curve completes (forward latency, batch composition) rather than assumed.

### P4.4 selection on reference v2 (MEASURED; rule applied exactly as pre-registered)

The eligible budgets are ≥ 64 and not degenerate (amendment A1; A2 removed the
search-gain gate). 256 is measurement-only.

| sims | eligible | degenerate? | trainable pos/s | decisive share | threefold + fifty + truncated | trainable H | top-1 | wall s (64 games) | peak VRAM MiB | max °C |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | no (< 64) | no | 110.7 | 0.469 | 0.188 | 1.729 | 0.263 | 152 | 943 | 70 |
| 16 | no | no | 59.5 | 0.656 | 0.078 | 2.352 | 0.187 | 221 | 943 | 76 |
| 32 | no | no | 33.3 | 0.734 | 0.078 | 2.838 | 0.148 | 384 | 958 | 79 |
| **64** | yes | no | **13.5** | 0.703 | 0.078 | 2.965 | 0.139 | 842 | 943 | 80 |
| 128 | yes | no | 6.7 | 0.812 | 0.062 | 3.007 | 0.134 | 1320 | 996 | 80 |

- **Primary:** among eligible budgets, 64 has the highest trainable
  positions/s, so it is the leader.
- **Override (128 vs 64):** the combined threefold + fifty + truncated rate
  improves by 0.016, and the override needs ≥ 0.15. 128 also keeps only
  49.6% of 64's throughput, against a required ≥ 50%. Both conditions fail,
  so there is **no override**.
- 128's more decisive (0.81) and shorter (138 plies) games are recorded but
  do not qualify under the pre-registered thresholds.
- Every cell had 0 inference errors. There was no thermal slowdown (80 °C
  maximum; throttle reasons were power cap and idle only).

256 (MEASURED, not eligible): decisive share 0.625, threefold+fifty 0.031, 0
errors, max 74 °C. Its throughput is single-wave and tail-biased, as
pre-declared. Artifacts: `docs/evidence/phase4/search-v2/`.

**FROZEN F10 search budget: 64 simulations/move.** The self-play contract
is otherwise unchanged: c_puct 1.0, temperature 1.0 through ply 29 then
argmax, root Dirichlet 0.3 / 0.25, ply cap 512.

INFERRED:

- With a zero-value, near-uniform network, trainable target entropy at
  64–128 sims (~3.0) is close to the prior entropy (~3.1). Visits spread
  nearly uniformly except where search finds forced mates.
- Purposeful targets require the value head to learn. That is exactly what
  the smoke's per-cycle `root_search` diagnostics are for.

## Addendum B — bounded performance investigation (MEASURED)

Rule: keep an engineering change only if it proves science parity AND gains
≥ 10% real self-play eval/s or trainable pos/s, or ≥ 15% forward latency, or
a substantial resource improvement (addendum D2/D3). Evidence:
`docs/evidence/phase4/perf/`.

**B3 — why v2 eval/s is lower than v1 (MEASURED from the sweep JSONs).**

| cell | ev/s | mean forward ms | batch mean / p50 | owner busy in forward | queue p50 µs |
|---|---:|---:|---|---:|---:|
| v1 s16 | 1325 | 13.8 | 21.0 / 30 | 87% | 1082 |
| v2 s16 | 948 | 15.6 | 19.8 / 24 | 75% | 4235 |
| v1 s64 | 1064 | 14.8 | 20.4 / 25 | 77% | 2177 |
| v2 s64 | 898 | 15.5 | 17.9 / 15 | 78% | 4133 |

- The model forward is not materially slower: v2's 14.5–15.6 ms sits inside
  v1's own run-to-run range of 13.8–17.4 ms.
- Evaluations per position are unchanged (≈ sims), so the terminal-node mix is
  not the cause.
- **The drop is smaller mean batches.** Batch p95 is still 32 in every cell.
- INFERRED: the likely cause is wave/tail structure. v2 games are longer and
  more variable in length (mean 186–287 plies vs 174–201), so a larger share
  of each cell runs with fewer than 32 live games.
- B7 (below) rules out CPU tree work as the cause.

**B4 — per-forward relative-index rebuild (MEASURED; NOT KEPT).** In-process
A/B, synchronized per forward, 50 iterations, 3 fresh processes, F10 R1:

| batch | rebuilt ms | cached ms |
|---:|---|---|
| 16 | 10.14 / 10.15 / 10.19 | 10.01 / 10.02 / 10.06 |
| 32 | 14.17 / 14.28 / 15.80 | 14.11 / 14.17 / 15.78 |

- The gain is ≈ 1% at batch 16 and ≈ 0–0.5% at batch 32, well below the
  threshold.
- The cache was implemented and its outputs were bit-identical (parity test),
  but it was **reverted**: tiny gain, added complexity.

**B9 — per-process forward variance (MEASURED).**

- Same binary and shapes: batch-32 forward was 14.1–14.3 ms in two processes
  and 15.8 ms in the third (+11%).
- The first-call ("cold") time was 0.95–1.71 s.
- INFERRED: this is consistent with CubeCL autotune choosing per process.
- Methodology consequence: schedule comparisons need repeated processes, and
  differences under ~10–15% are noise (P4.3 already treated < 5% as ties and
  found ±8–15% run-to-run spread).
- No framework change.

**B7 — CPU search costs versus game length (MEASURED, `bench-core`).** Search
clones the full `GameState`, including history, per expanded node.

| ply | history | clone+apply /s | clone /s | legal_actions /s | encode /s |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 10.8 M | 46.6 M | 4.95 M | 3.07 M |
| 50 | 51 | 3.18 M | 10.0 M | 4.38 M | 1.87 M |
| 100 | 101 | 1.68 M | 5.11 M | 3.15 M | 1.97 M |
| 200 | 201 | 1.31 M | 2.37 M | 14.9 M (3 legal) | 1.98 M |

- Clone cost grows about 20× by ply 200.
- In absolute terms, the per-tree-step CPU work (clone + apply + encode +
  legal-move generation) is about 1.5–2 µs. Each network evaluation costs
  about 0.75 ms of GPU time amortized over the batch (15 ms / batch ≈ 20).
- **CPU tree work is not the throughput bottleneck.** No change was made; it
  is recorded as a future memory and allocation item.

**B6 — inference pipeline.** The owner is busy in forward for 75–87% of
collection wall time (MEASURED). Pipelining could recover at most the
remaining idle or assembly share, which is also where the tail effect lives.
Instrumentation and implementation are deferred (backlog).

**B8 — more OS threads.** Not revisited: P4.3 measured CPU oversubscription
at 48–64 threads. Multi-leaf search / virtual loss changes search semantics
and is deferred until after the corrected F10 baseline.

**B5 — sparse legal-only policy scoring.** NOT RUN. Given B3/B9, the dense
64×64 grid is not an identified bottleneck, and the smoke comes first (E1).

## P4.5 pre-registration — corrected F10 smoke (written before the run)

Config: `configs/phase4/f10-smoke.toml` (frozen). Reference: v2 `d22c78bd…`.
Science: F10 head v2, 64 sims, c_puct 1.0, temperature 1.0 through ply 29
then argmax, root Dirichlet 0.3 / 0.25, ply cap 512, seed 1, seed policy
`base_seed_plus_global_game_id_v1`. Schedule (MEASURED P4.3):
32 / 32 / 500 µs, cpu_workers 32, learner 64 × 4 (effective 256). AdamW
contract unchanged; lr 3e-4; replay 100k; reuse target 2.0; sampler
unchanged; openings-v1; conservative-v2 promotion with floor 0.5 and
≥ 4 decisive games.

**Workload derivation (INFERRED from v2-s64, MEASURED as 177.6 trainable
positions per game):**

| quantity | value | reason |
|---|---|---|
| games_per_cycle | 32 | one wave at concurrency 32; ≈ 9–10 min self-play |
| expected new trainable positions / cycle | ≈ 5,685 | 32 × 177.6 |
| requested updates / cycle | ≈ 45 | ⌈5,685 × 2.0 / 256⌉ |
| max_updates (safety cap) | 150 | ≈ 3.3× the request; a config test pins that 3× the expected workload still fits |
| planned / warmup updates | 90 / 10 | 2 cycles × 45 |
| arena_games | 32 | v2-s64 self-play was 70% decisive; noise-free argmax arenas may differ, so expect ~16–22 decisive games, well above the gate of 4; covers 12 openings × 2 colours; ≈ 8–9 min per searched arena at 64 sims (100 games would be ≈ 25+ min per arena) |
| run_budget_minutes | 50 | 2 cycles estimated at 40–55 min; evaluation does not check the deadline, so any overrun is recorded |
| position_budget | 30,000 | ≈ 2.5× the expected 11,900 |

**Gate (unchanged from the task):**

- **SYSTEM:** zero illegal moves, replay audit clean, zero inference
  failures, no NaN/Inf, no checkpoint or optimizer mismatch, stable GPU
  resources.
- **DATA:** sufficient trainable positions, acceptable truncation, targets
  not collapsed (the P4.4 degeneracy thresholds apply), repetition not
  beyond threefold + fifty > 0.80.
- **TRAINING:**
  - achieved reuse ≥ 0.8 × target
  - `max_updates_cap_bound` false; an unexpected binding cap means
    CONDITIONAL or NO-GO
  - finite losses and gradients
  - candidate model_id changes after training
  - no immediate policy collapse (trainable target / raw policy entropy not
    collapsing)
- **EVALUATION:** every requested evaluation completes; uninformative results
  are labelled; no false promotion; parent and frozen-reference comparisons
  stay distinct.
- **STOP immediately** on an illegal move, replay corruption, NaN/Inf, a
  checkpoint or optimizer mismatch, a CUDA error, repeated OOM, lineage
  corruption, or scientific identity drift.

Diagnostics that are **not gates** (addendum C7/C8/E2):

- per-cycle `root_search`: noise vs search movement against the generating
  snapshot
- `search-gain` per cycle range: network-to-target divergence, predicted
  W/D/L, |value|, halfmove clock, repeated-position share
- per-update gradient norms (T1 revisit)
- replay freshness
- GPU per phase
- evaluation owner counts

The E2 questions are answered from these after the run.

**Lifecycle / owner residency:** the P4.4L result decides whether
`evaluate_candidate` is changed before the smoke. Any change must be
science-preserving, with a deterministic parity test.

## Owner-approved next steps after the smoke (2026-09-24; NOT RUN)

In this order, each measured before and after. Engineering changes must prove
science parity; science changes get a new experiment identity and are never
mixed into the recorded Phase 4 smoke evidence.

**Added by the smoke result (owner decision needed first):** the searched-arena
/ promotion contract. Arena play is deterministic, noise-free argmax; 24 of 32
arena games were threefold, leaving 3–5 decisive games, so the conservative-v2
gate cannot recognise an improved candidate. Options include:

- arena opening diversity or exploration
- more arena games
- a different acceptance statistic

Each is a science change with its own identity. It must be decided before
P4.6, because otherwise a longer qualification would repeat held cycles.

1. **Throughput.** Multiple leaves in flight per game (virtual loss) for
   larger batches without more OS threads. This is a search-execution change,
   so it needs its own ADR and identity. Kernel fusion / launch-overhead
   investigation (engineering) runs alongside.
   - Evidence: ~14 trainable pos/s at 64 sims; the forward runs at ~20–25% of
     peak and is launch-bound; the owner is busy in forward 75–87% of the time
     at batch ~18.
2. **Cross-cycle tail waste.** Overlapping collection or a continuous
   actor/learner. Evidence: single-wave cycles are tail-bound (P4.3A vs
   P4.3B, 2× throughput difference).
3. **Evaluation deadline enforcement** (D38). Correctness; required before
   P4.6.
4. **Crash-safe replay archival** (D37). Correctness; required before any 24h
   run.
5. **Inference-owner residency** (≤ 2 resident owners). Scope set by the
   P4.4L result.

## B2 — schedule transfer check at the frozen 64 sims (MEASURED)

Binary `eba8c8e`, reference v2, 64 sims, 500 µs, cap = concurrency, at least
2 waves per cell. Artifacts: `docs/evidence/phase4/transfer/`.

| schedule | games | trainable pos/s | ev/s | batch mean/p50/p95 | queue p50 / p95 µs | fwd ms | mean plies | VRAM MiB | °C |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| 24 / 24 | 48 | 15.5 | 984 | 15.9 / 24 / 24 | 729 / 14169 | 12.6 | 182.8 | 787 | 77 |
| **32 / 32 (frozen)** | 64 | 14.4 | 956 | 17.9 / 15 / 32 | 3166 / 14783 | 14.5 | 185.6 | 1041 | 79 |
| 48 / 48 | 96 | **19.7** | **1289** | 24.7 / 33 / 39 | **18569** / 22513 | 15.5 | 177.6 | 1043 | 80 |

**D2 science-parity check (MEASURED, PASS).** The 32 / 32 cell on the
rebuilt binary (all addendum code) reproduced the earlier v2-s64 cell:

- same 64 games, 11,880 positions and 11,368 trainable positions
- W/D/B/T 21/18/24/1 and identical terminations
- identical scientific hash
- target entropy equal to ~1e-15 (floating-point summation order across
  threads)

**Findings.**

- 24 vs 32 is a tie.
- 48-way is +35–37% over 32, beyond the ±10–15% noise band, with bigger
  batches despite the P4.3 oversubscription signature (queue p50 18.6 ms).
- INFERRED: at 64 sims each thread waits longer on the GPU, so extra threads
  fill batches.
- **Not a replacement yet:** the cells used different game sets (48 / 64 /
  96 games). The rule requires a *clear, reproducible* improvement, so the
  confirmation is 32 vs 48 on the identical 96-game set.
- It **cannot apply to the pre-registered smoke**: 32 games per cycle caps
  concurrency at 32, and `games_per_cycle` is scientific. The smoke keeps
  32 / 32 / 500 exactly as frozen.
- Recorded as the first measured lead for post-smoke item 1 (throughput).

**First exact search-contribution numbers at T0** (32 / 32 cell, trainable
plies; `root_search`):

| split | KL | argmax changed |
|---|---:|---:|
| network → noisy root prior (noise alone) | 0.068 | 91.2% |
| noisy root prior → visit target (search) | 0.022 | 11.4% |
| network → target (combined, the old metric) | 0.116 | 91.3% |

At T0 the combined metric is almost entirely exploration noise: with a flat
prior, noise flips the argmax. Search movement is small because the network
value is exactly 0; the mean |root value| is 0.003, from terminal
discoveries.

## P4.4L — GPU inference-owner lifecycle (MEASURED)

**Pre-fix probe: NO-GO.** Binary `eba8c8e`; `bench-lifecycle --reps 8
--arena-games 32 --concurrency 32 --max-batch 32 --timeout-us 500
--simulations 8`. The 8 sims are probe-only: the probe measures owner
residency and latency, not search quality. Artifact:
`docs/evidence/phase4/lifecycle/p44l-prefix.json`.

| mode | owners | VRAM after shutdown, rep 0 → rep 7 (MiB) | growth / rep | fwd ms | errors |
|---|---|---|---|---:|---:|
| one | 1 | 979 → 2803 | +64 … +352 | 12.2–13.8 | 0 |
| two | 2 | 2995 → 4117 | ≈ +160 | 12.8–12.9 | 0 |
| pilot (parent == reference) | 3 | 4597 → 6997 | +160 … +416 | 12.4–12.9 | 0 |
| pilot-promoted | 3 | 7478 → 10459 | +229 … +512 | 12.7–13.1 | 0 |

- VRAM never returned after an owner shut down, and it rose monotonically
  across all 32 lifecycles: 453 → 10,459 MiB.
- Latency was stable and there were 0 errors, so memory grew silently until
  the device would fill.
- INFERRED: at pilot scale (about 5 owners per cycle) that is ~1–2 GB per
  cycle, so a 16 GB device would be exhausted in about 10 cycles. This very
  likely explains the HP branch's "late-run degradation and near-full VRAM".

**Root cause (pinned framework source, cubecl 0.10).**

- `StreamId` is a thread-local id from an incrementing counter
  (`cubecl-common/src/stream_id.rs`).
- Streams, and the memory pools behind them, are indexed by
  `thread id % max_streams` with `max_streams = 128`
  (`cubecl-runtime/src/stream/base.rs`, `config/streaming.rs`).
- `memory_usage` and `memory_cleanup` are scoped to the calling thread's
  stream (`cubecl-runtime/src/client.rs`).
- Every `InferenceOwner` runs its model on a **fresh OS thread** that exits
  at shutdown. Each owner's stream pool was therefore orphaned: never reused,
  never released.

**Fix (A-class correctness, commit `7a8b492`).**

- `impl Drop for BatchedModel` calls `B::memory_cleanup(device)`. `Drop`
  runs on the owner thread, so it targets that thread's stream.
- Parameter buffers belong to the loading thread's stream and are
  unaffected.
- The fix is allocator-only, so there is no numeric effect.

**Diagnostic confirmation (MEASURED)**, same `one`-mode probe on the fixed
binary: VRAM after shutdown was 981 → 981 → 981 → 981 → 988 → 897 → 948 →
948 MiB, a plateau (it was 979 → 2803). Forward latency was 11.9–12.6 ms, 0
errors. Artifact: `docs/evidence/phase4/lifecycle/diag-cleanup-one.json`.

**Post-fix full probe (MEASURED).** Binary `7a8b492`, same protocol.
Artifact: `docs/evidence/phase4/lifecycle/p44l-postfix.json`.

| mode | VRAM after shutdown, rep 0 → rep 7 (MiB) | max growth / rep after rep 2 | fwd ms (reps 1–2 median → range) | errors |
|---|---|---:|---|---:|
| one | 916 → 961 → … → 1008 → 777 | ≤ 32 | 12.39 → 11.57–**14.66** | 0 |
| two | 777 (flat) → 664 → 666 | 0 | 12.66 → 11.86–12.86 | 0 |
| pilot | 858 → 922 → 894 (flat) | 0 | 12.68 → 12.09–13.05 | 0 |
| pilot-promoted | 926 → 958 → 994 → 1008 → 994 | ≤ 14 | 12.12 → 11.97–12.91 | 0 |

- **VRAM: PASS in every mode.** Plateau at 0.66–1.01 GB across all 32
  lifecycles (it was 10.5 GB), with no monotonic rise.
- **Latency:** PASS for `two`, `pilot` and `pilot-promoted`. `one` rep 7
  was +18% (14.66 ms), which violates the pre-registered ±15% band. It is
  isolated: reps 4–6 were −6%, and the following `two` reps were within
  band.
- Per the strict rule, the violation is not reinterpreted. The `one` mode
  is **re-measured** below.

**`one`-mode re-measurement (MEASURED).** Forward latency was
11.21–11.33 ms on every rep (within ±1% of the reps 1–2 median), and VRAM
plateaued at 602–634 MiB. The earlier +18% rep did not reproduce;
INFERRED: an isolated per-process / autotune spike of the kind measured in
B9. Artifact: `docs/evidence/phase4/lifecycle/p44l-postfix-one-repeat.json`.

**P4.4L: GO.**

- VRAM plateaus in every mode (0.6–1.0 GB), where the pre-fix run grew to
  10.5 GB.
- Latency is stable on re-measurement, and there were 0 errors.
- Owner residency stays at 3 in the pilot evaluation. With per-owner
  cleanup it no longer accumulates. The ≤ 2-owner refactor is a peak-memory
  optimization and is scheduled after the smoke (owner-approved item 5).

## P4.5 — corrected F10 smoke (MEASURED)

`recur64 pilot --config configs/phase4/f10-smoke.toml --run-dir runs/phase4-f10-smoke`.

- Binary `7a8b492` (clean, `main`), started 2026-09-24 17:01.
- Status **completed**, 2 cycles, wall **44 m 25 s** against a 50 min budget
  (overrun 0 s).
- Artifacts are in `docs/evidence/phase4/smoke/`: identity, lineage, T0,
  cycle reports, pilot report, replay identity, GPU log and per-cycle
  search-gain.

**Identity.**

- scientific `5548bfaf796a4a3da88da03407ef6ee0e7198b572d30de022d6dae018fa723a3`
- resolved `63b446c4ee428e34b5c3bc0c2061b7e9d729e1ffffce4f86d6473e9504fec428`
- reference `d22c78bd…`
- opening digest `66d6dcf5…`
- promotion rule `conservative-v2`

Both lineage records carry the same scientific hash.

**T0 under the smoke config.** Raw policy vs random: 1 W / 29 D / 2 L,
score 0.484, 3 decisive, informative. Policy entropy 3.285 of 3.291
uniform; top-1 0.046.

| | cycle 0 | cycle 1 |
|---|---|---|
| **Self-play** | | |
| games req / done / failed | 32 / 32 / 0 | 32 / 32 / 0 |
| positions / trainable / trainable games | 6394 / 6394 / 32 | 5486 / 4974 / 31 |
| W / D / B / T | 9 / 12 / 11 / 0 | 12 / 6 / 13 / 1 |
| terminations | mate 20, insuff 9, fifty 2, threefold 1 | mate 25, insuff 5, threefold 1, trunc 1 |
| draw share / repetition share / mean plies | 0.375 / 0.031 / 199.8 | 0.188 / 0.031 / 171.4 |
| mean halfmove clock / repeated-position share | 7.32 / 0.002 | 5.42 / 0.001 |
| **Targets** | | |
| all H / top-1 | 2.924 / 0.143 | 2.973 / 0.142 |
| trainable H / top-1 | 2.924 / 0.143 | 3.017 / 0.135 |
| **Root search** (trainable) | | |
| KL(noisy ‖ net) / argmax changed by noise | 0.067 / 90.6% | 0.068 / 92.0% |
| KL(target ‖ noisy) / argmax changed by search | **0.019 / 11.0%** | **0.025 / 11.8%** |
| mean abs network value / root value | 0.000 / 0.002 | 0.000 / 0.004 |
| **Inference** | | |
| requests / errors | 407,109 / 0 | 349,239 / 0 |
| batch mean / p50 / p95 / max | 13.0 / 14 / 30 / 32 | 10.6 / 7 / 32 / 32 |
| queue p50 / p95 us; forward ms | 5260 / 9809; 15.3 | 4858 / 10895; 16.0 |
| **GPU** | | |
| VRAM start, then peak in collect / train / eval (MiB) | 602; 602 / 2780 / 3245 | 3245; 3294 / 3242 / 3274 |
| mean busy util collect / train / eval; max C | 48 / 73 / 47 %; 74 | 49 / 81 / 45 %; 69 |
| **Replay** | | |
| total games / sampleable positions | 32 / 6394 | 64 / 11368 |
| current-cycle sample fraction / mean sample age (cycles) | 1.000 / 0.00 | 0.669 / 0.33 |
| reuse requested / achieved | 2.0 / 2.002 | 2.0 / 2.007 |
| **Updates** | | |
| requested / scheduled / completed / cap / cap_bound | 50 / 50 / 50 / 150 / false | 39 / 39 / 39 / 150 / false |
| **Training** | | |
| examples consumed | 12,800 | 9,984 |
| total loss, first to last | 4.059 to 3.849 | 4.193 to 4.090 |
| policy loss, first to last | 2.961 to 3.022 | 3.095 to 3.115 |
| **WDL loss, first to last** | **1.099 to 0.826** | **1.099 to 0.974** |
| policy entropy, first to last | 2.958 to 3.021 | 3.091 to 3.114 |
| pre-clip grad norm max / mean | 6.85 / 1.83 | 3.98 / 1.93 |
| LR, first to last | 0 to 2.0e-4 | 0 to 2.0e-4 |
| **Evaluation** | | |
| searched candidate vs parent: W / D / L, score | 1 / 27 / 4, 0.453 | 1 / 29 / 2, 0.484 |
| decisive / informative / terminations | 5 / true / threefold 24, fifty 3, mate 5 | 3 / true / threefold 24, fifty 5, mate 3 |
| searched candidate vs reference | = parent arena (parent is the reference; labelled) | = parent arena (labelled) |
| raw candidate vs random | 3 / 27 / 2, 0.516, 5 decisive | 2 / 26 / 4, 0.469, 6 decisive |
| raw candidate vs parent | 0 / 26 / 6, 0.406, 6 decisive | 0 / 31 / 1, 0.484, 1 decisive |
| **Lineage** | | |
| parent to candidate | d22c78bd to 9fe5a1f4 | d22c78bd to e8675fab |
| decision / hold reason | hold / score_not_above_parent | hold / arena_uninformative |
| optimizer step start to end; accepted after | 0 to 50; 0 | 0 to 39; 0 |
| eval owners spawned / max resident | 3 / 3 | 3 / 3 |
| **Time** | | |
| collect / train / eval / wall (s) | 664 / 32 / 667 / 1364 | 723 / 22 / 547 / 1292 |

- **Independent GPU log:** peak 3,276 MiB, max 74 C. Throttle reasons were
  idle and power cap only.
- **Replay audit:** 64 games, 11,880 plies, 0 errors.
- **Replay sidecar:** verified. Both cycles were generated by reference v2
  at 64 sims, argmax after ply 30, epsilon 0.25, head v2.

### Gate evaluation

| gate | result |
|---|---|
| zero illegal moves; replay audit clean | PASS (audit: 0 errors) |
| zero inference failures | PASS (0 of 756,348) |
| no NaN / Inf | PASS |
| no checkpoint or optimizer mismatch | PASS (the accepted trajectory stays at 0 on holds, by design) |
| stable GPU resources | PASS (3.2-3.3 GB across both cycles; D44 confirmed in a real pilot) |
| sufficient trainable positions; acceptable truncation | PASS |
| targets not collapsed; repetition below 0.80 | PASS (H about 3.0; threefold + fifty 0.03-0.09) |
| intended reuse achieved; cap not controlling | PASS (2.00 / 2.01; cap_bound false) |
| finite losses and gradients; model id changes | PASS |
| no immediate policy collapse | PASS (policy entropy 2.96 to 3.11) |
| all evaluations complete; uninformative results labelled; no false promotion | PASS |
| parent and frozen-reference comparisons distinct | N/A: with no promotion, parent == reference, and the reuse is labelled |

### E2 questions

1. **Does policy loss fall?** It is flat to slightly rising (2.96 to 3.02,
   3.10 to 3.12). That is interpretable: the targets are near-uniform (H
   about 3.0, the same as the model's entropy), so there is little policy
   signal yet.
2. **Does WDL loss learn?** **Yes.** It fell from 1.099 (uniform) to 0.826
   in cycle 0 and to 0.974 in cycle 1.
3. **Does value move off neutral?** The trained candidates' WDL loss fell,
   so their value predictions moved. The candidates' value distribution on
   positions was NOT MEASURED: `search-gain` only evaluates the generating
   network. The self-play networks stayed exactly neutral (|value| 0.000)
   because no candidate was promoted.
4. **Does search movement grow as value learns?** NOT TESTABLE in this
   smoke. Both cycles were generated by the untrained reference. Search
   moved the argmax on about 11% of positions (KL about 0.02) in both.
5. **Is repetition low?** In self-play, yes: 3% of games, and at most 0.2%
   of positions repeated. **In the searched arenas, no: 24 of 32 games
   ended in threefold** in both cycles.
6. **Are targets purposeful?** Not yet. They are broad, close to the prior.
7. **Does the raw policy improve on T0?** No: 0.516 and 0.469 vs 0.484.
8. **Is there decisive candidate-vs-parent evidence?** Weak: 5 and 3
   decisive games out of 32.
9. **Is replay fresh?** Healthy: the current-cycle fraction was 0.67 and
   the mean sample age 0.33 cycles.
10. **Is max_updates only a safety cap?** Yes (50 of 150, 39 of 150).
11. **Is GPU throughput stable across cycles?** Stable within the one-wave
    tail effect: collection went from 9.6 to 7.6 positions/s, and batch mean
    from 13.0 to 10.6 as games ended.
12. **Is the owner lifecycle stable?** Yes: VRAM went from 3,245 to 3,274
    MiB across cycle 1.

### Verdict: **F10 SMOKE CONDITIONAL**

The corrected F10 learning loop is **interpretable**: every system, data and
training gate passes, and the value head learns. Two measured conditions
stop learning from compounding under the current contract. Both must be
resolved before P4.6.

1. **The searched evaluation arena has its own repetition attractor.**
   - Arena play is deterministic argmax with no root noise. Two
     near-identical, near-zero-value networks shuffle into threefold
     repetition: 24 of 32 games, in both cycles.
   - The conservative-v2 gate therefore sees only 3-5 decisive games and
     held both cycles, correctly on that evidence.
   - Fixing this is an evaluation-contract (science) change: for example,
     arena opening diversity or exploration, more games, or a different
     acceptance statistic. It needs an owner decision and a new identity.
2. **Without promotion, learning does not accumulate.**
   - By design (D31), a held candidate is discarded. Each cycle retrains
     from the untrained reference, and self-play never uses a learned value
     head.
   - So the central question, whether search movement rises once the value
     head learns, cannot be answered until a candidate is promoted or the
     acceptance protocol is revisited.

INFERRED:

- The pipeline itself is no longer the blocker.
- The policy target can only become purposeful once the value head guides
  search.
- The value head can only guide search once the evaluation protocol can
  recognise a better candidate.

The owner-approved post-smoke items (throughput, tail waste, D38, D37,
residency) stand. The arena / promotion contract is added as the first
scientific decision before P4.6.

**STOP.** No P4.6, R10 or 24h run was started.

## D45 — arena evaluation contract (post-smoke; owner-approved Option A)

**Pre-registration (written before any run).**

- Model pair from the smoke: reference `d22c78bd` vs cycle-1 candidate
  `e8675fab`.
- `configs/phase4/f10-smoke.toml` science: 64 sims, 32 games, openings-v1,
  paired colours, measured 32 / 32 / 500 µs schedule.
- Tool: `recur64 eval-arena`, which runs the pilot's own batched arena path.

| variant | contract |
|---|---|
| V0 | current: argmax from ply 0, no noise, seed offset 1 (= smoke cycle 1). Deterministic, so it must reproduce the smoke's cycle-1 arena exactly (1 / 29 / 2, threefold 24), which also validates the tool path. |
| V1 | sample (T = 1) for the first 30 plies after the opening, then argmax; no noise |
| V2 | V1 plus root Dirichlet α 0.3, ε 0.25 (the full self-play exploration contract) |

**Rule.**

1. Adopt the least-perturbing variant (V1 before V2) with decisive share
   ≥ 0.5 **and** threefold share ≤ 0.3, 0 inference errors and truncation
   ≤ 0.1.
2. If neither qualifies, adopt the variant with the highest decisive share,
   provided it cuts the threefold share by ≥ 0.3 absolute vs V0.
3. Otherwise, adopt nothing and return the decision to the owner.

The identity of an adopted variant is new (see the config test):
pre-D45 hashes stay reproducible.

### D45 result (MEASURED): V2 adopted by the pre-registered rule

Binary `f269140`, reference `d22c78bd` vs candidate `e8675fab`, 64 sims, 32
games, seed offset 1, 0 inference errors in every variant. Artifacts:
`docs/evidence/phase4/d45/`.

| variant | W / D / L | decisive | threefold | fifty | truncated | score (95% CI) | secs |
|---|---|---:|---:|---:|---:|---|---:|
| V0 current | 1 / 29 / 2 | 3 (0.09) | 24 (0.75) | 5 | 0 | 0.484 (0.43-0.54) | 481 |
| V1 sample 30 | 5 / 24 / 3 | 8 (0.25) | 19 (0.59) | 5 | 0 | 0.531 (0.44-0.62) | 438 |
| **V2 sample 30 + root noise 0.25** | 12 / 8 / 12 | **24 (0.75)** | **0 (0.00)** | 3 | 0 | 0.500 (0.35-0.65) | 519 |

- **V0 reproduced the smoke's cycle-1 arena exactly** (1 / 29 / 2; mate 3,
  fifty 5, threefold 24). So `eval-arena` runs the pilot's arena path
  deterministically.
- **V1 fails the rule.** Sampling only the opening phase leaves the
  deterministic argmax phase to collapse into repetition between
  near-identical, near-zero-value networks.
- **V2 passes both criteria** (decisive 0.75 >= 0.5; threefold 0.00 <= 0.3),
  so **V2 is adopted**.
- **What V2 shows:** with real decisive evidence (24 games), the 39-update
  candidate scores exactly 0.500 against the untrained reference. It is not
  measurably stronger, a conclusion the V0 arena could not reach.
- **Trade-off (INFERRED):** per-move noise widens the score interval (±0.15
  vs ±0.05 at 32 games). The conservative gate can now decide, but only on a
  clear margin.

## D47 - multi-leaf search throughput (pre-registration, written before any run)

**Setup.** Reference v2 `d22c78bd`, `configs/phase4/f10-reference.toml`
self-play contract (argmax after 30, root noise 0.25), 64 sims, 64 games
(2 waves), concurrency 32, timeout 500 us, same seeds.

| cell | leaves_in_flight K | batch cap |
|---|---:|---:|
| baseline | 1 | 32 |
| D47 | 2 | 64 |
| D47 | 4 | 128 |

K = 1 is re-measured in the same batch because of the ±10-15% per-process
variance (B9).

**Rule.** Recommend adoption of the smallest K > 1 that gains >= 10%
trainable positions/s over K = 1 **and** keeps data health, meaning all of:

- decisive share within 0.10 of K = 1
- threefold + fifty not higher by more than 0.05
- trainable target entropy within 10%
- 0 inference errors

Adoption itself is an owner decision: K > 1 is a search-execution change and a
new identity.

### D47 result (MEASURED)

Binary `848bce7`, reference v2, 64 sims, 64 games, concurrency 32, same
seeds, 0 inference errors in every cell. Artifacts:
`docs/evidence/phase4/d47/`.

| K | batch cap | trainable pos/s | vs K=1 | ev/s | batch mean/p50/p95 | queue p50 / p95 ms | fwd ms | wall s | VRAM MiB | max C |
|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| 1 | 32 | 12.43 | - | 827 | 17.9 / 15 / 32 | 4.1 / 14.5 | 16.8 | 915 | 731 | 79 |
| **2** | 64 | **22.79** | **+83%** | 1451 | 26.5 / 33 / 42 | 18.1 / 30.6 | 15.8 | 518 | 746 | 79 |
| 4 | 128 | 26.76 | +115% | 1703 | 32.2 / 33 / 47 | 40.8 / 58.9 | 16.9 | 397 | 746 | 80 |

| K | decisive | threefold + fifty | truncated | mean plies | trainable H | top-1 | search-moved argmax | KL(target ‖ noisy) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.703 | 0.062 | 1 | 185.6 | 2.965 | 0.139 | 11.4% | 0.0219 |
| 2 | 0.719 | 0.062 | 0 | 184.4 | 2.951 | 0.141 | 11.3% | 0.0220 |
| 4 | 0.750 | 0.047 | 0 | 165.9 | 2.991 | 0.134 | 12.1% | 0.0246 |

**Rule outcome:**

- **K = 2 recommended**: the smallest K with at least +10% trainable pos/s
  (+83%) whose data health stays within every bound.
- K = 4 is also healthy, but it adds only +17% over K = 2, doubles queue
  p95, and leaves batch p50 at 33 (diminishing returns).
- INFERRED: this confirms the throughput root cause. Batches were capped by
  one evaluation in flight per search thread, and the forward cost per batch
  barely changes (15.8-16.9 ms) as batches grow.

**Adoption is an owner decision** (a search-execution change and a new
identity).

## F10 smoke v2 - pre-registration (written before the run)

`configs/phase4/f10-smoke-v2.toml` (frozen) is the P4.5 contract plus the
owner-adopted post-smoke decisions:

- **D45 arena:** sample 30 plies, then root noise 0.25 on every move.
- **D47 search:** K = 2 leaves per round, batch cap 64.
- **D48 trainer:** continuous.

Everything else is unchanged: reference v2, 64 sims, c_puct 1.0, self-play
exploration, measured schedule (32 concurrency, cpu_workers 32, 500 us,
learner 64x4), lr 3e-4, reuse 2.0, openings-v1, conservative-v2 with a 0.5
floor and at least 4 decisive games.

**Workload, re-derived from the measured D47 K = 2 cell:**

| item | value |
|---|---|
| games per cycle | 64 (two waves, about 9 min) |
| expected trainable positions per cycle | about 11,370 |
| requested updates per cycle | about 89 |
| max_updates | 300 (about 3.4x) |
| cycles | 3 |
| planned / warmup updates | 267 / 27 |
| arena | 32 games |
| budget | 60 min |
| position budget | 90,000 |

**Gate:** the same system, data, training and evaluation gates as P4.5.
With D48, `max_updates_cap_bound` must be false every cycle, and the trainer
step must advance continuously (cycle n starts at cycle n-1's end step).

**Learning questions (answered afterwards, not gates):**

- Does the WDL and policy loss keep falling across cycles as training
  accumulates?
- Is any candidate promoted, meaning the arena gives decisive evidence
  above 0.5?
- After a promotion, does search movement beyond the noise
  (`argmax_changed_by_search`, KL(target || noisy)) rise, with a non-zero
  network value in self-play?
- Does raw policy strength against random or the parent move off chance?
- Does self-play and arena repetition stay low?

## F10 smoke v2 (MEASURED)

`recur64 pilot --config configs/phase4/f10-smoke-v2.toml --run-dir runs/phase4-f10-smoke-v2`

- **Binary:** `97d58eb` (clean, `main`).
- **Hashes:** scientific `4f4969f850bd3eef5f924ea4d749f2e126b84caf583a444be58a0dceb47a1709`,
  resolved `5d43653d6aac4311c2c821a46203565738a6ad4cf6760c1958a81facd937c1dd`.
- **Reference:** `d22c78bd`.
- **Outcome:** completed, 3 cycles in 61.0 min against a 60 min budget.
  - Overrun 60 s: in-flight evaluation games finished after the soft
    deadline (D38). Recorded, not hidden.
- **Replay audit:** 192 games, 39,336 plies, 0 errors.
- **Inference:** 0 errors in every cycle.
- **GPU:**
  - VRAM 3.2-3.6 GB, stable across cycles.
  - Maximum temperature 80 C.
  - Evaluation used at most 2 resident models (D46).
- **Artifacts:** `docs/evidence/phase4/smoke-v2/`.

| cycle | plies / trainable | W/D/B/T | draw share | threefold+fifty | mean plies | trainable H / top-1 | mean abs net value | search-moved argmax / KL(target‖noisy) | trainer step | WDL loss | policy loss |
|---:|---|---|---:|---:|---:|---|---:|---|---|---|---|
| 0 | 11802 / 11802 | 21/18/25/0 | 0.281 | 0.062 | 184.4 | 2.951 / 0.141 | 0.000 | 11.3% / 0.022 | 0 to 93 | 1.099 to 0.902 | 3.055 to 3.095 |
| 1 | 10807 / 10807 | 25/16/23/0 | 0.250 | 0.062 | 168.9 | 2.971 / 0.140 | 0.000 | 11.4% / 0.023 | 93 to 178 | 0.920 to 0.805 | 3.192 to 3.088 |
| 2 | 16727 / 16215 | 8/47/8/1 | **0.734** | **0.328** | **261.4** | **2.220 / 0.265** | **0.194** | **36.9% / 0.398** | 178 to 305 | **0.677 to 0.629** | 2.976 to 2.923 |

| cycle | searched vs parent (W/D/L, score, decisive, 95% CI) | searched vs frozen reference | decision | raw vs random / raw vs parent | fresh-sample fraction / mean age | collect / train / eval s |
|---:|---|---|---|---|---|---|
| 0 | 14/3/15, 0.484, 29, 0.32-0.65 | = parent arena | hold (score_not_above_parent) | 0.548 / 0.484 | 1.00 / 0.00 | 552 / 59 / 398 |
| 1 | 10/13/7, 0.550, 17, 0.41-0.69 | = parent arena | **promote** | 0.562 / 0.453 | 0.67 / 0.33 | 455 / 47 / 456 |
| 2 | 7/22/3, 0.562, 10, 0.47-0.66 | **8/16/8, 0.500, 16 decisive** | **promote** | 0.531 / 0.484 | 0.50 / 0.67 | 688 / 82 / 909 |

- **Updates** requested/completed: 93/93, 85/85, 127/127. `cap_bound` was
  false in every cycle (cap 300).
- **Reuse** achieved: 2.02, 2.01, 2.01.
- **Lineage:**
  - reference `d22c78bd` (held) → `29516d8e` (promoted, cycle 1) →
    `bc33a27d` (promoted, cycle 2)
  - accepted step 0 → 178 → 305
  - the trainer step advanced continuously: 0 → 93 → 178 → 305

### Gate: PASS

Every system, data and training gate passes:

- 0 illegal moves or inference errors; replay audit clean; all values finite.
- No checkpoint or optimizer mismatch.
- Continuous trainer steps; cap never bound; reuse on target.
- Every requested evaluation completed; no degenerate budget.

### Learning questions (MEASURED unless marked)

1. **Does training accumulate?** Yes.
   - WDL loss: 1.10 → 0.90 → 0.80 → 0.63 across cycles.
   - Policy loss started to fall in cycles 1-2.
   - This is the D48 effect: nothing is reset on a hold.
2. **Are candidates promoted?** Yes, twice (cycles 1 and 2), on the
   pre-registered conservative-v2 rule. Both intervals include 0.5, so the
   evidence is weak.
3. **Does search improve on the policy once the value head guides it?**
   **Yes.** This is the central question behind the whole redesign. With
   the promoted network generating self-play (cycle 2):
   - mean abs network value went from 0.000 to **0.19**;
   - the search-moved argmax rose from 11% to **37%**;
   - KL(target ‖ noisy prior) rose from 0.02 to **0.40**;
   - targets became purposeful: entropy 2.95 → 2.22, top-1 0.14 → 0.27.
4. **Is playing strength above T0?** **Not demonstrated.**
   - Against the frozen reference: 0.500 over 16 decisive games.
   - Raw policy vs random: 0.53 (T0 0.48).
   - The gains are in the mechanics (value learning and search guidance),
     not yet in measurable strength.
5. **Watch item: a draw-weighted drift in cycle 2.**
   - Self-play draw share rose from 0.25 to 0.73, mostly insufficient
     material (26) and fifty-move (16) endings, with longer games (261
     plies).
   - Threefold + fifty is 0.33, still below the 0.80 degeneracy line.
   - INFERRED: a value head trained on a draw-heavy mix now steers search
     toward safe draws. This could be a second form of the draw attractor.
     A longer run must track it per cycle.
6. **Throughput:** self-play ran at 21-24 positions/s, up from 9.6 in the
   first smoke (D47 K=2 plus two-wave cycles).

### Verdict: **F10 SMOKE v2 GO**, for the learning mechanism

- The corrected loop now learns and compounds: the value head trains
  continuously, promotions occur, and search improves on the policy once
  the value head guides it.
- Strength over T0 is not yet shown.
- Self-play draw share is a live risk for any longer run.
- **STOP:** no P4.6, R10 or 24h run was started.

## P4.6 - bounded F10 qualification (pre-registration, written before the run)

`configs/phase4/f10-qual.toml` (frozen) keeps the smoke v2 contract
unchanged: reference v2, 64 sims, D41 self-play exploration, D45 arena
exploration, D47 K = 2, D48 continuous trainer, and the measured schedule.
The config test asserts that it matches smoke v2.

**Execution:**

- 10 cycles of 64 games.
- LR schedule planned over the whole run: 1100 updates with 110 warmup (about
  110 per cycle from the smoke v2 measurements).
- max_updates 400.
- 300 min budget and a 250k position budget. Not a 24h run.
- Replay capacity is 100k positions, so D37 archival is exercised from about
  cycle 7.

**Health stops (D49), checked at each cycle boundary:**

- self-play draw share >= 0.85 in two consecutive cycles;
- threefold + fifty-move >= 0.60 in any cycle;
- truncation >= 0.25 in any cycle.

**Per-cycle gate:** as in smoke v2. `cap_bound` must be false, the trainer
step must advance continuously, and any system failure (inference error,
audit failure, NaN, checkpoint or optimizer mismatch, CUDA error) stops the
run.

**Strength test after the run.** `eval-arena` runs the final promoted
snapshot against the frozen reference `d22c78bd`: 128 games, the same D45
arena contract, seed offset 1000 (disjoint from the cycle seeds).

- A strength claim requires the 95% CI lower bound on the score to be above
  0.5.
- Otherwise the result is "no measurable strength over T0", recorded as
  such.

**Tracked across cycles (not gates):**

- WDL and policy loss
- search movement beyond noise
- mean absolute network value
- target entropy
- draw share and termination mix
- raw policy vs random
- arena score vs parent and vs reference
- replay freshness
- throughput, VRAM and temperature
