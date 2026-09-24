# HP H1 measured results — 2026-09-23

Status: **in progress; no R15 entry decision yet**. This is the authoritative
measured H1 record. MEASURED, INFERRED, and NOT RUN are labelled. Do not infer
F15 learning health from a section that does not say it measured learning.

## Completed and verified

- Harness commits: `01bb6d4` (shared collector and strict inputs), `e5cd3f9`
  (reuse, optimizer continuation, parent evaluation and lineage), `213a943`
  (HP sweep and sampled VRAM). Branch: `experiment/hp-r15`.
- `cargo test --workspace` passed once. Targeted runtime/evaluation tests
  passed after the final harness edits. `cargo clippy --workspace --all-targets`
  completed with two harmless warnings from new code, then those warnings were
  fixed. `cargo build --release -p recur64-cli --features cuda` passed.
- The frozen `configs/openings-v1.toml` suite was validated by a targeted test.
- RTX 2050 was checked idle before the measurement. No competing substantial
  GPU workload was observed. A 0.5 s external `nvidia-smi` log was collected
  locally at `runs/hp-h1-systems-ag12-mb32-to1000/gpu.csv`.

## Measured CUDA scheduling cell

Binary built from the harness at `213a943`; F15 512/8/768, 8 feed-forward
blocks, FP32, standard-start games, 16 simulations/move, 400-ply cap.
16 total games, 12 concurrent games, batch cap 32, timeout 1000 us.
This was an **exploratory** cell: the sweep built fresh random weights. The
next sweep must use one frozen checkpoint for every cell.

| Measure | Value |
|---|---:|
| Games completed / requested | 16 / 16 |
| Positions | 4,535 |
| Collect wall time | 205.015 s |
| Evaluations/s | 351.68 |
| Positions/s | 22.12 |
| Games/hour | 280.96 |
| Inference errors | 0 |
| Peak in-flight | 12 |
| Batch mean / p50 / p95 / max | 7.69 / 7 / 12 / 12 |
| Queue wait p50 / p95 | 7,478 / 10,277 us |
| Mean forward latency | 15,151 us |
| Sampled peak VRAM | 321 MiB |
| Busy-sample mean / max GPU utilization | 54.6% / 80% |
| Maximum sampled GPU temperature | 68 C |
| Checkmate / insufficient / threefold / 50-move / truncated | 1 / 2 / 5 / 6 / 2 |
| Mean target entropy / top-1 visit share | 1.329 / 0.437 |

The measured sweep JSON and GPU monitor CSV are local run artifacts under
`runs/hp-h1-systems-ag12-mb32-to1000/`; the table above is the durable summary.

## Post-reboot resume (correction to the reboot note)

The reboot note said the `freeze-reference` command and `bench-runtime
--checkpoint` were uncommitted. That was stale: both are in `e865be4`, the
commit made just before the reboot. After the reboot the branch was
`experiment/hp-r15` at `e865be4`, in sync with origin. The only worktree
change was the pre-existing `Grok-plan.md` deletion, which is kept out of
every commit. `nvidia-smi` showed the RTX 2050 idle (7 MiB, 0%), with no
compute processes.

## Foundation completion (commit after `e865be4`)

MEASURED on CPU (FP32 Flex):

- **Optimizer continuation through the pilot's real load path.** Train K
  updates, `save_training`, then `load_training` into a *freshly built* module
  (new random weights, new ParamIds) with a fresh AdamW, then continue through
  `train_from_games`. The result is **bit-exact** against the continuous run
  (|Δloss| = 0, |Δweight| = 0). A negative control (same weights, fresh
  moments) diverges (|Δw| = 2.1e-5), so the test can detect lost moments.
  Burn 0.21 restores ParamIds from the record (`Param::load_record`).
- **Promotion (P0 fix).** The old rule promoted on any informative arena with
  score ≥ 0.35. A candidate with 1 loss and 19 draws (0.475) would have
  promoted. The new rule `conservative-v2` requires audit OK, 0 inference
  errors, finite metrics, reuse ≥ 0.8× target, decisive parent-arena games ≥
  `promotion_min_decisive_games` (default 4), and a score strictly above 0.5
  (and ≥ floor). Decisions are `promote` or `hold`, with hold reasons.
- **Frozen reference in the pilot.** `reference_checkpoint` (path,
  operational) plus `reference_model_id` (content hash, scientific). The pilot
  refuses a mismatched id, a mismatched model config, or a trained checkpoint.
  It writes `identity.json` and a T0 baseline (`eval/baseline-t0.json`: raw
  reference vs random plus policy diagnostics on the suite) before cycle 0.
- **Scientific identity v2.** The hash now covers the opening-suite *content*
  digest (not the path), `OPTIMIZER_CONTRACT` (AdamW with Burn 0.21 defaults
  pinned explicitly, wd 1e-4, per-parameter L2 clip 1.0, example-weighted mean
  reduction), the promotion rule version and its parameters, snapshot policy,
  arena games, games_per_cycle, shard_max_games (it sets the sampler's
  recency weights), and reference_model_id. The LR schedule is now one
  function shared by the pilot and the hash. Before this, the pilot trained
  `max(planned, max_updates*cycles)` while the hash recorded something else.
  Excluded: device, concurrency, cpu_workers, inference batch/timeout, labels,
  run_id, cycles, budgets, and the max_updates cap. Tests: a scheduling change
  keeps the scientific hash and changes the resolved hash; search, training,
  and promotion changes change it; the same suite path with new contents
  changes it; provenance text alone does not.
- **Replay/training telemetry.** TrainReport now truthfully reports trainable
  games, skipped (result-less) games, total replay games, and sampleable
  positions. New: `current_cycle_sample_fraction` and
  `mean_sample_age_cycles`, derived from in-memory example game ids (no
  replay schema change).
- **Evaluation informativeness.** Arena, raw-vs-random, and raw-vs-parent
  results carry `decisive_games` and `informative`.
- **Target health** is reported for all plies and for trainable plies, from
  one post-hoc helper used by the pilot, run, selfplay, and the sweep.
- **Concurrent evaluation.** Arena and raw games run on a thread pool
  through batched inference owners. Reports are aggregated in game-index
  order, and a test shows concurrent = sequential on a deterministic
  evaluator. Reason: direct F15 batch-1 inference is 8.5 ms (118 ev/s,
  `runs/hp-f15-cuda/bench.json`). A sequential 20-game 16-sim arena would
  take about 11 min (INFERRED), and each cycle runs two.
- **Provenance fix.** `build.rs` only watched `.git/HEAD`, which does not
  change on commit. A CPU test binary reported `ee250b4` while HEAD was
  `e865be4`. It now watches branch refs and sources, and appends `-dirty`
  when crates or manifests differ from HEAD.
- **Sweep.** The HP grid is revised (see S1). Cells record W/D/L, lengths,
  trainable positions/s, all/trainable target health, and sampled GPU util
  and temperature.
- **Gate.** `cargo test --workspace` passed (1 pre-existing ignored deep
  perft); `cargo fmt --all --check` and `cargo clippy --workspace
  --all-targets` are clean.

## Frozen F15 reference (F1/F2) — MEASURED

`recur64 freeze-reference --config configs/hp/f15-reference.toml --output
runs/hp-h1-ref-f15`, CUDA FP32, built at `1b0faa8`.

| Field | Value |
|---|---|
| model_id | `4271e19fbd6bc32f95017d7808ed14feb4e86d5ae8877193f0b8736139c9dda3` |
| seed | 1 |
| git | `1b0faa86982b3cd403393a25ec3ff285196bebbe`, `experiment/hp-r15` |
| scientific hash (freezing config) | `2b8a3902d05322d9380c513c58181b7490a2afffd683a7b61982bb73fcde8390` |
| resolved hash (freezing config) | `e3587044ebab063c2cd9e9fb9febee04a2c4ddb8828aaa61d42e8df3afa64211` |
| opening suite v1 digest | `66d6dcf5cd805f3283c2d7a6d38305bdbdb3d3ae4b87585b46b596eaa78e327c` |
| update_counter / lr_schedule_step | 0 / 0 |

The directory holds `meta.json`, `config.toml`, `model.mpk`, `optimizer.mpk`,
and `reference.json` (model id, seed, git SHA/branch, both hashes, suite
digest, optimizer contract). Later runs pin the reference by
`reference_model_id` in their own scientific identity.

**Invalid first attempt (kept for the record).** The first freeze ran from a
shell without the process-local CUDA env. cudarc could not load NVRTC. It
panicked only on a worker thread, and the command still wrote a checkpoint
(`cd85d509…`), whose weights differ from the valid one. That directory was
renamed `runs/hp-h1-ref-f15-INVALID-no-nvrtc` and is never used.
`ensure_supported()` now refuses a CUDA run when NVRTC is not on PATH or
`CUDA_PATHin` (verified: the command errors). `scripts/hp-cuda-env.sh`
sets the env for Git Bash.

T0 raw-policy diagnostics on the 12 suite positions (`eval-policy`,
sequential batch-1 CUDA, 8 games): mean policy entropy 2.280 nats (uniform
3.291), mean top-1 probability 0.330. Raw policy vs random: 8 draws (6
insufficient material, 2 stalemate), 0 decisive, `informative = false`, wall
time 4 m 25 s. This is not a strength claim.

## S1 — HP scheduling (MEASURED, gate: GO)

Every cell uses the frozen reference `4271e19f…`, binary `1b0faa8`, F15
FP32 on CUDA, standard start, 16 sims/move, c_puct 1.0, temperature 1.0,
ply cap 400, seed 1, and game seeds `seed + game_index`. Only concurrency,
batch cap, and timeout change. JSON: `runs/hp-h1-s1a/`,
`runs/hp-h1-s1a-ring-*/`, `runs/hp-h1-s1b-*/`.

**Data identity check (MEASURED).** Every 16-game cell produced the same
games: W/D/L 2/13/1, terminations 3/2/9/1/1/0, mean 157.8 plies, target
entropy 1.153. The two 32-game cells also match each other. With the frozen
reference and seed schedule, scheduling changes throughput only, never the
data.

**Grid change.** Search keeps one leaf in flight per game, so batch size is
at most the concurrency (measured maximum = concurrency in every cell). A cap
at or above the concurrency never binds, so the old grid's pairs (16,32)/(16,64)
and (24,32)/(24,64) were duplicates. The coarse pass instead varies
concurrency with a non-binding cap and adds one binding-cap probe.

S1A coarse (16 games per cell, one process, cells run in order):

| conc | cap | to us | sims | games | wall s | ev/s | pos/s | trainable pos/s | games/h | inflight | batch mean/p50/p95/max | wait p50/p95 us | fwd us | err | VRAM | util busy/max | temp | W/D/L | mate/stale/insuf/3fold/50/trunc | mean plies | ent all/train | top1 all/train |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 8 | 1000 | 16 | 16/16 | 147.9 | 269.6 | 17.08 | 17.08 | 389.5 | 6 | 4.98/6/6/6 | 7803/9778 | 11683 | 0 | 353 | 44.0/54 | 67 | 2/13/1 | 3/2/9/1/1/0 | 157.8 | 1.153/1.153 | 0.539/0.539 |
| 8 | 8 | 1000 | 16 | 16/16 | 91.2 | 436.9 | 27.67 | 27.67 | 631.2 | 8 | 6.39/8/8/8 | 124/9079 | 12473 | 0 | 449 | 70.4/90 | 70 | 2/13/1 | 3/2/9/1/1/0 | 157.8 | 1.153/1.153 | 0.539/0.539 |
| 12 | 16 | 1000 | 16 | 16/16 | 107.7 | 370.2 | 23.45 | 23.45 | 534.9 | 12 | 8.37/12/12/12 | 7870/10041 | 16206 | 0 | 513 | 60.5/88 | 66 | 2/13/1 | 3/2/9/1/1/0 | 157.8 | 1.153/1.153 | 0.539/0.539 |
| 16 | 16 | 1000 | 16 | 16/16 | 93.7 | 425.5 | 26.95 | 26.95 | 614.7 | 16 | 10.29/12/16/16 | 3789/10499 | 19133 | 0 | 609 | 69.1/95 | 67 | 2/13/1 | 3/2/9/1/1/0 | 157.8 | 1.153/1.153 | 0.539/0.539 |
| 24 | 32 | 1000 | 16 | 16/16 | 95.3 | 418.5 | 26.50 | 26.50 | 604.6 | 16 | 10.28/12/16/16 | 4340/10513 | 18966 | 0 | 673 | 67.9/89 | 67 | 2/13/1 | 3/2/9/1/1/0 | 157.8 | 1.153/1.153 | 0.539/0.539 |
| 24 | 16 | 1000 | 16 | 16/16 | 92.1 | 432.8 | 27.41 | 27.41 | 625.3 | 16 | 10.29/12/16/16 | 3777/10515 | 18779 | 0 | 769 | 69.9/95 | 67 | 2/13/1 | 3/2/9/1/1/0 | 157.8 | 1.153/1.153 | 0.539/0.539 |

With 16 games, a 24-concurrency cell can only keep 16 games in flight, so
24 was confirmed at 32 games.

Timeout ring (16 games per cell, one process per cell):

| conc | cap | to us | sims | games | wall s | ev/s | pos/s | trainable pos/s | games/h | inflight | batch mean/p50/p95/max | wait p50/p95 us | fwd us | err | VRAM | util busy/max | temp | W/D/L | mate/stale/insuf/3fold/50/trunc | mean plies | ent all/train | top1 all/train |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 8 | 500 | 16 | 16/16 | 199.5 | 199.8 | 12.66 | 12.66 | 288.7 | 8 | 6.39/8/8/8 | 172/7692 | 29901 | 0 | 578 | 89.3/100 | 70 | 2/13/1 | 3/2/9/1/1/0 | 157.8 | 1.153/1.153 | 0.539/0.539 |
| 8 | 8 | 500 | 16 | 16/16 | 94.4 | 422.1 | 26.73 | 26.73 | 609.8 | 8 | 6.39/8/8/8 | 134/9117 | 12922 | 0 | 289 | 67.6/89 | 68 | 2/13/1 | 3/2/9/1/1/0 | 157.8 | 1.153/1.153 | 0.539/0.539 |
| 8 | 8 | 2000 | 16 | 16/16 | 105.2 | 379.0 | 24.00 | 24.00 | 547.5 | 8 | 6.39/8/8/8 | 135/10628 | 13427 | 0 | 289 | 65.7/89 | 69 | 2/13/1 | 3/2/9/1/1/0 | 157.8 | 1.153/1.153 | 0.539/0.539 |
| 16 | 16 | 500 | 16 | 16/16 | 94.2 | 423.0 | 26.79 | 26.79 | 611.2 | 16 | 10.27/12/16/16 | 4271/10517 | 19247 | 0 | 353 | 67.5/95 | 67 | 2/13/1 | 3/2/9/1/1/0 | 157.8 | 1.153/1.153 | 0.539/0.539 |
| 16 | 16 | 2000 | 16 | 16/16 | 96.4 | 413.5 | 26.18 | 26.18 | 597.3 | 16 | 10.29/12/16/16 | 4792/10625 | 19548 | 0 | 353 | 67.8/95 | 69 | 2/13/1 | 3/2/9/1/1/0 | 157.8 | 1.153/1.153 | 0.539/0.539 |

The first conc-8/500 µs cell (199.8 ev/s) ran at *higher* GPU utilization
(89% mean) with forward latency 29.9 ms, against about 13 ms elsewhere. It
looks like external contention, not the timeout. A rerun of the identical
cell gave 422.1 ev/s. The directory was later found renamed
`…-INVALID-contention`, a rename not made by this session. The cell is
excluded. Timeouts of 500 and 2000 µs are within 2.5% of 1000 µs at
concurrency 16.

S1B confirmation (32 games per cell, one process per cell):

| conc | cap | to us | sims | games | wall s | ev/s | pos/s | trainable pos/s | games/h | inflight | batch mean/p50/p95/max | wait p50/p95 us | fwd us | err | VRAM | util busy/max | temp | W/D/L | mate/stale/insuf/3fold/50/trunc | mean plies | ent all/train | top1 all/train |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 8 | 1000 | 16 | 32/32 | 181.0 | 448.2 | 28.38 | 28.38 | 636.6 | 8 | 6.71/8/8/8 | 118/9543 | 13163 | 0 | 289 | 72.6/89 | 70 | 5/25/2 | 7/5/15/1/4/0 | 160.5 | 1.156/1.156 | 0.537/0.537 |
| 16 | 16 | 1000 | 16 | 32/32 | 168.8 | 480.6 | 30.43 | 30.43 | 682.6 | 16 | 11.32/14/16/16 | 1641/10104 | 20544 | 0 | 353 | 77.1/95 | 71 | 5/25/2 | 7/5/15/1/4/0 | 160.5 | 1.156/1.156 | 0.537/0.537 |
| 24 | 16 | 1000 | 16 | 32/32 | 159.8 | 507.7 | 32.14 | 32.14 | 721.0 | 24 | 11.80/16/16/16 | 26735/27572 | 21164 | 0 | 353 | 80.6/95 | 71 | 5/25/2 | 7/5/15/1/4/0 | 160.5 | 1.156/1.156 | 0.537/0.537 |

**Invalid interrupted attempts.** Two conc-8/32-game attempts were deliberately
stopped after a second controller launched an overlapping GPU process. Neither
produced a cell result or JSON and neither is evidence. A later exclusive run,
with exit code and stderr captured, completed with exit 0 and is the row above.

**Decision.** By eval/s at 32 games, 24/16 (507.7) beats 16/16 (480.6) by
5.6%, below the pre-set 15% bar for exceeding 16. Conc 24 also oversubscribes
the 12 logical CPUs and pushes queue wait to 26.7 ms p50. 16/16 beats 8/8
(448.2) by 7%. Ranking vs S1A: 8, 16, and 24 were within 3% at 16 games,
where tail effects dominate; at 32 games the order is 24 > 16 > 8. The coarse
leaders reproduce. Errors 0 everywhere; VRAM ≤ 769 MiB; max temperature
71 °C; peak in-flight = configured concurrency.

**Frozen HP profile** (`configs/hardware/hp-home.toml`): `concurrent_games =
16`, `cpu_workers = 16`, `max_inference_batch = 16` (= observed p95),
`batch_timeout_us = 1000`, training physical batch 32 × accumulation 4 =
effective 128. A current-head CUDA qualification run completed finite training
at physical batches 16/32/64: 132.3/195.3/225.4 examples/s. It did not include
a new VRAM sampler, so the memory basis remains the prior measured F15 peak of
1,953 MiB at batch 64. Batch 32 is frozen for margin and continuity with the
effective-batch contract; batch 64 throughput alone does not override it.

## S2 selection rule (pre-registered before any S2 cell ran)

Budgets 8, 16, 32, 64 sims/move on the frozen reference, the frozen S1
schedule, standard start, and the same game seeds, with 16 games each. 128
runs only if 64 is not more degenerate than lower budgets and its runtime is
practical. No 256.

- **Degenerate** if any of: mean top-1 visit share > 0.9;
  (threefold + fifty-move) > 0.8 of games; trainable-position target entropy
  < 0.1; truncated > 0.5 of games.
- **Candidates** are 8/16/32/64/128. Also excluded: any
  budget where 8 games would take more than 12 minutes to collect at the
  measured rate.
- **Pick** the highest trainable positions/s among non-degenerate candidates.
  **Override** to a higher budget only if its (threefold + fifty-move +
  truncated) share is at least 0.15 lower (absolute) *and* its trainable
  positions/s is at least 0.5× the leader's.
- If every candidate is degenerate: search gate CONDITIONAL/NO-GO, and stop
  before learning. No anti-draw, contempt, or material changes.
- Prior-vs-posterior divergence: NOT RUN (priors are not stored in replay).

## S2 — search-budget qualification (MEASURED, gate: GO)

Final curve built from commit `12f761e`, frozen reference `4271e19f…`, F15
CUDA FP32, standard start, seed 1, frozen S1 schedule 16/16/1000 us, 16 games
per budget. Every cell completed with zero inference errors, 353 MiB sampled
VRAM, and temperature at or below 70 C. Per-cell hashes below name the actual
search and scheduling overrides; this corrects the earlier base-config-only
sweep hash.

| sims | games | wall s | train pos/s | eval/s | W/D/L | mate/stale/insuf/3fold/50/trunc | mean plies | entropy all/train | top1 all/train | scientific hash | resolved hash |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 8 | 16/16 | 51.16 | 52.44 | 415.3 | 3/13/0 | 3/1/8/3/1/0 | 167.7 | 0.931/0.931 | 0.553/0.553 | `b90a03be…` | `60aa29f3…` |
| 16 | 16/16 | 94.11 | 26.83 | 423.7 | 2/13/1 | 3/2/9/1/1/0 | 157.8 | 1.153/1.153 | 0.539/0.539 | `2b8a3902…` | `eeb7b353…` |
| 32 | 16/16 | 206.89 | 12.47 | 393.3 | 4/10/2 | 6/0/7/2/1/0 | 161.3 | 1.276/1.276 | 0.518/0.518 | `f3a6b728…` | `da34f7e6…` |
| 64 | 16/16 | 429.92 | 6.34 | 399.2 | 4/11/1 | 5/0/6/0/5/0 | 170.3 | 1.375/1.375 | 0.492/0.492 | `843071d9…` | `9f2595dc…` |

No budget was degenerate by the registered thresholds. Eight simulations had
the best useful throughput. Relative to 8, no deeper budget reduced the
combined threefold/fifty-move/truncation share by the required 0.15 while
retaining at least half the leader's throughput. At 64, fifty-move draws rose
to 5/16. The frozen budget is therefore **8 simulations/move**. The 64-sim
result does not justify 128, so 128 and 256 are NOT RUN.

The smoke workload formula is
`clamp(round(600 s * 52.44 trainable positions/s / 167.7 mean plies), 8, 24)`,
which freezes `games_per_cycle = 24`.

## Next measured gates

1. Run the bounded two-cycle F15 smoke from `configs/hp/f15-smoke.toml`.
2. Apply the smoke gate without extending or retuning the run.
3. If and only if the smoke passes, run the bounded qualification pilot with
   the identical scientific hash.
4. Issue exactly one evidence-backed R15 GO / CONDITIONAL GO / NO-GO decision.

No F15 training, qualification pilot, or R15 training has run in H1 yet.
