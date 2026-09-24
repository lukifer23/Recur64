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

## S2 selection rule (pre-registered before any S2 cell ran)

Budgets 8, 16, 32, 64 sims/move on the frozen reference, the frozen S1
schedule, standard start, and the same game seeds, with 16 games each. 128
runs only if 64 is not more degenerate than lower budgets and its runtime is
practical. No 256.

- **Degenerate** if any of: mean top-1 visit share > 0.9;
  (threefold + fifty-move) > 0.8 of games; trainable-position target entropy
  < 0.1; truncated > 0.5 of games.
- **Candidates** are 16/32/64/128; 8 is a curve point only. Also excluded: any
  budget where 8 games would take more than 12 minutes to collect at the
  measured rate.
- **Pick** the highest trainable positions/s among non-degenerate candidates.
  **Override** to a higher budget only if its (threefold + fifty-move +
  truncated) share is at least 0.15 lower (absolute) *and* its trainable
  positions/s is at least 0.5× the leader's.
- If every candidate is degenerate: search gate CONDITIONAL/NO-GO, and stop
  before learning. No anti-draw, contempt, or material changes.
- Prior-vs-posterior divergence: NOT RUN (priors are not stored in replay).

## Next measured gates

1. Verify the frozen-reference command on CUDA and record model ID, git SHA,
   scientific/resolved hashes, and seed.
2. Rerun the scheduling comparisons with that same checkpoint, then freeze
   concurrency, batch cap, timeout, and physical/effective training batch.
3. Compare 32/64/128 simulations with the frozen schedule and checkpoint.
4. Run a bounded real F15 smoke only if search data passes its gate; then run
   the qualification pilot only if the smoke passes.
5. Issue exactly one evidence-backed R15 GO / CONDITIONAL GO / NO-GO decision.

No F15 training, qualification pilot, or R15 training has run in H1 yet.
