# HP H1 reboot checkpoint — 2026-09-23

Status: **in progress; no R15 entry decision yet**. This checkpoint is for a
planned Windows restart. Do not infer F15 learning health from the data below.

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

## In progress at the restart

Uncommitted follow-up code adds a seeded `freeze-reference` CLI command and a
`--checkpoint` input to `bench-runtime`, so cells can use identical initial
F15 weights. `cargo check -p recur64-cli` passed for these edits. CUDA feature
check, release build, and execution of the new command have **not run** yet.
The CUDA check was stopped for this reboot; no GPU job remains active.

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
