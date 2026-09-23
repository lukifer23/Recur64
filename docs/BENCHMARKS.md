# Recur64 — Benchmarks (Phase 0)

Systems measurements on the target workstation. Not chess-learning results.
Two backends were measured: **CPU (Burn Flex, FP32)** and **CUDA (Burn 0.21.0,
FP32)** on the RTX 2000 Ada. GPU timings synchronize the device around each
timed region.

## Exact commands

```
cargo run --release -- doctor
cargo run --release -- model-info --config configs/micro.toml
cargo run --release -- model-info --config configs/f10.toml
cargo run --release -- model-info --config configs/r10-probe.toml
cargo run --release -- bench --config configs/micro.toml --output runs/micro-cpu \
    --inference-batches 1,16,64,128 --recurrences 1,2,4 \
    --train-batches 32 --iters 5 --warmup 2 --train-steps 3
```

Raw data: `runs/micro-cpu/bench.json` and `runs/micro-cpu/bench.md` (git-ignored).
`--release` is required for meaningful throughput; debug Flex is roughly an order
of magnitude slower.

## Environment

- OS: Windows 11 build 26200; CPU: Intel Core Ultra 9 285K (24c/24t); 63.46 GB RAM.
- Backend: Burn 0.21.0 `Flex` (pure-Rust CPU), FP32. No accelerator synchronization
  is needed on CPU.
- Model: `micro` — 1,351,840 unique params, 4 unique blocks.
- Cold = first invocation; warm = mean of 5 steady-state iterations after 2 warmups.
- Timing: wall-clock `std::time::Instant`.

## Inference (micro)

| batch | R | executed blocks | cold ms | warm ms | examples/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 4 | 5.5 | 4.7 | 213.0 |
| 1 | 2 | 8 | 9.1 | 9.3 | 107.2 |
| 1 | 4 | 16 | 18.2 | 17.5 | 57.2 |
| 16 | 1 | 4 | 95.4 | 97.0 | 164.9 |
| 16 | 2 | 8 | 179.3 | 183.0 | 87.4 |
| 16 | 4 | 16 | 344.5 | 495.1 | 32.3 |
| 64 | 1 | 4 | 542.7 | 545.1 | 117.4 |
| 64 | 2 | 8 | 1051.0 | 1052.0 | 60.8 |
| 64 | 4 | 16 | 2063.3 | 2069.2 | 30.9 |
| 128 | 1 | 4 | 1084.9 | 1087.2 | 117.7 |
| 128 | 2 | 8 | 2101.8 | 2099.6 | 61.0 |
| 128 | 4 | 16 | 4136.6 | 4131.3 | 31.0 |

Throughput scales roughly inversely with executed blocks (1.0x / 2.0x / 4.0x for
micro's core-only layout), consistent with compute-bound behaviour.

## Training (micro, batch 32)

| R | cold ms | warm ms | examples/s | finite |
|---:|---:|---:|---:|:--:|
| 1 | 809.5 | 542.4 | 59.0 | yes |
| 2 | 1567.9 | 1045.0 | 30.6 | yes |
| 4 | 3076.0 | 2044.6 | 15.7 | yes |

Training step cost scales with executed blocks and the backward pass.

## GPU (CUDA) — R10 probe, 9,805,288 params

Environment: NVIDIA RTX 2000 Ada (16 GB), driver 596.71, user-space CUDA 12.9.1,
Burn 0.21.0 `burn-cuda`, FP32. `--device cuda`; device synchronized.

```
$env:CUDA_PATH = "$env:LOCALAPPDATA\Recur64\cuda\12.9.1"
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
cargo run --release -p recur64-cli --features cuda -- bench \
    --config configs/r10-probe.toml --device cuda --output runs/r10-cuda \
    --inference-batches 1,16,64,128 --recurrences 1,2,4 \
    --train-batches 32,64,128 --iters 20 --warmup 5 --train-steps 3
```

Raw: `runs/r10-cuda/bench.json`, `runs/r10-cuda/bench.md`.

### Inference (R10)

| batch | R | blocks | cold ms | warm ms | examples/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 8 | 2294.63 | 7.50 | 133.3 |
| 1 | 2 | 12 | 9.67 | 9.52 | 105.1 |
| 1 | 4 | 20 | 15.06 | 14.88 | 67.2 |
| 16 | 1 | 8 | 1340.29 | 11.69 | 1368.2 |
| 16 | 2 | 12 | 15.92 | 14.68 | 1090.0 |
| 16 | 4 | 20 | 24.46 | 24.00 | 666.7 |
| 64 | 1 | 8 | 995.62 | 26.11 | 2451.0 |
| 64 | 2 | 12 | 38.28 | 38.66 | 1655.3 |
| 64 | 4 | 20 | 68.02 | 64.36 | 994.4 |
| 128 | 1 | 8 | 515.13 | 62.78 | 2038.9 |
| 128 | 2 | 12 | 94.10 | 95.62 | 1338.7 |
| 128 | 4 | 20 | 155.96 | 159.08 | 804.6 |

### Training (R10, physical batch)

| batch | R | cold ms | warm ms | examples/s |
|---:|---:|---:|---:|---:|
| 32 | 1 | 6944.93 | 77.85 | 411.0 |
| 32 | 2 | 136.66 | 90.54 | 353.4 |
| 32 | 4 | 203.53 | 133.26 | 240.1 |
| 64 | 1 | 1074.35 | 110.10 | 581.3 |
| 64 | 2 | 215.73 | 148.03 | 432.3 |
| 64 | 4 | 344.70 | 225.82 | 283.4 |
| 128 | 1 | 827.57 | 203.78 | 628.1 |
| 128 | 2 | 438.03 | 293.62 | 435.9 |
| 128 | 4 | 714.15 | 474.22 | 269.9 |

Large cold times reflect CubeCL/CUDA kernel autotuning and JIT compilation on
first use; warm numbers are the steady state. Warm throughput falls roughly with
executed blocks, as expected.

## Not run / unsupported

- **F10 / R10 CPU benchmarks:** not run (expected to be far slower than micro);
  their exact parameter counts and executed-block accounting are reported by
  `model-info`.
- **BF16 / FP16:** refused by the precision gate until the full graph is verified.
- **OOM cases:** none observed at batch <= 128 on either backend.
- **Peak VRAM / host RAM / checkpoint timing:** not yet measured under load.
- **GPU determinism:** not claimed; only tolerance-bounded equality holds.

## Limitations

- CPU numbers do not predict GPU throughput.
- Do not extrapolate games/sec, Elo, or training-time forecasts from these
  measurements. No self-play exists yet.
- Warm-up effects and OS scheduling noise are present on a desktop CPU.

---

# Phase 1 — chess core baselines (CPU)

Release-mode, CPU-only rules/encoding throughput. These establish a baseline for
the future self-play path; they are **not** an optimization target.

Command:

```
cargo run --release -p recur64-cli -- bench-core --output runs/bench-core
```

Raw: `runs/bench-core/bench-core.json`, `runs/bench-core/bench-core.md`.
Environment: Intel Core Ultra 9 285K; cozy-chess 0.3.4 default features;
`apply(clone)` = clone + apply one legal move (includes history clone).

| position | legal | movegen/s | apply(clone)/s | encode/s | perft nps |
|---|---:|---:|---:|---:|---:|
| startpos | 20 | 5,844,194 | 8,607,333 | 3,229,453 | 55,086,634 |
| kiwipete | 48 | 2,348,741 | 10,731,917 | 3,013,137 | 51,360,344 |
| endgame_kp | 14 | 9,102,494 | 10,530,750 | 3,036,053 | 40,114,123 |
| promotion | 7 | 15,567,837 | 9,285,051 | 3,601,657 | 40,714,286 |
| castling | 26 | 4,173,187 | 10,273,269 | 3,177,730 | 45,359,736 |

Sizes: `GameState` 88 B, `Board` 96 B, `StandardMove` 4 B (fixed overhead plus
the per-ply history allocation).

Limitations: `apply` includes a history clone; perft uses cozy movegen routed
through Recur64's conversion. Numbers are single-run and not statistically
characterized. No optimization was performed in Phase 1.
