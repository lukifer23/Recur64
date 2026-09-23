# Recur64 — Benchmarks (Phase 0)

All numbers below are **CPU (Burn Flex, FP32)** on the target workstation. They
are systems measurements, not chess-learning results. No GPU benchmark exists
yet because no CUDA runtime is installed (see `DECISIONS.md` D3).

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

## Not run / unsupported

- **F10 / R10 CPU benchmarks:** not run (expected to be far slower than micro);
  their exact parameter counts and executed-block accounting are reported by
  `model-info`.
- **GPU inference/training:** NOT RUN — no CUDA runtime installed.
- **BF16 / FP16:** refused by the precision gate until the full graph is verified.
- **OOM cases:** none observed on CPU.
- **Peak VRAM / host RAM / checkpoint timing:** not yet measured (CPU path).

## Limitations

- CPU numbers do not predict GPU throughput.
- Do not extrapolate games/sec, Elo, or training-time forecasts from these
  measurements. No self-play exists yet.
- Warm-up effects and OS scheduling noise are present on a desktop CPU.
