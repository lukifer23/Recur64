# Recur64 — HP experimental branch (`experiment/hp-r15`)

This document describes a **separate experimental lineage** of Recur64 that runs
on a home HP machine. It is not the primary workstation branch and must never be
merged into `main` without a separate architecture decision.

## Branch ancestor

| Item | Value |
|---|---|
| Canonical ancestor | `78be2052612236547f6b5232b417175c2ccdcfc9` — "Phase 2: first complete vertical slice (search, runtime, eval, coordinator)" |
| Ancestor tag | `hp-phase2-base` |
| Branch | `experiment/hp-r15` |
| Depends on Phase 3? | **No.** Independently reproducible from `78be205`. |

The branch is deliberately more aggressive than the workstation's conservative
progression: it targets a larger control model (F15, ~15M) and reaches the
recurrence experiment sooner, while keeping enough controls to make the result
meaningful.

Intended progression:

```
HP HARDWARE PROOF
  → F15 SYSTEMS PROOF
  → F15 REAL-SEARCH LEARNING SMOKE
  → SHORT F15 CONTROL RUN
  → MATCHED-PARAMETER R15
  → R=1/2/4 RECURRENCE
  → INTERNAL COMPUTE vs EXTERNAL SEARCH COMPARISON
```

## Hardware (DETECTED, read-only)

Collected with read-only inspection only. No serial numbers, product IDs, UUIDs,
or secrets are recorded.

| Item | Value |
|---|---|
| Machine | HP desktop (home) |
| OS | Windows 11 Home, build 26200 |
| Architecture | x86_64 |
| CPU | AMD Ryzen 5 7535HS with Radeon Graphics |
| CPU topology | 6 physical cores / 12 logical threads, max 3.301 GHz |
| Host RAM | 31.21 GB visible (2 × 16 GB SODIMM, 5600 MT/s rated / 4800 configured) |
| Free RAM at idle | ~17 GB |
| Storage (C:) | ~134 GB free |
| Storage (D:) | ~70 GB free |
| Discrete GPU | **NVIDIA GeForce RTX 2050**, 4,096 MiB (4 GB) |
| Compute capability | **8.6** (Ampere) |
| GPU driver | 616.92 (Windows); driver-reported CUDA UMD 13.4 |
| GPU power cap | not reported by `nvidia-smi` (`[N/A]`; laptop-class GPU) |
| Integrated GPU | AMD Radeon Graphics — **must not be used for compute** |
| Other GPU load | LM Studio was running at discovery time (contends for VRAM/compute) |

> **Note:** the 4 GB VRAM is the tightest constraint on this machine. It bounds
> `max_inference_batch`, the training physical batch, and R=4 feasibility. This
> is measured, not assumed; see the benchmark and concurrency sections below.

## Environment / toolchain

| Item | Status |
|---|---|
| Rust | 1.97.1 pinned by `rust-toolchain.toml` — **present** |
| Cargo | 1.97.1 — present |
| Git | 2.55.0.windows.3 |
| Linker | VS 2022 Build Tools with the C++ x86/x64 workload — **present** (no custom `~/.cargo/config.toml`) |
| PowerShell | 5.1.26100.9444 |
| CUDA toolkit | **absent** (`CUDA_PATH` empty, no `nvcc`, no user-space redist) |
| CUDA user-space redist | not yet installed |

Toolchain decision for this machine: use the standard **MSVC** linker (VS Build
Tools is installed), rather than replicating the workstation's `rust-lld` +
`xwin-splat` setup. The framework version (Burn 0.21.0) and Rust pin are
unchanged. CUDA runtime setup is tracked separately below.

## F15 / R15 model family

Same fundamental Recur64 architecture as Phase 0/1/2: 64 square tokens,
Observation V1 `[64,119]`, bidirectional transformer, pre-RMSNorm, GeLU FFN,
relative positional bias, sparse legal-candidate policy, WDL head, no CNN/ResNet
trunk.

### F15 — feed-forward control

```toml
width = 512, heads = 8 (head_dim 64), ffn = 768
input_blocks = 0, core_blocks = 8, output_blocks = 0   # 8 unique blocks
```

### R15 — matched-parameter shared recurrent core

```toml
width = 512, heads = 8 (head_dim 64), ffn = 768
input_blocks = 2, core_blocks = 4, output_blocks = 2   # 8 unique blocks
```

Both store exactly eight unique transformer blocks of identical geometry, so
their unique parameter counts are **identical by construction**.

| Quantity | F15 | R15 |
|---|---:|---:|
| Unique parameters | 15,154,120 | 15,154,120 |
| Unique blocks | 8 | 8 |
| Final blocks R=1 | 8 | 8 |
| Final blocks R=2 | — | 12 |
| Final blocks R=4 | — | 20 |
| Deep-sup blocks R=1 | 8 | 8 |
| Deep-sup blocks R=2 | — | 14 |
| Deep-sup blocks R=4 | — | 26 |

Matched on: unique parameter count, input representation, policy head, WDL head,
optimizer family, replay/data source, evaluation protocol. The recurrence loop is
the only intended difference.

Configs: `configs/f15.toml`, `configs/r15.toml`.
Contract test: `crates/recur64-model/tests/f15_r15_parity.rs`.

## Divergences from the primary branch

- **Model size:** F15/R15 (~15.15M) instead of F10/R10 (~9.8M).
- **Search budget:** real PUCT (target ~128 simulations/move) from the first
  learning result, not toy 8/16 budgets.
- **Recurrence entry:** after a *proven-healthy short* F15 control, not a polished
  24h baseline.
- **Toolchain:** standard MSVC linker instead of `rust-lld` + `xwin-splat`.
- **Hardware:** RTX 2050 (4 GB, Ampere 8.6) instead of RTX 2000 Ada (16 GB).

Unchanged: chess contracts, PUCT formula, replay policy, optimizer family,
evaluation protocol, FP32 precision for the first learning result.

## Deferred on this branch

Gumbel, diffusion, geometric attention, SSRL, 30M+ models, distributed self-play,
cloud training, engine-labelled data. BF16 is a later isolated performance
experiment; the first control result stays on FP32.

## Infrastructure changes vs ancestor

The branch closes several gaps found in the Phase 2 ancestor before trusting any
throughput measurement. Full detail and evidence: **`docs/HP_CHANGES.md`**.

- **Real concurrency:** `selfplay`/`collect_only` now use the same parallel
  collect path as `run` (it was sequential at the ancestor). A configured
  `active_games`/`cpu_workers` now always means real concurrent execution.
- **Concurrency gauge:** `MetricsSnapshot::peak_in_flight` counts simultaneous
  evaluator calls — the direct evidence a sweep actually ran concurrently.
- **Run provenance:** `build.rs` bakes the git SHA/branch into every run's
  `metadata.json` (was always `None`); the seed and hardware/model profile labels
  are recorded too.
- **Precision gate:** `ensure_supported` now runs on `run`/`selfplay`/`train`/
  `arena`, not only `bench`.

## Observed on this machine (TESTED)

- F15 and R15 both report **15,154,120** unique parameters via `recur64
  model-info`.
- `recur64 selfplay` (Micro, CPU, 8 games / 4 workers): `peak_in_flight = 4`,
  batch p50 = 3, 0 inference errors — concurrency is real, not configured-only.
- `recur64 run` (Micro, CPU smoke): COLLECT → AUDIT → TRAIN → EVALUATE → REPORT
  completed; `metadata.json` recorded `git_branch: experiment/hp-r15`.
- CPU Flex forward is ~25 ms per Micro batch, so the default 500 µs batch
  timeout flushes small batches on CPU. GPU batching must be measured, not
  assumed (H0.7).

## Status log

Updated as work lands. DETECTED ≠ TESTED.

- **H0.1** — branch created at `78be205`; tag `hp-phase2-base`. DONE.
- **H0.2** — read-only hardware discovery. DONE (table above).
- **H0.4** — F15/R15 configs + exact parameter counts. DONE (verified by
  `model-info` and the parity test).
- **Infra C3/C4/C5** — provenance, precision gate, real concurrency + gauge.
  DONE (verified).
- CUDA runtime/toolchain, F15 CPU/GPU correctness, benchmarks, concurrency sweep,
  search-budget comparison, learning smoke, control run, R15: **NOT YET RUN**.
