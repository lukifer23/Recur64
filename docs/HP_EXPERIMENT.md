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
unchanged.

### CUDA runtime (user-space, pinned)

The HP has a CUDA-capable GPU (Ampere, compute 8.6) and an NVIDIA driver that
reports a CUDA UMD version newer than 12.x, but no CUDA toolkit and no
`CUDA_PATH`. To keep the CUDA version **byte-identical** to the workstation and
avoid a system-wide install, this branch uses the same pinned **user-space
CUDA 12.9.1 redistributable** approach as `docs/DECISIONS.md` D3:

- Components extracted into `%LOCALAPPDATA%\Recur64\cuda\12.9.1`:
  `cuda_cudart`, `cuda_nvrtc`, `libnvjitlink`, `libcublas` (the runtime
  libraries `burn-cuda`/`cubecl-cuda`/`cudarc` load dynamically).
- `CUDA_PATH` and `PATH` are set **for the `recur64` process only**; no PATH,
  registry, driver, or system change.

A system-wide CUDA Toolkit install was considered (admin is available) and is
**not** used: it would not simplify the runtime and would break the cross-machine
reproducibility goal. This is a deliberate divergence-in-tooling but not in
framework version or numerics.

Status: install + `cuda-smoke` on F15 are **in progress** (see status log).

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

## Upstream reconciliation (Phase 3 merged)

`main` advanced to `5ac291c` ("Phase 3: F10 + PUCT control baseline"). It is
merged into this branch. Phase 3 had already implemented and measured much of
what this branch planned to discover, so the merge replaces planned work with
measured work:

| Topic | Phase 3 result | Effect on this branch |
|---|---|---|
| Concurrency | `active_games` = concurrent games (one thread each); batch mean 1.0 → 20–34 | **Adopted**; `collect_parallel` now uses this model for both `run` and `collect_only` |
| Provenance | `config_hash` + `lineage.jsonl`; `metadata.json` still lacked the git SHA | Combined with this branch's build-time git SHA/branch |
| Batching / search budget | sims 16/32/64 → 47/33.5/17.5 pos/s; sims frozen at 64 | Start from 64 and re-measure on the RTX 2050 |
| Learner | loss split, grad norm, warmup+cosine, gradient accumulation, health guards, bit-exact resume | **Adopted wholesale** |
| Replay | streaming sampler + capacity archiving (bounded memory) | **Adopted wholesale** |
| Evaluation | raw-policy evaluator + frozen opening suite + arena 95% CI | **Adopted**; replaces the planned raw-policy harness |
| Multi-cycle control | `recur64 pilot` bounded controller | **Adopted** as the learning loop |
| F10 pilot outcome | loss unstable, grad norm 44–88, reuse 0.07–0.13 vs target 2.0, raw policy ≤ random, searched play repetition-dominated | **Learning health is the blocker, not recurrence** |

**Updated prior (important).** The Phase 3 F10 baseline shows the *feed-forward*
learner is not yet learning well: raw policy is at or below random, and searched
self-play is repetition-dominated (the untrained value function shuffles into
draws). A recurrence-vs-search comparison is meaningless until the base learner
is healthy. On this branch, R15 entry is therefore gated on the same
learning-health criteria Phase 3 set before its 24h run, not merely on "F15
runs". The HP branch's advantage is a *larger* model and a *different GPU*, not
a licence to skip the health gate.

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

### CUDA on the RTX 2050 (TESTED)

`recur64 cuda-smoke --config configs/f15.toml`: **PASS** — FP32 forward R=1
(8 blocks) finite, backward + AdamW moved the core weight (0.029754 →
0.029455), GPU checkpoint restore exact (delta 0).

F15 (8 blocks, R=1) inference, warm, CUDA FP32:

| batch | warm ms | examples/s |
|---:|---:|---:|
| 1 | 8.50 | 117.7 |
| 8 | 13.57 | 589.5 |
| 16 | 25.04 | 638.9 |
| 32 | 51.03 | 627.1 |
| 64 | 101.27 | 632.0 |

F15 training (autodiff), warm:

| batch | warm ms | examples/s |
|---:|---:|---:|
| 16 | 145.05 | 110.3 |
| 32 | 167.87 | 190.6 |
| 64 | 290.47 | 220.3 |

R15 inference, warm, by recurrence (executed blocks 8 / 12 / 20):

| batch | R=1 ex/s | R=2 ex/s | R=4 ex/s |
|---:|---:|---:|---:|
| 1 | 173.6 | 91.1 | 53.4 |
| 8 | 607.7 | 416.4 | 253.5 |
| 16 | 624.6 | 421.4 | 255.1 |
| 32 | 615.2 | 412.7 | 248.9 |
| 64 | 617.2 | 412.9 | 248.6 |

R15 training, warm:

| batch | R=1 ex/s | R=2 ex/s | R=4 ex/s |
|---:|---:|---:|---:|
| 16 | 166.4 | 148.7 | 95.5 |
| 32 | 249.2 | 179.4 | 118.8 |

**Compute accounting sanity:** inference throughput scales ≈ inversely with
executed blocks (8/12/20 → ~617/413/249 ex/s), i.e. recurrence cost is linear in
blocks, as designed. The F15/R15 comparison can therefore be accounted in
executed blocks.

GPU resource envelope (measured with a 0.5 s sampler during the benchmarks):

| Metric | F15 | R15 |
|---|---:|---:|
| Peak VRAM | 1,953 MiB | 2,113 MiB |
| Peak power draw | 25.8 W | 44.8 W |
| Peak temperature | 62 °C | 63 °C |
| Peak SM clock | 1,710 MHz | 1,710 MHz |
| Peak GPU util | 100% | 100% |

The card has 4,096 MiB; these runs used ~half. Batch 64 inference and batch 32
training (R=4) are safely inside the envelope. Batch 128 training at R=4 is
**not** yet tested and may exceed VRAM; it is not assumed.

**Cold/JIT caveat:** first use of each shape JIT-compiles and autotunes
(0.5–8 s per case). Warm numbers are the steady state; cold numbers must never
be quoted as throughput.

### CUDA runtime reproducibility notes

Two non-obvious fixes were required and are part of the HP setup:

1. `cudarc` 0.19.9 looks for `nvrtc64_12.dll`, but the CUDA 12.9.1 redist ships
   `nvrtc64_120_0.dll`. An alias copy (`nvrtc64_12.dll`) was created in the
   user-space `bin`. Without it, JIT compilation silently produced no-op
   kernels.
2. `nvrtc` needs the compiler-internal headers and libdevice, which live in the
   **`cuda_nvcc`** component (`include/crt/*`, `nvvm/libdevice/libdevice.10.bc`),
   not `cuda_cudart`. The full set is `cuda_cudart`, `cuda_nvrtc`, `cuda_nvcc`,
   `libnvjitlink`, `libcublas`.

## Status log

Updated as work lands. DETECTED ≠ TESTED.

- **H0.1** — branch created at `78be205`; tag `hp-phase2-base`. DONE.
- **H0.2** — read-only hardware discovery. DONE (table above).
- **H0.3** — CUDA runtime setup (user-space 12.9.1: cudart, nvrtc, **nvcc/crt
  headers**, nvjitlink, cublas + `nvrtc64_12.dll` alias). DONE.
- **H0.4** — F15/R15 configs + exact parameter counts. DONE (verified by
  `model-info` and the parity test).
- **H0.5** — F15 CUDA correctness. DONE (`cuda-smoke` PASS).
- **H0.6** — F15/R15 GPU benchmark matrix + VRAM/power/thermal envelope. DONE.
- **Infra C3/C4/C5/C7** — provenance, precision gate, real concurrency + gauge,
  schedule overrides + sweep tooling. DONE (verified).
- **H0.7** — concurrency sweep (pre-merge): real concurrency confirmed; best
  334.6 evals/s at 12 workers, batch p50 = 11. DONE (superseded by Phase 3's
  structural concurrency fix).
- **Merge** — Phase 3 (`5ac291c`) merged and reconciled (commit `fa66c32`);
  build (CUDA) + full test suite + fmt/clippy clean. DONE.
- **H0.8** freeze profile, **H1.x** search budget / pilot / control, **R1.x**
  recurrence: **NOT YET RUN**.

## Revised next steps (post-merge)

Phase 3 supplies the tooling this branch was about to build. The HP progression
is therefore re-based on it:

1. **HP batching sweep** — `recur64 bench-runtime` on the RTX 2050 (replaces the
   hand-rolled `scripts/hp-concurrency-sweep.ps1`), then freeze the HP hardware
   scheduling profile.
2. **Search budget** — start from Phase 3's frozen `sims = 64`; re-measure on
   the RTX 2050 before changing it.
3. **F15 learning smoke / pilot** — `recur64 pilot` with `configs/hp/f15-pilot.toml`.
4. **Raw policy + openings** — `recur64 eval-policy` and `configs/openings-v1.toml`.
5. **R15 entry gate** — requires F15 *learning health*, not just "F15 runs"
   (see the updated prior above).
