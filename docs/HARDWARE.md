# Recur64 — Hardware

Observed on the target workstation during Phase 0 planning and implementation.
Only read-only inspection was used. Windows product/device IDs, serial numbers,
and other identifiers are intentionally omitted.

## Observed (DETECTED)

| Item | Value |
|---|---|
| OS | Windows 11 Business, build 26200 (25H2-class) |
| Architecture | x86_64 |
| CPU | Intel Core Ultra 9 285K |
| CPU topology | 24 cores / 24 logical (no SMT) |
| Host RAM | 63.46 GB total, ~36 GB free at idle |
| Storage | C: ~1904 GB total, ~1544 GB free |
| Discrete GPU | NVIDIA RTX 2000 Ada Generation |
| GPU memory | 16,380 MiB total; ~15,750 MiB free at idle |
| Compute capability | 8.9 (Ada Lovelace) |
| NVIDIA driver | 596.71 (Windows); driver-reported max CUDA runtime 13.2 |
| Display model | WDDM (desktop-shared) |
| Integrated GPU | Intel Graphics (must not be used for compute) |
| WSL | WSL 2.5.10, kernel 6.6.87.2-1, one distro `BendExp` (v2, stopped) |
| Rust | rustc 1.97.1 (2026-07-14), cargo 1.97.1, rustup 1.29.0, host `x86_64-pc-windows-msvc` |
| Linker | `rust-lld` + a bundled `xwin-splat` MSVC/CRT/SDK library set via `~/.cargo/config.toml`; **no Visual Studio / Windows Kits install** |
| CUDA toolkit | **user-space redist 12.9.1** at `%LOCALAPPDATA%\Recur64\cuda\12.9.1` (no installer, no admin) |
| Python | present (unrelated venv); **not used as a trainer** |

## What was actually TESTED

| Capability | Status |
|---|---|
| Native Windows Rust build/link (Burn 0.21.0 + Flex) | **TESTED — works** |
| CPU FP32 full probe graph (fwd/bwd/AdamW/checkpoint) | **TESTED — passes** |
| CPU FP32 bit-exact resume | **TESTED — Δloss = 0, Δweight = 0** |
| CPU FP32 bounded benchmark | **TESTED — see BENCHMARKS.md** |
| Native Windows CUDA build/link (Burn 0.21.0 `burn-cuda`, CubeCL/CUDA) | **TESTED — works** |
| CUDA FP32 graph on the RTX 2000 Ada (fwd R=1/2/4, bwd, AdamW, checkpoint) | **TESTED — `recur64 cuda-smoke` PASS** |
| CUDA FP32 synchronized benchmark (R10, batch 1–128, R=1/2/4) | **TESTED — see BENCHMARKS.md** |
| BF16 full graph | **NOT YET TESTED** (hardware capable; backend path not exercised) |

## Remains unknown

- Whether BF16 is numerically stable for the full graph on this GPU.
- Whether WSL2 (fresh Ubuntu) would be materially faster (native Windows works,
  so WSL2 is not currently required).
- Peak VRAM at R=1/2/4 and GPU checkpoint timings under sustained load.
- GPU numerical determinism / bit-exactness (only tolerance-bounded claims hold).
