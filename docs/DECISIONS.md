# Recur64 — Decisions (Phase 0)

Architecture decision records. Status values: **ACCEPTED**, **PENDING**,
**REJECTED**, **DEFERRED**.

## D1 — Burn 0.21.0 stable as the framework

- **Status:** ACCEPTED
- **Decision:** Pin `burn = "=0.21.0"` (and `burn-flex`, `burn-cuda` when
  enabled). Commit `Cargo.lock` and `rust-toolchain.toml` (Rust 1.97.1).
- **Why:** 0.21.0 is the latest stable release with a stable `burn-cuda`; the
  0.22 line is prerelease. Stable pins are more reproducible.
- **Revisit if:** the graph cannot run or performs poorly and a specific 0.22
  prerelease fix is demonstrably required. Re-evaluation is a separate decision.

## D2 — Native Windows first; WSL2 as fallback

- **Status:** ACCEPTED
- **Decision:** Evaluate native Windows + Burn/CUDA first (tree A). Fall back to
  a fresh Ubuntu under WSL2 (tree B) only if native is blocked or materially
  inferior. Do not maintain both.
- **Why:** Rust builds work natively here (validated), avoiding a second
  toolchain. WSL2 remains available but its only current distro is an unfamiliar
  `BendExp`; a fresh Ubuntu would be used.
- **Evidence so far:** native Windows CPU FP32 graph passes all correctness gates.

## D3 — CUDA runtime via user-space redistributables (no admin)

- **Status:** PENDING (owner-approved direction; not yet installed)
- **Decision:** Because there are no administrator rights, do not use the CUDA
  Windows installer. Instead extract the CUDA 12.x redistributable component
  archives (`cuda_nvrtc`, `libcublas`, `cuda_cudart`; `cuda_nvcc` only if
  required by `cudarc`'s build script) into a user directory such as
  `%LOCALAPPDATA%\Recur64\cuda\12.x`, and set `CUDA_PATH`, `PATH`, and
  `CUDA_VERSION` **for the `recur64` process only**. No PATH/registry/system
  changes.
- **Why:** `burn-cuda` requires CUDA 12.x on `PATH`; the display driver is
  already present, so only user-space runtime libraries are missing.
- **Open risk:** `cudarc`/CubeCL must find the extracted `nvrtc`/`cublas`; if
  they cannot, fall back to WSL2 (D2/B).
- **Not yet done:** no CUDA component has been downloaded or extracted.

## D4 — CPU FP32 correctness baseline via Flex

- **Status:** ACCEPTED
- **Decision:** All correctness work runs first on `Autodiff<Flex>` (pure-Rust
  CPU). GPU evidence is added only after the CPU graph passes.
- **Why:** No installs required; deterministic; isolates model/contract bugs from
  backend/GPU bugs.

## D5 — Minimal two-crate workspace

- **Status:** ACCEPTED
- **Decision:** Only `recur64-model` and `recur64-cli`. `recur64-core`,
  `recur64-search`, `recur64-runtime`, `recur64-eval` are deferred to the phases
  that need them.
- **Why:** Avoids fake scaffolding for hypothetical future needs.

## D6 — Eager parameter initialization

- **Status:** ACCEPTED
- **Decision:** `ProbeModel::new` force-initializes all parameters.
- **Why:** Burn 0.21 lazily initializes parameters; cloning an uninitialized
  module copies the deferred initializer and re-samples on first access. This
  silently broke value-preserving clones and made a resume test diverge. With
  eager init, CPU FP32 resume is bit-exact.
- **Consequence:** any future module clone in a resume path must ensure
  parameters are materialized first.

## D7 — No Python trainer, no custom autodiff, no custom CUDA kernels

- **Status:** ACCEPTED
- **Decision:** Training is Rust. Any deviation requires its own ADR.

## Version pins

| Component | Pin |
|---|---|
| Rust | 1.97.1 (`rust-toolchain.toml`) |
| burn | =0.21.0 |
| cubecl (transitive) | 0.10.0 |
| serde / serde_json / toml / anyhow / clap | caret, locked by `Cargo.lock` |

## Rejected / deferred

- **tch-rs**, **Candle**: deferred fallbacks (see `ARCHITECTURE.md`).
- **0.22.0-pre.x Burn**: deferred until a stable release or a demonstrated need.
- **Docker, Python trainer, custom CUDA kernels**: rejected.
