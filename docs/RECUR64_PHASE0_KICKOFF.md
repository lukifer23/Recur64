# Recur64 — initial agent assignment

Give the implementation agent this file, `AGENTS.md`, and `docs/RECUR64_RESEARCH_AND_BUILD_PLAN.md`. Execute Phase 0 first. The master plan describes later work; it does not authorize implementing everything immediately.

> **Naming note (2026-09-23):** These specifications were originally drafted under the working name "LoopZero". That name is obsolete; the project is **Recur64** and all implementation artifacts use Recur64 naming. This rename changed labels only — the technical requirements are unchanged.

---

You are the initial implementation/integration agent for **Recur64**, a new Rust-first chess learning experiment. Read the attached master specification and agent rules in full. Build a real system, but execute **only Phase 0** in this assignment.

## Project goal

Our eventual model is a roughly 10M-parameter square-token transformer trained from random initialization on its own search and self-play. It will have a feed-forward baseline and a shared transformer core that can execute 1, 2, or 4 refinement passes. We want to determine whether recurrent internal computation improves chess more than spending the same time on additional tree search. A later discrete-diffusion future-trajectory branch is optional and is not part of this task.

The primary application, model/training integration, search, learner, and storage must be Rust. No CNN/ResNet trunk, pretrained model, teacher-labelled main training corpus, Docker, or Python trainer. Established Rust tensor and legal-move libraries are permitted. The exact GPU is unknown: the screenshot shows 16 GB under graphics and multiple GPUs, but that does not establish the accelerator name or dedicated VRAM.

## Your assignment: prove the hardware/backend before building the engine

### A. Inspect the environment safely

Establish the working directory and inspect existing files before creating or changing anything. Do not overwrite another project. Read the environment/toolchain and identify the actual GPU, driver, dedicated memory, OS, CPU, available RAM, and WSL status where accessible. Use appropriate platform commands; do not attempt Windows commands blindly inside Linux or vice versa.

Read-only examples for the Windows host:

```powershell
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion
wsl --status
wsl --list --verbose
```

If running in a remote agent sandbox rather than the target workstation, state that immediately. You may build CPU correctness tests there, but those results do not certify the user's GPU. Produce a reproducible local probe/benchmark procedure and distinguish pending hardware measurements. Never invent device identity or performance.

Do not install/change drivers, enable WSL, reboot, modify security settings, request paid compute, or upload data without approval. In an already approved WSL CUDA environment, use the Windows host driver and current NVIDIA guidance; do not install a Linux display driver.

### B. Create a minimal pinned Rust workspace

Prefer Burn and its suitable accelerator backend, contingent on the actual hardware. Read the documentation of the exact release you select. Commit `Cargo.lock` and a toolchain file. The current repository notes deprecation of Burn's LibTorch backend as of 0.22.0; do not assume older examples still describe the chosen API.

Implement only the crates/modules needed for the doctor, model probe, and tests. Do not create empty fake modules for every future subsystem. Keep the proposed observation/action/model contracts in documentation, and write an architecture decision naming the selected backend and any unresolved support questions.

A direct `tch-rs` fallback preserves a Rust application but changes the native runtime; propose it with evidence only if Burn proves unsuitable. Do not silently switch to Python or begin a custom autodiff framework.

### C. Implement the real model-shaped test graph

Use the master plan's square-token input shape and configurable transformer dimensions. Implement a Micro configuration first, then allow F10-sized instantiation and parameter reporting. The graph must contain the real operations needed by the planned model: bidirectional attention, pre-normalization, GeLU FFNs, residual paths, relative positional bias, joint legal-candidate policy scores including a promotion path, WDL prediction, cross-entropy losses, actual backpropagation, and AdamW updates.

Exercise a shared core at R=1,2,4. Shared parameters must actually be reused, and repeated uses must contribute gradients. Keep a deterministic R=1 parity test. Use full-gradient recurrence in this proof rather than quietly detaching state. Include masked candidate padding and a terminal-case bypass in the model API tests.

For this systems proof, deterministic synthetic tensors and legal-candidate-shaped fixtures are allowed and must be clearly labelled. They are not real chess self-play or evidence of chess strength. Do not implement a fake inference/training path just to make the benchmark pass.

Print unique parameter counts by module and executed block counts separately. Approximate F10 target: width 384, 12 heads, FFN 768, eight unique transformer blocks, with compact policy/WDL heads. Explain material parameter-count deviations.

### D. Verify correctness before speed

Required tests:

- CPU FP32 forward/backward checks, including small numerical gradient checks where practical.
- All losses and gradients finite; real optimizer updates alter parameters.
- A fixed small fixture can be overfit substantially; report its actual loss curve, not a hardcoded assertion that it learned.
- Masked/padded action candidates receive no probability, legal candidates normalize jointly, and promotion branches participate in gradients.
- No all-masked softmax for terminal positions.
- R=1 parity and shared-parameter/gradient checks.
- Batch versus single-item inference agreement within stated tolerance.
- Checkpoint reload preserves inference.
- Training checkpoint restore preserves optimizer moments, learning-rate step, and RNG/sampler state; resumed-versus-uninterrupted continuation agrees under a stated CPU/GPU tolerance.
- Unsupported precision or device settings produce an explicit error or an explicitly selected alternative, not an invisible fallback.

BF16 is an optional optimization after FP32 passes. Test the full forward/backward/optimizer path, maintain appropriate FP32 state/reductions, and demonstrate numerical agreement. Device BF16 capability by itself is not proof of backend support. Use smaller FP32 batches if necessary.

### E. Benchmark the actual target workload

Use short bounded runs, preferably under 30 minutes total after compilation unless a necessary build requires longer. Do not start self-play or a long training run. Synchronize GPU work for timings. Separate first-use compilation from warmed steady-state execution.

Measure, where feasible:

- Inference at batch sizes 1,16,64,128.
- Training at physical batch sizes 32,64,128.
- R=1,2,4.
- FP32 and only verified mixed precision.
- Peak GPU/host memory, actual selected adapter, parameter count, transfer-inclusive and compute-only timing where both can be measured accurately.
- Checkpoint save/load time and file size.

Record out-of-memory and unsupported combinations; do not hide them. Do not extrapolate a rating or full-game generation speed from this microbenchmark. Until self-play exists, end-to-end chess throughput is unmeasured.

### F. Produce the Phase 0 decision package

Deliver actual source, tests, configuration, lockfile, and these truthful documents:

- `docs/HARDWARE.md`: observed hardware and what remains unknown.
- `docs/ARCHITECTURE.md`: selected backend and initial model graph.
- `docs/BENCHMARKS.md`: exact commands, measurements, warm/cold distinction, precision, limitations, and raw-log locations.
- `docs/DECISIONS.md`: why the backend is accepted, rejected, or pending target-hardware verification.
- `docs/STATUS.md`: completed, verified, failed, and not-run work.
- A minimal README with copy-paste build/test/benchmark commands for the actual environment.

Run formatting, compile/check, unit tests, and linting appropriate to the selected workspace. List every command actually executed and any failure or unavailable device test. Do not claim that the engine, self-play, Gumbel search, UCI, or chess learning exists yet.

## Agent collaboration

The primary agent owns architecture, contracts, and integration. A second agent may independently review the backend graph, masking, recurrence gradients, and checkpoint restoration, or implement a narrowly assigned benchmark/test module. Assign non-overlapping files and use a branch/worktree. Do not run parallel rewrites of the same model or schema.

## Stop condition

Stop after Phase 0 with one of:

1. Backend validated on the target accelerator; propose the bounded Phase 1 chess-contract assignment.
2. CPU implementation validated but target GPU evidence unavailable; provide the exact local command/probe needed.
3. A genuine backend blocker with logs and a narrowly scoped fallback recommendation.

Do not proceed automatically into the whole engine, change the architecture to avoid a failed test, or launch a multi-day training run. The next phase follows review of actual evidence.
