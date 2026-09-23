# Recur64 — Agent operating rules

Reconstructed for Recur64 from the master specification and the Phase 0 kickoff.
These rules govern implementation agents.

## Scope discipline

- Execute **only the assigned phase**. The master plan describes later work; it
  does not authorize building it now.
- Do not scaffold empty crates/modules for future phases.
- Do not implement chess rules, search (PUCT/Gumbel), self-play, replay, UCI,
  arenas, diffusion, or long training runs before their phase.
- Do not switch training to Python, write custom autodiff, write custom CUDA
  kernels, or use Docker without a separate explicit architecture decision.

## Evidence discipline

- No fake benchmarks, no placeholder implementations, no passing tests achieved
  by weakening assertions.
- Never claim a device/precision was validated unless the actual graph ran on it.
- Never claim recurrence works until repeated shared parameters contribute
  gradients (verified).
- Never claim resumability from a weights-only file.
- Keep failed and unrun tests visible.
- Distinguish DETECTED from TESTED in all reports.
- No silent fallback: if GPU/CUDA/BF16 is requested but unavailable, error
  visibly rather than substituting CPU/FP32.

## Backend and versioning

- Pin the Rust toolchain and the framework version; commit `Cargo.lock`.
- Read documentation for the exact pinned release; do not copy examples from
  framework `main`.
- One primary backend. Fallbacks require an ADR with logs.

## Repository and naming

- Canonical repository: `https://github.com/lukifer23/Recur64`.
- All new artifacts use the **Recur64** name. The former working name
  "LoopZero" is obsolete.
- Preserve the master specification and Phase 0 kickoff as documentation; do not
  silently change their technical requirements.

## Collaboration

- One agent owns shared contracts. Delegate narrow modules only after interfaces
  exist. Use non-overlapping file ownership.
- A ticket states: objective, prerequisites, files owned, acceptance tests,
  benchmark requirements, prohibited scope expansion, outputs, stop conditions.
- Review actual diffs and machine logs, not persuasive summaries.

## System safety

- Do not install/alter drivers, enable Windows features, modify security
  settings, use Docker, or start paid compute without owner approval.
- No administrator rights are assumed; prefer user-space solutions.
- Do not copy Windows product/device IDs, serials, or secrets into documentation.
