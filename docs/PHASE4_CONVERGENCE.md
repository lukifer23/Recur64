# Recur64 — Phase 4: Mainline Harness Convergence

Status: **P4.0 (integration) and P4.1 (CPU correctness gate) complete.**
P4.2+ (frozen reference, scheduling, search budget, smoke, qualification) are
**NOT RUN**.

This branch (`main-integration`) transplants the proven generic harness
improvements from the experimental branch `experiment/hp-r15`
(`3430e6ca23be18ced66d5d16b9d4663eff50850c`) onto `main` (`5ac291c`), keeping
the mainline F10/R10 research lineage. It is not a merge and not a
fast-forward: the HP branch remains a separate, independently reproducible
experimental lineage.

## Why not fast-forward

`main` is a strict ancestor of `experiment/hp-r15`, so a fast-forward is
technically possible. It is rejected: it would import HP-only configs, docs,
scripts, the F15/R15 model family, the RTX 2050 schedule, and the 8-sim search
budget as mainline content. The HP material is evidence, not a mainline default.

## Ported (generic)

One authoritative `collect_parallel`; `games_per_cycle` / `concurrent_games` /
`cpu_workers` collection semantics; global-game-id self-play seed policy
(`SELFPLAY_SEED_POLICY`); explicit input errors; `SelfPlayMetrics` and
termination/WDL/truncation data health; `TargetHealth` (all vs trainable plies);
example-weighted mean gradient reduction and weighted metrics; replay accounting
plus `current_cycle_sample_fraction` / `mean_sample_age_cycles`; optimizer
continuation through `load_training` with accepted-trajectory promotion;
conservative-v2 promotion with hold reasons; candidate-vs-parent and
candidate-vs-reference arenas; raw policy vs parent; frozen reference
checkpoints (`freeze-reference`, `identity.json`, T0 baseline); scientific vs
resolved config identity; build-time git provenance; NVRTC fail-fast; precision
gate on every run path; concurrency gauge (`peak_in_flight`); GPU
VRAM/utilization/temperature sampling; strict opening-suite validation and
content digest; schedule overrides; learner deadline checks; partial per-cycle
reports.

## Deliberately NOT ported (HP-only / evidence)

`configs/hp/*`, `configs/hardware/hp-home.toml`, `configs/f15.toml`,
`configs/r15.toml`, `crates/recur64-model/tests/f15_r15_parity.rs`,
`sweep::hp_grid`, `scripts/hp-*`, `docs/HP_*`, `Grok-plan.md`; the frozen RTX
2050 schedule (16/16/1000, physical 32 x 4); the 8-simulation search budget; the
R15 NO-GO decision. None of these are mainline facts.

## Mainline adaptations

- `collection_shape()` preserves D24 for legacy configs (no explicit pair):
  `active_games` is both the total and the concurrency, and `cpu_workers` is not
  applied. Historical `configs/f10-*.toml` behavior is unchanged.
- `hp_grid` / `--grid hp` replaced by `workstation_grid` / `--grid workstation`
  (main-workstation candidates, to be frozen by the P4.3 sweep).
- NVRTC guard generalized to match any `nvrtc*.dll` / `libnvrtc.so*` name so a
  valid user-space runtime is never refused over an exact-name mismatch.
- HP-specific config tests removed; mainline tests added
  (`f10_r10_parity.rs`, legacy collection semantics, `f10-reference.toml`
  parse).
- `model_info` control-block baseline and profile-label examples generalized.

## Historical evidence preserved

The Phase 3 F10 result (`docs/F10_BASELINE.md`, `docs/STATUS.md` Phase 3,
D23/D24/D28) is kept verbatim. It was produced before the seed and gradient
fixes and is **not** a clean modern baseline. `configs/f10-*.toml` are kept
unchanged; corrected Phase 4 configs live under `configs/phase4/`.

The HP measured record (`docs/HP_H1_RESULTS.md` on `experiment/hp-r15`) is
external evidence for F15/R15 on the RTX 2050. It is not represented here as a
mainline result.

## P4.1 gate evidence (main workstation, CPU)

- `cargo fmt --all -- --check`: clean.
- `cargo clippy --workspace --all-targets`: clean.
- `cargo test --workspace`: all tests pass (one pre-existing ignored deep perft).
- `cargo check -p recur64-cli --features cuda` with the user-space CUDA 12.9.1
  environment: clean (type-check only; no GPU workload run).
- `f10_r10_parity`: F10 and R10 both report 9,805,288 unique parameters; R10
  executes 2+4R+2 = 8/12/20 blocks at R=1/2/4.

## Next (not run here)

P4.2 freeze the F10 reference (`recur64 freeze-reference`, CUDA), P4.3
workstation scheduling sweep, P4.4 F10 search-budget requalification, P4.5 F10
smoke, P4.6 F10 bounded qualification, P4.7 R10 entry decision. These are
hardware runs and require explicit owner approval. No 24h run is authorized.
