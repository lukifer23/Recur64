# Recur64 — HP branch change ledger (`experiment/hp-r15`)

A running record of every change this experimental branch makes on top of the
Phase 2 ancestor `78be205`. Kept separate from `HP_EXPERIMENT.md` (the experiment
narrative) so code/infrastructure changes are auditable on their own.

DETECTED ≠ TESTED. Each entry states what was actually verified and how.

---

## C1 — Branch and ancestor

- **Why:** fork the HP experiment from the completed Phase 2 state without
  inheriting the workstation's Phase 3 work.
- **What:** branch `experiment/hp-r15` created at
  `78be2052612236547f6b5232b417175c2ccdcfc9`; annotated tag `hp-phase2-base`.
- **Evidence:** `git rev-parse HEAD` == ancestor SHA; clean tree.

## C2 — F15 / R15 model family and parameter parity (H0.4)

- **Why:** reach a ~15M matched-parameter control/recurrent pair sooner.
- **What:**
  - `configs/f15.toml` — feed-forward control, `512/8/768`, `0+8+0` blocks.
  - `configs/r15.toml` — shared recurrent core, `512/8/768`, `2+4+2` blocks.
  - `crates/recur64-model/tests/f15_r15_parity.rs` — contract tests.
- **Key fact:** both store **15,154,120** unique parameters; R15 executes
  `2+4R+2` = 8/12/20 blocks at R=1/2/4.
- **Evidence:** `recur64 model-info` on both configs prints `TOTAL UNIQUE
  15154120`; the parity test passes (asserts equality, block accounting,
  R=1 topology parity, config parse).

## C3 — Git provenance in run metadata (G4)

- **Why:** the plan requires every run to record its git SHA and branch; Phase 2
  left `git_revision` permanently `None`.
- **What:** `crates/recur64-runtime/build.rs` bakes `RECUR64_GIT_SHA` /
  `RECUR64_GIT_BRANCH` at build time. `RunMetadata` gains `git_branch`, `seed`,
  `hardware_profile`, `model_profile`; `RunConfig` gains the two profile labels
  and typed `device_kind()` / `precision_kind()` helpers.
- **Evidence:** a full `recur64 run` on this machine wrote `metadata.json` with
  `git_revision: 78be205...` and `git_branch: experiment/hp-r15`.

## C4 — Precision/device gate on every run path (G3)

- **Why:** `precision::ensure_supported` was only called by `bench`; a BF16 run
  via `run`/`selfplay`/`train`/`arena` would not have been refused.
- **What:** `RunConfig::ensure_supported()` maps the device/precision strings to
  typed enums and calls the gate; invoked at the top of all four command paths in
  `crates/recur64-cli/src/phase2.rs`.
- **Evidence:** non-FP32 requests now fail visibly on these paths (gate logic is
  the inherited, tested `precision` module).

## C5 — Real concurrent self-play + concurrency gauge (G1, G2)

- **Why:** the primary branch already discovered a benchmark sweep that was
  secretly running games sequentially. At this commit, `collect_only` (the
  `selfplay` CLI) played games in a single sequential loop, ignoring
  `active_games`/`cpu_workers`.
- **What:**
  - Extracted the coordinator's parallel collect into one function
    `collect_parallel`; both `run` and `collect_only` now use it. A configured
    concurrency value now always means real concurrent execution.
  - Added an in-flight gauge to `InferenceMetrics`
    (`peak_in_flight` in `MetricsSnapshot`) that counts simultaneous evaluator
    calls — the direct proof of concurrency.
  - Added `SelfPlayMetrics` to the run report (requested vs actual workers, peak
    in-flight, games, plies) plus a concurrency line in `report.md`.
- **Evidence (TESTED):**
  - `recur64 selfplay --config configs/smoke.toml`: `peak_in_flight_evaluations
    = 4` (= `cpu_workers`), batch p50 = 3, 0 errors.
  - `recur64 run --config configs/smoke.toml`: `peak_in_flight = 4`, batch
    p50 = 1 (CPU forward ~25 ms makes the 500 µs batch timeout flush early — the
    H0.7 sweep exists to tune this on the GPU).

## C6 — HP documentation and hardware scheduling profile

- **What:** `docs/HP_EXPERIMENT.md` (hardware, configs, divergences, status);
  `configs/hardware/hp-home.toml` (provisional scheduling overlay template);
  this ledger.
- **Evidence:** hardware values were collected read-only; no identifiers recorded.

## C7 — CUDA runtime, schedule overrides, and sweep tooling

- **Why:** enable the GPU path and make hardware-scheduling sweeps possible
  without editing scientific configs.
- **What:**
  - CUDA 12.9.1 user-space runtime installed at
    `%LOCALAPPDATA%\Recur64\cuda\12.9.1` from the pinned redist components
    `cuda_cudart`, `cuda_nvrtc`, **`cuda_nvcc`**, `libnvjitlink`, `libcublas`.
  - Two reproducibility fixes recorded in `HP_EXPERIMENT.md`: the
    `nvrtc64_12.dll` alias cudarc 0.19.9 expects, and the `cuda_nvcc`
    `include/crt` + `nvvm/libdevice` headers nvrtc needs.
  - `ScheduleOverrides` CLI flags on `selfplay`/`run`
    (`--active-games`, `--cpu-workers`, `--max-inference-batch`,
    `--batch-timeout-us`, `--simulations-per-move`, `--ply-cap`).
  - `scripts/hp-concurrency-sweep.ps1` — real concurrency sweep with a hard stop
    when `peak_in_flight ≤ 1` or `batch p50 ≤ 1`.
  - `configs/hp/f15-selfplay.toml`, `configs/hp/r15-selfplay.toml` — standard-start
    CUDA run configs with hardware/model profile labels.
  - Contract tests: `RunConfig` device/precision parsing + gate rejection;
    `RunMetadata` seed/profile/git provenance + JSON round-trip.
- **Evidence (TESTED):** `cuda-smoke` PASS on F15; F15/R15 benchmark matrices
  finite; peak VRAM 1,953 / 2,113 MiB; `cargo test --release` and clippy clean.

## C8 — Merge Phase 3 (`origin/main`) and reconcile

- **Why:** main advanced to `5ac291c` (F10 + PUCT control baseline) with work
  that supersedes several planned HP tasks.
- **What:**
  - Merged `8f14203` + `5ac291c`; resolved conflicts in `config.rs`,
    `coordinator.rs`, `lib.rs`, `run_dir.rs` (README/phase2 auto-merged).
  - Adopted Phase 3's `active_games` = concurrency model as the single
    `collect_parallel` path (used by both `run` and `collect_only`); kept
    `SelfPlayMetrics` (terminations/WDL/truncation + `peak_in_flight`).
  - Kept HP provenance additions (`git SHA`/branch, seed, profile labels) on top
    of Phase 3's `lineage.jsonl`/`config_hash`.
  - Added `configs/hp/{f15,r15}-pilot.toml` using the Phase 3 pilot contract.
  - Superseded tooling: Phase 3's `bench-runtime` replaces the hand-rolled HP
    sweep scripts; `eval-policy` + `openings-v1` replace the planned raw-policy
    harness.
- **Evidence:** merge commit `fa66c32`; CUDA build OK; full test suite OK;
  fmt/clippy clean.

---

## Not yet done (tracked in `HP_EXPERIMENT.md`)

HP batching sweep on the RTX 2050, freeze hardware profile, search-budget
re-measurement, F15 pilot / learning-health analysis, R15 training/evaluation.

## H1 harness implementation (in progress)

The reboot-safe measured status is in [`HP_H1_RESULTS.md`](HP_H1_RESULTS.md).

The current branch now shares one self-play collector across `run`, `selfplay`,
`pilot`, and the runtime sweep. New configs can specify total games and maximum
concurrent games separately; `cpu_workers` caps the actual worker threads.
Legacy `active_games` configs still parse, with `cpu_workers` as the thread cap.
Collection and explicit evaluation opening errors are visible. Pilot training
uses the existing full training checkpoint to continue Adam state and LR steps
only after promotion. Reuse schedules updates from new completed-game plies,
and candidate promotion requires informative parent results.

The workstation F10 baseline in `docs/F10_BASELINE.md` is historical evidence
from different hardware. It is not an HP F15 learning measurement. GPU systems,
search-budget, smoke, and pilot results must be reported separately when run.
