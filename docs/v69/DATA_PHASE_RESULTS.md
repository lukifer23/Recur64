# Recur64 V69 — data-feasibility phase results

Branch `experiment/hp-v69-two-clock-workspace`, created from `main` @
`78be2052612236547f6b5232b417175c2ccdcfc9` (clean, no dirty files) in the new worktree
`Recur64-v69`. Existing worktrees (`Recur64`, `-v5`, `-v6`) and historical evidence were not
modified. Nothing was merged into main. This branch (and only this branch) is pushed to
`origin` regularly at the owner's request. Contracts: `CONTRACT.md` (frozen before the run).
Receipts: `evidence/`. The generated dataset itself stays local under `artifacts/v69/`
(git-ignored; sealed test positions are deliberately not published; only hashes are).

Labels: **MEASURED** (observed here), **PROPOSED** (decided, unvalidated), **NOT RUN**.

## 1. Source lineage and imported components

| Component | Source | Status |
|---|---|---|
| `recur64-core` (rules, cozy-chess 0.3.4, `rules::classify`) | main @ 78be205, unmodified | reused, software only |
| Cargo workspace, `Cargo.lock`, `rust-toolchain.toml` (1.97.1), burn =0.21.0 pin | main | reused; lock diff = +`recur64-v69` entry only |
| `sha2 =0.10.9` | already in `Cargo.lock` | new direct dependency of V69 crate only |
| `recur64-cli cuda-smoke` | main | run as a backend check only (synthetic fixture, no data) |
| Mate teacher | **new code** `crates/recur64-v69/src/oracle.rs`, written from the definition | historical MateSolver/ProofTarget code **not** ported |

Asset-exclusion audit (**MEASURED**): the V69 crate contains no path to, and reads no file
from, any historical run/dataset/checkpoint/cache; all `v69-data` paths pass the custody
guard (tests: refuses `..` escape, absolute outside paths, `runs`/`evidence` components,
historical worktree roots). Empty-cache oracle per root. Tests use hand-built positions and a
TEST-ONLY seed. The generation run used only the fresh seed. Historical positions were not
accessed, so **accidental position overlap with historical data is not excluded** (stated
limitation; the guarantee is independent generation).

## 2. Platform (MEASURED unless noted)

- GPU: NVIDIA GeForce RTX 2050, 4096 MiB, 0 MiB in use at start; driver 616.92.
  Host: 12 logical CPUs, 32 GB RAM, Windows 11. Rust/rustc 1.97.1 (pinned).
- **CUDA FP32 (TESTED, probe graph only):** `recur64 cuda-smoke` built with
  `--features cuda`, ran FP32 forward, backward, AdamW step and a training-checkpoint round
  trip on `Cuda(0)`: PASS, exit 0, 20 s, sampled VRAM max 161 MiB. This validates the
  backend/toolchain, **not the V69 graph** (NOT RUN; V69 has no graph yet). BF16 not tested.
- **Hardware/environment issues found and handled (no silent fallback occurred):**
  1. First smoke attempt failed visibly: `Unable to dynamically load the "nvrtc" shared library`
     → "optimizer did not move GPU parameters" (exit 1). Cause: the repo's documented
     user-space CUDA 12.9.1 runtime (`%LOCALAPPDATA%\Recur64\cuda\12.9.1\bin`) was not on
     PATH. Fix (process-scoped, no install, no driver change): set `CUDA_PATH` and prepend that
     `bin` to PATH for the process. Second attempt passed. **Later phases must do this explicitly.**
  2. `cargo build -p recur64-cli --no-default-features --features cuda` fails on main
     (`recur64-model/src/train.rs` references `burn::backend::Flex` unconditionally); build with
     default features + `cuda`. Log kept: `evidence/cuda_build_attempt1_no_default_features.log`.
     Not fixed here (out of scope; CPU is not substituted at run time — device is chosen by the
     explicit `device` setting).
  3. Memory/time limits are enforced by `scripts/v69/run_limited.ps1` (native exit code,
     reason, peak memory, process-tree kill). The watchdog polls every 500 ms, so very short
     runs (< 1 s audit) record peak 0 MiB; the generation run recorded **46 MiB** peak.
- 4 GB-VRAM feasibility of the V69 model: **NOT RUN** (needs CUDA qualification).

## 3. Seed and artifact identities

Master seed from OS CSPRNG, recorded before generation: fingerprint
**`5710d34e89d9dd35`**, 2026-10-08T21:51:33Z (`evidence/seed_record.txt`). Run id `gen-001`;
source digest and git head in `evidence/generation_report.json` (head `36a8150…`, 0 dirty
files at run time). Hash manifest: `evidence/MANIFEST.sha256.json`. Streams: see CONTRACT §2.
Single run; no seed was searched or redrawn. (A separate smoke run with the TEST-ONLY seed
validated the pipeline; its outputs are not V69 data and it found and fixed two bugs: an
audit rule wrongly requiring halfmove 0 on children, and id collisions from using a key
prefix — ids now derive from SHA-256 of the full key.)

## 4. Measured generation cost and capacity (MEASURED, `gen-001`)

Limits actually applied: 7200 s wall, 8 GiB, 5,000,000 nodes/query, 10 threads.
Outcome: `complete_feasible`, **0.8 s** wall (2 rounds × 1000 attempts per family), peak 46 MiB,
exit 0, no query hit the node limit (`unresolved_node_limit = 0`), no round timed out.

| Family | attempted | overlap | adj. kings | def. in check | duplicate | analysed | M1 rej. | none ≤3 | **M2** | **M3** |
|---|---|---|---|---|---|---|---|---|---|---|
| KQQvK | 2000 | 193 | 178 | 934 | 1 | 694 | 189 | 0 | 419 | 86 |
| KQRvK | 2000 | 168 | 186 | 816 | 1 | 829 | 107 | 133 | 310 | 279 |
| KRRvK | 2000 | 183 | 172 | 643 | 1 | 1001 | 76 | 391 | 290 | 244 |

Illegal-other 0, unresolved 0, timed out 0 in all families. Accepted pool 1628 roots.
Oracle CPU cost (summed over threads): ≈ 0.29 s/0.54 s (KQQ M2/M3 total), ≈ 0.17/1.0 s (KQR),
≈ 0.13/0.51 s (KRR); ≈ 0.4–2 ms per M2 root and ≈ 6 ms per M3 root on average.
Bottleneck: none relevant at this scale; the binding quantity is M3 yield for KQQvK (86 of
2000 attempts → the scarcest cell source), still far inside capacity.

## 5. Dataset, groups, custody receipts (MEASURED)

- Exactly 1536 examples: fit 768 / val 384 / sealed test 384; all 36 (partition × family ×
  budget × class) quotas met exactly (64/32/32). All examples are nonterminal immediate
  children; `terminal_children` per root are counted in `pool/roots.jsonl` and excluded.
- Groups: 1212 connected groups over 1628 roots; sizes (roots) 1:974, 2:155, 3:45, 4:11,
  5:15, 6:2, 7:3, 8:6, 9:1; **largest component 9 roots** (no giant component); 95 groups
  mix M2 and M3 roots (kept together). By partition (groups, roots): fit (605, 814), val
  (303, 407), test (304, 407). Used by the dataset: 403 groups, 453 roots.
- 1536 distinct canonical children; 0 symmetric duplicate keys inside the dataset; no
  group, root or canonical child spans partitions (full 65-byte key comparisons).
- **Audit PASS** (`evidence/audit_receipt.json`, exit 0, 0 failures): manifest hashes;
  id uniqueness; quotas; leakage; no terminal child; canonical keys recomputed; all 453
  contributing roots re-analysed from an empty cache reproduce depth and every child label;
  independent exhaustive reference agreed on 576 sampled child targets (per-cell sample,
  both budgets, both classes, both colours) and 192 sampled M2 root depths.
  Unit tests: 11/11 pass (`cargo test --release -p recur64-v69`); includes solver↔reference
  agreement, stalemate/mate boundary, budget-relative negatives, node-limit no-poisoning.
- Not covered by the independent reference: minimal depth of M3 roots (cost) — covered by
  the oracle re-analysis and child-level reference checks only. The reference shares the
  `cozy-chess` move generator (cross-checked against brute-force `is_legal`; shakmaty
  differential **NOT RUN**).
- Sealed test: written to `sealed/` read-only, hash-pinned; **no model run on it**; labels
  read only by the audit for exactness.

Known properties of the data (not defects of the pipeline, but relevant to interpretation):
negatives are uniform over incorrect moves so many are gross blunders; every example is
defender-to-move, so the attacker-relative indicator is constant; children carry halfmove
clock 1 (domain: no history, restricted to these three families).

## 6. Architecture and parameter estimate

See CONTRACT §9–§11 (all **PROPOSED**; **NOT RUN**). Estimate ≈ **1.877 M** parameters
(arithmetic in the contract), below the stated 2–3 M target and under the 4 M ceiling —
**decision for the owner**: accept ≈1.9 M with the specified dims, or amend a dimension.

## 7. Status summary

| Item | State |
|---|---|
| Fresh generator, exact teacher, grouping, partitions, audit | MEASURED, pass |
| Data feasibility (quotas, limits) | MEASURED, **feasible** (0.8 s, 46 MiB) |
| CUDA FP32 backend (probe graph) | MEASURED, pass with NVRTC PATH fix |
| V69 model graph, 4 GB VRAM fit, latency | NOT RUN |
| Three-arm fits, validation gates, derangement control | NOT RUN (PROPOSED) |
| Sealed test inference | NOT RUN (forbidden this pass) |
| BF16, shakmaty differential, M3 independent root-depth check | NOT RUN |

## 8. Next prompt (CUDA qualification and bounded fitting)

> Build Recur64 V69 on branch `experiment/hp-v69-two-clock-workspace` (current head; do not
> merge or touch other worktrees). Read AGENTS.md and `docs/v69/CONTRACT.md` +
> `DATA_PHASE_RESULTS.md`. The fresh dataset is `artifacts/v69/gen-001` (manifest
> `docs/v69/evidence/MANIFEST.sha256.json`); verify hashes before use and keep every path
> behind `Custody`. Never read `sealed/`. Environment: set `CUDA_PATH` and prepend
> `%LOCALAPPDATA%\Recur64\cuda\12.9.1\bin` to PATH; build with default features +
> `cuda`; use `scripts/v69/run_limited.ps1` for all runs (2 h/arm, 8 GiB host). No Python
> training, custom autodiff/kernels, Docker, driver changes, CPU fallback.
> Phase A (qualification, no fitting): implement the V69 model in burn 0.21.0 exactly as
> CONTRACT §9; report actual parameter count; run FP32 forward+backward+AdamW on CUDA for
> arms A/B/C at microbatch 2 and effective batch 16; verify shared parameters in the fast and
> slow blocks receive gradients from repeated calls; record peak VRAM (must fit 4096 MiB),
> latency and FLOP/latency ratio B:C; verify bit-identical initial tensors across arms from the
> `model_init` stream; checkpoint+optimizer round trip. Stop if VRAM/latency fail and report.
> Phase B (only if A passes): the frozen 600-update fits for arms A, B, C on `fit.jsonl`
> (CONTRACT §10), fixed endpoint, evaluate on `val.jsonl` only with the §11 gates, the
> target-independent within-(family,budget) derangement (from the `intervention` stream),
> donor-label accuracy, board erasure, per-cell metrics, Brier, group-bootstrap uncertainty,
> latency, memory. Distinguish MEASURED/PROPOSED/NOT RUN. Commit and push this branch after
> each milestone. No sealed-test inference, search, self-play, or merge.
