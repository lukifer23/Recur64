# Recur64 HP — throughput / latency ledger

**Rules:**
- One change per entry, and one GPU job at a time. No builds or tests run
  during a timing run, and a keep-awake hold covers every run.
- Each entry records:
  - hypothesis, change, exact command, git SHA;
  - before / after with raw JSON under `docs/evidence/perf/`;
  - **output parity** against the saved baseline outputs;
  - decision (KEEP / REVERT / DEFER).
- Short probes (1–3 min) for iteration. One full-length validation before
  training resumes.
- An optimization that changes what the network computes beyond float noise
  is a scientific change: it needs an ADR, not just a ledger entry.

**Exit criteria:**
- **Target:** at least 2× on a full pilot cycle vs H3.6, which ran
  960–1,518 s per cycle, 52–75 % of it evaluation.
- **Stop early** if a round of changes gains less than 5 %.

**Probes:**
- `recur64 bench-forward` times the production `BatchedModel::evaluate_batch`
  path on 256 reproducible positions. It reports per-batch p10/p50/p90 and
  evals/s, plus a phase breakdown (prepare+upload / forward submit / readback
  incl. device wait / host post) and a parity check against a baseline.
- `bench-runtime` runs short self-play (8 games).
- `eval-arena` runs short arenas (8 games).
- `bench-train` measures the training step.

## Entries

### #1 — evaluation scheduling (pre-registered R15-P0.5 M3) — REJECT (gate)

- **Hypothesis:** eval batches are half-empty (4–6 of 16), so 32 concurrent
  games and a batch cap of 32 will fill them.
- **Setup:** the H3.6 cycle-2 reference arena (`d0ee3ced` vs `d89b408f`),
  root_player_v1, paired RNG, K = 2, 32 sims, 32 games. Binary `049d2a5`.
  Evidence: `docs/evidence/hp-r15-p0/d54-impact/m{1,3}-*`.

| schedule | wall | GPU util mean | peak VRAM | errors |
|---|---|---|---|---|
| c8 / batch 16 (M1) | 587 s | 74 % | 417 MB | 0 |
| **c32 / batch 32 (M3)** | **489 s (1.20×)** | **91 %** | 417 MB | 0 |

- **Parity:** **32/32 games move-for-move identical**, so scheduling does not
  change results even with CUDA batching. The adjudicated score is 0.750 in
  both.
- **Decision:** **REJECT** for R15 under the pre-registered ≥ 1.3× bar. The
  knobs stay available as execution-only settings.
- **Finding:**
  - The GPU is now ~91 % busy, so the arena is compute-bound. Further gains
    need a cheaper forward.
  - The wall-time tail is the 11/32 games that ran the full 400 plies
    (unconverted wins).
  - **Candidate lever (scientific, needs an ADR):** adjudicate clearly
    decided games before the cap.

### #2 — baseline profile (MEASURED, `3b06eae`)

- **Probe:** `bench-forward` on the R15 reference at R1 and R4, 256 positions,
  20 iterations.
- **Evidence:** `docs/evidence/perf/00-baseline/`.

| batch | R1 p50 ms | R1 evals/s | R4 p50 ms | R4 evals/s |
|---|---|---|---|---|
| 1 | 6.6 | 152 | 14.9 | 67 |
| 8 | 15.0 | 532 | 32.7 | 245 |
| 16 | 26.4 | 607 | 63.0 | 254 |
| 32 | 52.9 | 605 | 127.8 | 250 |
| 64 | 104.9 | 610 | 254.8 | 251 |

**Findings:**
- **Per-position GPU cost dominates.** Throughput plateaus from batch ≈ 8–16,
  with a marginal cost of 1.64 ms per position at R1 and 3.97 ms at R4.
- **Host prepare and post-processing are under 1 % of batch time.** The
  planned host-side micro-fixes (`rel_idx` cache, single readback, copies)
  are therefore not worth doing, and are dropped.
- **Achieved throughput is about 1.2 TFLOP/s FP32** (INFERRED from about
  2 GFLOP per position at R1).
- The production operating point (K = 2 × c8 = batch ≤ 16) already sits at
  the plateau, which is why M3's extra concurrency gave only 1.2×.

### #3–#5 — fusion and candidate-width bucketing (MEASURED)

**Setup:**
- The trained H3.6 network `d0ee3ced` (F15, R1). Parity is against A, so a
  real, non-zero value head is exercised.
- Builds: `--features cuda` for A/B and `--features cuda,fusion` for C/D.
  Both are built from the same source.
- Evidence: `docs/evidence/perf/0{2,3,4,5}-*`, plus the R15 R1/R4 fusion run
  in `01-fusion`.

| variant | b8 evals/s | b16 | b32 | b64 | parity (max policy / value abs diff) |
|---|---|---|---|---|---|
| A baseline | 413 | 570 | 592 | 595 | — |
| B + candidate buckets | 556 | 594 | 594 | 595 | 2.2e-8 / 0 |
| C fusion | 527 | 721 | 757 | 782 | **0 / 0** (bit-identical) |
| **D fusion + buckets** | **600 (+45 %)** | **721 (+26 %)** | **761 (+29 %)** | **780 (+31 %)** | 2.2e-8 / 0 |

- **R4, fusion without buckets, on the R15 reference:** b16 254 → 321 (+26 %),
  b64 251 → 332 (+32 %).
- **Why bucketing helps:** the candidate width (the maximum legal-move count)
  changes almost every batch, and the backend compiles kernels per shape.
  Rounding the width to 8 fixed buckets removes that churn. Padded slots are
  exactly masked (test `candidate_bucketing_does_not_change_outputs`).
- **Decision:** D is the **leading candidate**. It is not adopted yet. Pending
  before adoption:
  - autotune (#6);
  - the fused training step (Autodiff over Fusion): parity and speed;
  - a D44 VRAM lifecycle re-check (fusion's `memory_cleanup` passes through);
  - short end-to-end self-play and arena probes;
  - an ADR, since this is a backend build-configuration change.

### #6 — + autotune (fusion + buckets + autotune) (MEASURED)

- **Build:** `--features cuda,fusion,autotune`, on the same trained `d0ee3ced`,
  with 6 warmup iterations.
- **Evidence:** `docs/evidence/perf/06-fusion-bucket-autotune/`.

| batch | A baseline | D fusion + buckets | **+ autotune** | vs A |
|---|---|---|---|---|
| 8 | 413 | 600 | **674** | +63 % |
| 16 | 570 | 721 | **789** | +38 % |
| 32 | 592 | 761 | **854** | +44 % |
| 64 | 595 | 780 | **875** | +47 % |

- **Parity:** max |Δ| is 9.1e-5 on policy and 1.3e-5 on value. This is float
  noise from different matmul kernels; values are not bit-identical.
- **Trade-off:** autotune picks kernels by timing, so exact cross-process CUDA
  replay (used for the H3.6 root-cause replay) may no longer be bit-exact.
  Within one process nothing changes, including CRN pairing.
- **Decision:** leading candidate. The next checks are training, VRAM and
  end-to-end.

### #7 — training step under fusion + autotune (MEASURED)

- **Setup:** `bench-train` on the trained `d0ee3ced`, the H3 replay, 32 × 4, 20
  updates, same seed.
- **Evidence:** `docs/evidence/perf/07-train-*`.

| build | ex/s | step | loss first → last | max grad | peak VRAM |
|---|---|---|---|---|---|
| baseline | 145.4 | 880.6 ms | 4.2508 → 4.0257 | 4.626 | 1,281 MB |
| fusion + autotune | **161.9 (+11 %)** | 790.8 ms | **4.2508 → 4.0257** | **4.626** | 1,287 MB |

The loss trajectory and gradient norm are identical to printed precision.
Training is about 5 % of cycle wall time, so this is a minor contributor.

### #8 — D44 lifecycle under fusion: leak found, root-caused, fixed (MEASURED)

**Initial finding (FAIL):** the full 4-mode check on the fusion + autotune build
(`docs/evidence/perf/08-lifecycle-fusion-autotune/`):
- post-shutdown VRAM grew **~64 MB per owner lifecycle** without plateau:
  291 → 357 → 549 → 613 → 677 → 741 → 805 → 901 MB;
- 0 errors.
- Across a pilot (3 owners per cycle) this would exhaust the 4 GB card.

**Isolation, with 4-minute probes (`two` mode, 3 reps each):**

| build | post-shutdown VRAM by rep | verdict |
|---|---|---|
| fusion + autotune + sync-before-cleanup (`08b`) | 419 → 483 → 547 | still leaks |
| fusion only (`08c`) | 417 → 481 → 545 | **fusion is the cause**, not autotune |
| **fusion + drop-order fix (`08d`)** | **417 → 449 → 449** | **plateau, same as the baseline build** |

**Root cause (VERIFIED by the fix):**
- `BatchedModel::drop` ran sync and cleanup *before* its fields dropped.
- On the plain backend a tensor frees immediately. Under fusion, freeing a
  tensor is a **queued operation**.
- So the model's parameter frees were queued after the final sync on an owner
  thread that then exited, and never ran. The ~64 MB leaked per rep is about
  one F15 parameter set (15.15 M × 4 B ≈ 61 MB).

**Fix:** the owner releases the model explicitly, then syncs, then cleans up.
This is harmless on the plain backend.
