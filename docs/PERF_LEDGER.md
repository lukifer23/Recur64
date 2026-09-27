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
