# Workstation V2.5 — build results

Status: IN PROGRESS. Nothing below is claimed until a gate has actually run.
TESTED = a test or command executed and passed on this machine. DETECTED = observed
without a gating test. NOT RUN is listed explicitly.

- Base main SHA: `fef1ffcf9c38381d4adc671e5e2c5ead9f141e33` (main had not advanced; `git fetch` run).
- Branch: `experiment/workstation-v25`; safety tag `main-pre-workstation-v25-fef1ffc`.
- HP branch `origin/experiment/hp-r15-h3-integration` advanced during the fetch
  (`a8aef66` → `79ffd14`). CandidateFactsV1 and the exact mate-in-2 generator
  (`x2_data.rs`) exist there and were inspected read-only. The first exploration pass
  found neither because the tracking ref was stale. Nothing was imported; semantics
  were re-implemented independently.

## Exact architecture and parameters (TESTED, CPU build)
`candidate_v25`: board width 640, 10 heads x 64, FFN 1280, 8 unique blocks, no
input/output blocks, one pass; candidate dim 256, 4 heads, FFN 512, 1 block; facts
encoder 8→64→256; policy scorer 256→128→1; WDL = Linear(mean(B)→3), zero-init.

| Group | Parameters |
|---|---:|
| input projection | 76,800 |
| square embeddings | 40,960 |
| board blocks (8) | 26,278,480 |
| final norm | 640 |
| from/to candidate projection | 327,936 |
| global candidate projection | 164,096 |
| promotion embedding | 1,280 |
| facts encoder | 17,216 |
| token norm | 256 |
| candidate block (1) | 526,592 |
| policy scorer | 33,025 |
| WDL head | 1,923 |
| **Total (CF and C0, identical)** | **27,469,204** (104.8 MiB fp32) |

Beside it: V25-LargeLegacy (cell L, `ProbeModel` at the same board geometry) =
26,809,944; historical F10 = 9,805,672 (V2.5 is 2.80x larger; descriptive, not a
matched control). Evidence: `docs/evidence/v25/model-info-*`.

## Fresh prior (TESTED, 48 deterministic random-walk positions, full geometry, CPU)
Mean policy entropy / uniform entropy = 0.99999; worst top-1 probability = 1.03x
uniform. Fresh WDL logits are exactly 0 (value exactly 0).

## Gate table
| Gate | State |
|---|---|
| Historical identity: frozen P4.5 hash, legacy `meta.json`, Probe serialization has no new keys | TESTED, pass |
| CandidateFactsV1 vs independent `apply()`+FEN-material reference | TESTED, identical on 41,353 random-game positions + perft/edge fixtures + repetition shuffle |
| Facts alignment across the search/evaluator/inference-channel boundary | TESTED, pass (single and multi-leaf; missing/misaligned facts are errors) |
| P0: construction, finite forward, policy sums to 1, padding exactly 0, terminal bypass | TESTED, pass |
| P0: candidate-order permutation equivariance | TESTED, pass |
| P0: C0 independent of fact values; CF responds | TESTED, pass |
| P0: nonzero finite facts gradient on update 1 | TESTED, pass |
| P0: WDL neutral, fresh entropy >= 0.98x uniform | TESTED, pass |
| P0: checkpoint round trip exact (CPU), cross-architecture refusal both directions | TESTED, pass on real checkpoint directories |
| P0: parameter count in 26–29M | TESTED, 27,469,204 |
| P0: fmt / clippy / workspace release tests | TESTED, clean / clean / green |
| P0: CUDA known-answer guard (`verify_device` before any GPU work) | TESTED, passed on the RTX 2000 Ada (`v25-qual` CUDA runs) |
| P0: no memory growth over build/forward/drop lifecycle | TESTED on CUDA, 4 reps: growth after rep 2 = +32 MiB (rule <= 256 MiB): pass |
| P0.5 CandidateFacts CPU cost | TESTED: see below (1.4% of batch-eval wall proxy; under the 20% trigger) |
| P0.6 CUDA forward and learner layouts | TESTED: see below |
| ProofTargetsV1 generation, audit, disjointness | TESTED: see below |
| P1, P2, P3, P4 | NOT RUN |

## P0.5 — CandidateFacts CPU cost (TESTED, 1,024 deterministic positions, CPU)
131,725 positions/s, 4.17M candidate moves/s, mean legal width 31.6, per-position
p50 7.4 us / p95 11.5 us. In a batch-32 evaluation proxy (observation+legal 0.17 ms,
facts 0.35 ms, CUDA forward 24.1 ms) facts are 1.4% of evaluation wall. The proxy
excludes search overhead; the real self-play share is re-measured at P4 scheduling.
Evidence: `docs/evidence/v25/cuda/v25-qual-candidate-v25-cf-cuda.json`.

## P0.6 — CUDA qualification (TESTED on the RTX 2000 Ada, FP32, no fusion/autotune/TF32)
Forward latency p50 (upload + forward + readback, CF; batch means are polluted by
one-off kernel JIT for new candidate widths, so p50 is the honest figure):

| batch | 8 | 16 | 32 | 48 | 64 | 96 |
|---|---:|---:|---:|---:|---:|---:|
| CF p50 ms | 28.6 | 51.4 | 26.2 | 36.5 | 51.2 | 83.9 |
| CF positions/s (mean) | 49 | 258 | 1,242 | 1,064 | 1,250 | 1,051 |
| L p50 ms | 11.6 | 16.1 | 23.1 | 35.8 | 50.2 | 82.7 |
| L positions/s | 222 | 551 | 1,377 | 1,293 | 1,265 | 1,125 |

Throughput plateaus near 1,250-1,380 positions/s at batch 32-64 for both models, and
CF is within ~13% of L there. At small batches CF is clearly slower (p50 28.6 vs 11.6 ms
at batch 8, 51.4 vs 16.1 ms at batch 16). The cause is NOT isolated: it could be launch
overhead from the extra candidate/facts kernels or residual kernel JIT for new candidate
widths (only 10 timed reps per batch). This matters for inference-owner scheduling at
low fill and is carried into the P4 scheduling qualification rather than assumed away.

Learner layouts, effective batch 256, finite loss and gradients in every cell
(peak VRAM from 500 ms `nvidia-smi` sampling, which can miss short peaks; it includes
~0.3 GB of other GPU use):

| layout | CF s/update | CF ex/s | CF peak VRAM | L s/update | L ex/s | L peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| 32x8 | 0.92 | 277.7 | 1.9 GB | 0.83 | 309.3 | 1.9 GB |
| 64x4 | 0.92 | 278.8 | 3.2 GB | 0.92 | 278.7 | 3.2 GB |
| 128x2 | 1.01 | 253.8 | 4.8 GB | 1.01 | 252.9 | 5.0 GB |

All fit far under the 12 GB ceiling. The pre-registered rule (fastest stable layout)
selects 64x4 for CF and 32x8 for L; the two are within noise for CF, so **64x4 is used
for every cell** for comparability (the learner reduction is an example-weighted mean,
so the layout does not change the update's math). Evidence: `docs/evidence/v25/cuda/`.

## ProofTargetsV1 datasets (TESTED)
Generated from the EXACT pools (exhaustive enumeration of every placement, symmetry-
canonical classes, filters applied) under the pre-registered scale rule in
`WORKSTATION_V25_EXPERIMENTS.md` E-DATA-1. Every position of every split was re-derived
through an independent implementation (`GameState::apply`, full termination
classification, separate memo) with 0 disagreements; splits are hard-disjoint by exact
FEN and canonical class. Generation is deterministic (identical digests at 20 and 8
threads).

| split | positions | seed | digest (sha256) | chance top-1 M1 / M2 / M3 |
|---|---:|---|---|---|
| TRAIN | 11501 | 0x7a110001 | `8d6440d214e606a7f01ad0584d292cf6ac500495040bda96816fb9f599e74cab` | 0.044 / 0.058 / 0.073 |
| TUNE | 1208 | 0x7a110002 | `d48ccf8616091437210e2c21f5ad33638a1847bd8220abf6c36b2f510c0185a3` | 0.045 / 0.058 / 0.072 |
| CONFIRM | 1208 | 0x7a110003 | `ed36378381eed0a05958c2302752b219a1d2b2f94996bb4e373e9a9510ccc101` | 0.042 / 0.058 / 0.071 |

Pool limits (eligible canonical classes; the reason the two-piece cells are smaller):
KQvK 306 / 576 / 1,076 and KRvK 189 / 532 / 438 for M1 / M2 / M3; every other family
has at least 4,409 per cell. Per-family CONFIRM counts for KRvK M1 (18), KQvK M1 (30) and
KRvK M3 (43) are small, so per-family numbers for those cells carry wide intervals.
Generator benchmark gate (before full generation): ~15,000 / 4,100 / 185 accepted
positions per second for M1 / M2 / M3 on 20 threads; the whole request projected to
under one minute, so the scale limit is pool size, not time. Evidence:
`docs/evidence/v25/proof/`.

**Bug caught by a repeat run:** the first exhaustive generation was not reproducible
(digests changed between runs) because the stored FEN was whichever symmetric
representative a worker thread saw first. Fixed by storing the canonical representative
itself; `enumeration_is_independent_of_the_thread_count` now guards it. The digests above
are from the fixed generator; earlier digests from the buggy generator were discarded.

## Notes
- `SyncEvaluator` previously fell back to a uniform policy on zero/NaN legal mass
  (the batched evaluator already refused). It now refuses too, per AGENTS.md
  "no silent fallback". This is a behaviour change on a failure path only.
- `PolicyOutput.base_all` became `Option` (Probe head only) and `CandidateTensors`
  gained the raw promotion code. Probe outputs and all Probe tests are unchanged.


---

## FINAL STATUS (supersedes the gate table above, which predates P1-P2.5)
The table near the top listed P1, P2, P3 and P4 as NOT RUN. Current state: P0, P0.5, P0.6, P1a, P1b,
P2, P2.5-F and P2.5-D have all been run and recorded in `WORKSTATION_V25_EXPERIMENTS.md` and
`WORKSTATION_V25_P25_RESULTS.md`; P3 and P4 have NOT been run and are not authorized by the
pre-registered rules. The P2 and P2.5 outcomes are summarized in `WORKSTATION_V25_SUMMARY.md`.
Later corrections to this document's early numbers: the replacement split digests and sizes are
in the P2 addendum of the ledger; LF is 26,810,584 parameters (contract 2).
