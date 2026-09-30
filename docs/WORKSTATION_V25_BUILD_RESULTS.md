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
| P0: CUDA known-answer guard, no memory growth over build/forward/drop lifecycle | NOT RUN (P0.6) |
| P0.5 facts cost | NOT RUN |
| P0.6 CUDA qualification | NOT RUN |
| Proof datasets, P1, P2, P3, P4 | NOT RUN |

## Notes
- `SyncEvaluator` previously fell back to a uniform policy on zero/NaN legal mass
  (the batched evaluator already refused). It now refuses too, per AGENTS.md
  "no silent fallback". This is a behaviour change on a failure path only.
- `PolicyOutput.base_all` became `Option` (Probe head only) and `CandidateTensors`
  gained the raw promotion code. Probe outputs and all Probe tests are unchanged.
