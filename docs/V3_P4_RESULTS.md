# Recur64 V3 - P4 results (data, ProofTraceV1, frozen feasibility measurement)

Status: MEASURED. The rules, the code and the specification were committed before any TRAIN trace was
generated (commit `156cc0f`, `docs/V3_P4_PLAN.md`, decisions V3-D11 to V3-D14). Nothing in the rule was
changed after the measurement. This file records what was observed; it is not edited to reinterpret it.

## Result

| | |
|---|---|
| primary cell | **KQRvK M3** (P25_DATA_V1 TRAIN, every position, no sampling) |
| primary metric | **C_8** = fraction of positions whose minimal complete certificate has at most 8 query edges |
| n | 5000 |
| positions with Q* <= 8 | 2168 |
| measured value (unrounded) | **0.4336** |
| frozen threshold | 0.25 (rule `count_le_8 * 4 >= n`, exact integers) |
| pass | **True** |
| classification | **SCIENTIFICALLY QUALIFIED FOR THE B8 PRIMARY EXPERIMENT** |

What this means, and what it does not. It means only that B8 can contain a complete ideal certificate
often enough in the frozen primary stress cell for the planned neural experiment to be interpretable.
It does not mean the model learns such certificates, that B8 will beat B0, that ACTIVE will beat FIXED,
or that V3 succeeds. The measurement is a property of the exact TRAIN data under `proof_trace_v1`; it
needs no bootstrap.

P5 NOT RUN - awaiting owner review and approval.

## Custody (read-only)

- P25_DATA_V1 TRAIN loaded through `ProofTargets::load`: 44332 positions, content digest
  `3b25dc8549dd2fc9d47c30e294c273b3306aecb3eba91b964715326ddf74f2e6` (read from the file), schema `proof_targets_v1`, history contract
  `fresh_no_history_v1`. The existing independent label audit checked 44332
  positions with 0 failures. 44332 unique exact FENs and
  44332 unique canonical classes (recomputed keys equal the stored ones). Overlap with HOLDOUT_C:
  0 canonical classes, 0 FENs.
- HOLDOUT_C: expected `4ab951c6edd8dd4f531bb87d2f4373895d1fddb70efdf052b24c09a1b71d87d5`, verified `4ab951c6edd8dd4f531bb87d2f4373895d1fddb70efdf052b24c09a1b71d87d5`
  (4500 positions), **evaluated = false**. Use in P4:
  custody and disjointness only: integrity check, canonical-class and exact-FEN exclusion. No ProofTrace, feasibility number, metric or position-level output was produced
  for it, and no exposure was logged (integrity checking is custody; see V3-D13). Seal:
  `docs/evidence/v3/v3-confirm-seal.json`.
- Exclusion inventory (70624 canonical classes, manifest digest
  `b3d5a7eeab183be9b501aea503ef2511857ffe2e39d5d407c2f1092743314cb7`):

| split | positions | canonical classes | file |
|---|---:|---:|---|
| train | 44332 | 44332 | `runs/v25/p25/data/proof-train.json` |
| tune | 1208 | 1208 | `runs/v25/p25/data/proof-tune.json` |
| holdout_a | 4500 | 4500 | `runs/v25/p25/holdouts/proof-holdout_a.json` |
| holdout_b | 4500 | 4500 | `runs/v25/p25/holdouts/proof-holdout_b.json` |
| holdout_c | 4500 | 4500 | `runs/v25/p25/holdouts/proof-holdout_c.json` |
| confirm | 1208 | 1208 | `runs/v25/proof/proof-confirm.json` |
| train | 11501 | 11501 | `runs/v25/proof/proof-train.json` |
| tune | 1208 | 1208 | `runs/v25/proof/proof-tune.json` |
| confirm | 1208 | 1208 | `runs/v25/proof-v2/proof-confirm.json` |
| train | 11501 | 11501 | `runs/v25/proof-v2/proof-train.json` |
| tune | 1208 | 1208 | `runs/v25/proof-v2/proof-tune.json` |

- Limitation, stated and not hidden: HP/X1/X2 exact datasets are not present on this workstation in a
  compatible form, so disjointness from them was **not** verified.

## v3_tune_v1

Identity `v3_tune_v1`, seed `0x7a130004`, KQRvK and KRRvK x M1..M3, 750 per cell,
4500 positions, digest `c66018657009c9c5eade58369b5f451466d6662c910f76b8aacddcac99921b53`. Every cell supplied exactly 750; nothing
was relaxed.

| cell | eligible pool | excluded | available | taken |
|---|---:|---:|---:|---:|
| KQRvK M1 | 111273 | 7884 | 103389 | 750 |
| KQRvK M2 | 306595 | 7898 | 298697 | 750 |
| KQRvK M3 | 211215 | 7891 | 203324 | 750 |
| KRRvK M1 | 41612 | 7864 | 33748 | 750 |
| KRRvK M2 | 122082 | 7893 | 114189 | 750 |
| KRRvK M3 | 108086 | 7884 | 100202 | 750 |

Audit: 4500 positions through the independent label audit, 0
failures. Disjointness: 4500 unique FENs and
4500 unique canonical classes inside the set, 0 overlap with any
of the 11 inventoried datasets (checked directly and pairwise).
Deterministic regeneration from the seed: identical digest = True.

## ProofTraceV1

Implementation summary: AND/OR proof graph (every winning attacker move, every legal defender reply),
exact `Q*` in query edges, refutation sets for incorrect root moves, edge identity by root-relative
action path, set-valued `A(S)` by dynamic programming over residual costs keeping every tie, sharded
resumable store, independent audit, live teacher adapter over the real `Tree`. See
`docs/V3_P4_PLAN.md` for the contract.

- TRAIN: 44332 traces in 89 shards, trace manifest digest `8160734ed5e3a145c12dd72d8e9dc49cc893984fc1cf714ea93eba904f58c488`.
- Independent audit of every trace: checked 44332, failures 0, audit manifest digest
  `895636669bbc149a89c0de7e1512606f5c674c5d626309f7260c9144401003b0`.
- Determinism: shards 0, 30, 60 and 88 regenerated on 1 and 20 threads are identical to each other and to the stored
  shards; the feasibility output is byte-identical on a re-run.
- `v3_tune_v1`: 4500 traces, 9
  shards, manifest digest `acb786fad82573c9b4427652e5e1b060f7b312c5f5e9121e02980c21cfaa6d25`, audit
  4500 checked / 0 failures.
- 19 tests (see `docs/evidence/v3/v3-p4-prooftrace.json`): DP target versus brute-force enumeration of every
  certificate, live order-invariance, completion in exactly `Q*` queries, ten audit corruptions on M2 and on M3
  traces, thread-count determinism, resume and corruption refusal, exact threshold boundaries, custody guards,
  and the answer-free packet.

## Q* distribution and C_k, P25_DATA_V1 TRAIN (the gating table)

| family | M | n | Q* min | median | p90 | p95 | max | C2 | C4 | C8 | C16 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KQQvK | 1 | 5000 | 1 | 1 | 1 | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| KQQvK | 2 | 5000 | 3 | 3 | 5 | 7 | 13 | 0.0000 | 0.7778 | 0.9922 | 1.0000 |
| KQQvK | 3 | 1831 | 5 | 9 | 17 | 19 | 33 | 0.0000 | 0.0000 | 0.2944 | 0.8624 |
| KQRvK | 1 | 5000 | 1 | 1 | 1 | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| KQRvK | 2 | 5000 | 3 | 3 | 5 | 7 | 13 | 0.0000 | 0.7214 | 0.9944 | 1.0000 |
| KQRvK | 3 | 5000 | 5 | 9 | 21 | 25 | 37 | 0.0000 | 0.0000 | 0.4336 | 0.8142 |
| KQvK | 1 | 246 | 1 | 1 | 1 | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| KQvK | 2 | 462 | 3 | 3 | 5 | 5 | 7 | 0.0000 | 0.8831 | 1.0000 | 1.0000 |
| KQvK | 3 | 862 | 5 | 7 | 11 | 17 | 21 | 0.0000 | 0.0000 | 0.6671 | 0.9327 |
| KRRvK | 1 | 5000 | 1 | 1 | 1 | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| KRRvK | 2 | 5000 | 3 | 3 | 5 | 7 | 9 | 0.0000 | 0.7656 | 0.9956 | 1.0000 |
| KRRvK | 3 | 5000 | 5 | 9 | 21 | 29 | 37 | 0.0000 | 0.0000 | 0.3856 | 0.8648 |
| KRvK | 1 | 153 | 1 | 1 | 1 | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| KRvK | 2 | 426 | 3 | 3 | 3 | 3 | 3 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| KRvK | 3 | 352 | 5 | 9 | 11 | 11 | 13 | 0.0000 | 0.0000 | 0.4176 | 1.0000 |

Pooled by mate depth:

| family | M | n | Q* min | median | p90 | p95 | max | C2 | C4 | C8 | C16 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 1 | 15399 | 1 | 1 | 1 | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| ALL | 2 | 15888 | 3 | 3 | 5 | 7 | 13 | 0.0000 | 0.7652 | 0.9944 | 1.0000 |
| ALL | 3 | 13045 | 5 | 9 | 19 | 25 | 37 | 0.0000 | 0.0000 | 0.4107 | 0.8532 |

Unweighted macro mean over cells: C2 0.3333, C4 0.6099, C8 0.8120, C16 0.9649.

Highlights (MEASURED). M1 always costs exactly 1 edge. M2 costs 3 to 13 edges, with C2 = 0 and C8 above 0.99
in every heavy family. In the primary cell KQRvK M3 the minimal certificate has
5 to 37 edges, median 9, p90 21, p95 25;
C2 = 0.0000, C4 = 0.0000, **C8 = 0.4336**, C16 = 0.8142. The neighbouring M3
cells have C8 of 0.2944 (KQQvK, n = 1831), 0.3856 (KRRvK), 0.6671 (KQvK) and 0.4176 (KRvK). The minimum
certificate sizes follow the structure: 1, 3 and 5 edges for a forced mate in 1, 2 and 3 when the defender has
a single reply at each step.

## v3_tune_v1 diagnostics (not part of the rule)

| family | M | n | Q* min | median | p90 | p95 | max | C2 | C4 | C8 | C16 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KQRvK | 1 | 750 | 1 | 1 | 1 | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| KQRvK | 2 | 750 | 3 | 3 | 5 | 7 | 11 | 0.0000 | 0.7227 | 0.9960 | 1.0000 |
| KQRvK | 3 | 750 | 5 | 9 | 21 | 25 | 35 | 0.0000 | 0.0000 | 0.4320 | 0.8093 |
| KRRvK | 1 | 750 | 1 | 1 | 1 | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| KRRvK | 2 | 750 | 3 | 3 | 5 | 7 | 9 | 0.0000 | 0.7533 | 0.9947 | 1.0000 |
| KRRvK | 3 | 750 | 5 | 9 | 21 | 29 | 35 | 0.0000 | 0.0000 | 0.3733 | 0.8507 |

The primary cell on `v3_tune_v1` (KQRvK M3, n = 750) has C8 = 0.4320, close to the TRAIN
value. This is recorded as a diagnostic only; it did not enter, and could not alter, the frozen rule.

## Runtime and storage (relevant to P5)

- TRAIN trace generation: about 4 s on 20 threads for 44,332 positions; independent audit: 21 s; TUNE
  generation with audit, disjointness checks and a second regeneration: 1 m 41 s. None approaches the two-hour
  rule, so sharding was a safeguard, not a necessity.
- Storage: TRAIN traces 161 MB in 89 JSON shards, TUNE traces 16 MB, `v3_tune_v1` 2.1 MB (git-ignored run
  storage; only manifests and summaries are committed). Loading and validating all TRAIN shards takes under a
  second. A P5 teacher that holds every trace in memory should expect a few hundred MB; lazy per-shard loading
  or a binary encoding are options, not requirements.
- `A(S)` is computed by dynamic programming over at most a few dozen queried edges, so it is cheap per step.

## Incidents and deviations

- The first `trace-gen` invocation was cut short by my own `| head` pipe (broken pipe) after five shards had
  been written. It was re-run without the pipe: the five finished shards were re-validated and reused, the rest
  generated. This is the designed resume path; no shard was reused without validation.
- The mate-in-three audit-corruption test was added after the feasibility measurement as a test-only
  strengthening. No production code changed after the measurement, and the measurement was taken once.
- `GameState::from_fen` accepts illegal positions (adjacent kings, an attacked opponent king); not reachable
  from the solver-verified data used here. Unchanged since P3.

## NOT RUN

P5 to P11 in full: no LR screen, no information-sufficiency control, no active training, no model evaluation.
HOLDOUT_C has not been evaluated, traced or analysed. No fusion, graph capture, cache or architecture change.
