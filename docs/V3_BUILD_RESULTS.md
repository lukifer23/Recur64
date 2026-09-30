# Recur64 V3 - Engineering Qualification (build results)

Engineering only. Nothing here is a science result. No HOLDOUT_C exposure, no TUNE/CONFIRM
evaluation, no training recipe.

## P1 - StateQueryV1 (`recur64-statequery`)  [MEASURED]

Commits `bb6a5c0`, `e61c2e5`, `3072caf`. 35 tests, fmt and clippy clean.

- Differential against `GameState::apply` (reference decodes ActionIds itself): 203,426 random
  descent edges, 105,670 fixture edges (castling, en passant, promotion, stalemate, fifty-move,
  mate), start-position depth-3 BFS. Every packet equals the reference child.
- Packet field whitelist frozen; dependency boundary enforced by test (core, serde, sha2 only).
- Digest contract proved by mutation: every state-content field changes `state_digest`; every
  `QueryIdentity` component changes its digest; ephemeral handles change neither.
- One successful query = one stored-set membership check, one direct ActionId decode, one
  authoritative `apply`, one child legal generation (`legal_generations == 1 + queries`). Refused
  queries perform no chess work.

## P2 - `active_search_v3` model skeleton on CPU FP32  [MEASURED]

Commands:
`cargo test -p recur64-model --release --test active_v3`;
`cargo test --workspace --release`; `cargo fmt --all -- --check`;
`cargo clippy --workspace --all-targets`.

Result: 17 new engineering tests pass; the whole workspace release suite passes (exit 0) including
every historical V2.5 and mainline test and all pinned hashes; fmt and clippy clean.

### Parameters (full geometry, measured)

| Part | Parameters |
|---|---:|
| root board encoder (input proj, square emb, 8 blocks, norm) | 26,396,880 |
| root candidate path (from/to, global, promo, facts, token norm, 1 candidate block) | 1,037,376 |
| root node projection + neutral WDL head | 166,019 |
| query-state encoder (256w, 2 blocks) incl. action head | 1,235,208 |
| planner (initial memory, event, workspace update, branch update) | 1,453,312 |
| selector edge scorer | 332,545 |
| selector STOP head (zero gradient while masked) | 68,097 |
| root readout | 164,353 |
| **total unique** | **30,853,790** |

Within the 27M-35M sanity range. The count is independent of the budget by construction and is
asserted unchanged across B0/2/4/8/16 runs.

### What the tests establish

- Finite policy at B0/2/4/8/16; forced budget spent exactly (`successful_queries == budget` for
  every example); unique nodes `= batch * (1 + budget)`; no duplicate edge; STOP never called.
- Root encoder executes exactly once at every budget, **measured by thread-local counters inside
  the model functions, independently of the driver's accountant**; query-encoder calls and planner
  updates equal the number of rounds; query-encoder and planner examples equal successful queries.
- B0 through `run()` is bit-identical to the `NeuralModel::forward_inputs` path, and performs no
  query, query-encoder or planner work.
- Traces are deterministic for ACTIVE, FIXED and seeded RANDOM; different seeds differ.
- `fixed_bfs_actionid_v1` is independent of the weights (two different weight draws give identical
  traces) and, as stated in the frozen spec, spends the whole budget on the first B root moves in
  ActionId order at depth 1.
- Terminal children contribute no frontier; an exhausted frontier is reported in the accounting,
  not hidden.
- Unsupported requests refuse visibly: STOP unmasked, budget above the supported range, a terminal
  root, an empty batch.
- **Gradient coverage on update one:** every parameter except the STOP head receives a finite,
  non-zero gradient; the STOP head receives exactly zero (it is masked). Three real AdamW updates
  reduce the loss.
- Strict identity: the 4x4 ordered-pair architecture refusal matrix (probe, candidate_v25,
  legacy_facts_v25, active_search_v3); every one of the 11 V3 contract ids, when changed, is refused
  by config validation, by model identity, and by checkpoint contracts; historical identities gain
  no `active` key.
- Save/load restores weights exactly; a run resumed from a checkpoint matches an uninterrupted run
  to `max|diff| < 1e-4` in log-probabilities; a V3 checkpoint refuses a `candidate_v25` template.
- CandidateFacts belong to the root path only (asserted over the parameter breakdown).

### Findings

1. **Inert key bias (fixed in V3 code).** The gradient-coverage test found `planner.k_proj.bias`
   with exactly zero gradient: a bias added to every key of a query cancels in the softmax (the same
   class of defect as D58's inert LF bias). The V3 planner's key projection now has no bias.
2. **Inherited inert key biases (NOT changed).** The V2.5 `Block` and `CandidateBlock` use a key
   bias with the same property. They are reused unchanged by the root encoder (contract
   `v25_root_encoder_v1` requires the V2.5 block) and by the query-state encoder. They receive
   gradient only at numerical-noise level, which is why the test does not flag them. Changing them
   would alter historical checkpoint layouts. Impact is about 256 + 640 parameters per block, which
   is negligible. Recorded so a later contract version can remove them deliberately.
3. The trait `forward_inputs` for V3 is explicitly budget 0; budgets above 0 exist only through
   `ActiveSearchModel::run`, because they need the live query tool.
4. Existing commands (`proof train`, `proof eval`, `v25-qual`) refuse `active_search_v3` with a
   visible error rather than running a wrong path.

### NOT RUN at P2
CUDA (P3), compute/VRAM measurement (P3), cached-versus-live packets (no cache exists yet), every
science phase.


## P2.1 - hardening after the P2 review  [MEASURED]

Same phase, corrections only; the architecture and thesis are unchanged. Decision V3-D9.

- **CLI boundaries.** `model-info` now has an exhaustive architecture dispatch and a real
  `active_search_v3` report; previously an active config fell through to the Probe report. `bench`
  and `cuda-smoke` refuse before any device work. `RunConfig::from_toml_str` refuses for the whole
  mainline runtime; `model_io::build_as` refuses before the device check; `ProbeModel::new`
  asserts. The integration test `active_boundary` runs the real binary against 18 historical
  commands with an active config and proves each refuses visibly with the architecture message
  before creating any output. `cuda-smoke` was checked by hand on the CUDA build (refused, GPU idle).
- **Frozen query heads.** `query_heads = 4`; the "measure 4 vs 8" wording is removed.
- **Budget ceilings.** `ACTIVE_MAX_BUDGET = 16` (science). B17 is refused; an explicit
  `engineering_stress` mode (ceiling 64) is marked `engineering_only`. `v3-qual` validates the
  budget lists before any work (previously `--budgets 0,32` ran for a minute and recorded failed
  rows while exiting 0).
- **Terminal roots.** Refused at the top of `run()`: checkmate, stalemate, threefold and fifty-move
  are tested, with zero CandidateFacts and zero neural work (in-function counters).
- **Honest accounting.** Query-encoder and planner rows are compacted to the examples that queried.
  A mixed batch (one example exhausting after 1 query, one spending 8) executes exactly 9 rows, not
  16. Selector padding is reported separately. `check_invariants` was strengthened and now runs
  before every successful `run()` returns; a tamper test rejects 12 distinct corruptions.
- **Depth consistency.** `Tree` refuses a root with `ply_from_root != 0` and a child whose
  `ply_from_root != parent depth + 1`.
- **CPU resume is bit-exact.** After restoring a checkpoint and two more optimizer steps, the
  maximum difference from the uninterrupted run over every parameter is exactly `0e0`. The loose
  `1e-4` tolerance in P2 was unnecessary and is replaced by exact equality (parameters and optimizer
  continuation).
- **Flaky gradient test fixed.** The V2.5 `Block`/`CandidateBlock` key biases have a mathematically
  zero gradient; whether float noise exceeded the test threshold varied with initialisation. They
  are now exempt by name and required to stay at noise level (< 1e-5). The suite passed 10
  consecutive full loops.

## P3 - CUDA and system qualification, real FP32 CUDA  [MEASURED]

Command: `recur64 v3-qual --config configs/v3/active-search-v3-cuda.toml --output runs/v3/cuda-qual
--positions 64 --budgets 0,2,4,8,16 --batches 1,8,16 --reps 5 --train-budgets 2,4,8 --train-batch 8
--train-updates 3 --lifecycle-reps 12 --sustained-seconds 10` (CUDA 12.9.1 user-space runtime, RTX
2000 Ada 16 GB, binary built with `--features cuda`). Evidence:
`docs/evidence/v3/v3-qual-cuda-fp32.json` and `docs/evidence/v3/model-info-active-search-v3.json`.
Every section runs `ActiveSearchModel::run` with the live `StateQuery` tool on 64 deterministic KQRvK
positions; nothing here is a science result and no HOLDOUT_C, TUNE or proof data is touched.

**Gate:** all sections passed (`all_sections_ok = true`). The device known-answer guard passed.

### What was exercised

B0/2/4/8/16 with ACTIVE selection and with the frozen FIXED schedule; the scripted (teacher-forced)
path through real training updates at B2/4/8; forward and backward finiteness; real AdamW updates;
gradient coverage on update one (234 parameter tensors; none without a finite non-zero gradient
except the masked STOP head, whose gradient is exactly zero); checkpoint save/load on CUDA (policy
outputs saved versus loaded: max abs difference 0.0 over 160 values); root-once and query/planner
execution counters measured inside the model functions and equal to the accountant at every budget;
forced budget accounting; dynamic queried-state legal widths; VRAM.

### Query latency curve (same checkpoint, steady state, means of 5 repetitions)

Times are completion times: the device is synchronised before every section clock read.

| batch | B | total, health on (ms) | total, health off (ms) | FIXED, health off (ms) | CPU exact queries | root GPU | query-state GPU | planner + selector | wall vs B0 | peak VRAM (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 10.9 | 9.0 | 9.0 | 0.0 | 9.7 | 0.0 | 1.1 | 1.00x | 591 |
| 1 | 2 | 27.8 | 27.2 | 30.0 | 0.1 | 8.5 | 5.8 | 13.2 | 2.54x | 591 |
| 1 | 4 | 42.3 | 42.4 | 37.8 | 0.1 | 8.3 | 10.5 | 23.3 | 3.87x | 591 |
| 1 | 8 | 74.8 | 70.2 | 66.5 | 0.2 | 8.0 | 20.7 | 45.8 | 6.84x | 591 |
| 1 | 16 | 175.8 | 165.2 | 167.9 | 0.4 | 12.3 | 45.0 | 117.8 | 16.07x | 591 |
| 8 | 0 | 30.1 | 30.7 | 28.3 | 0.0 | 27.8 | 0.0 | 1.9 | 1.00x | 655 |
| 8 | 2 | 48.9 | 47.1 | 44.2 | 0.2 | 22.0 | 8.0 | 18.3 | 1.62x | 655 |
| 8 | 4 | 65.6 | 48.2 | 48.0 | 0.4 | 19.1 | 15.0 | 30.9 | 2.18x | 655 |
| 8 | 8 | 83.8 | 81.5 | 92.2 | 0.8 | 13.2 | 21.4 | 48.0 | 2.78x | 655 |
| 8 | 16 | 167.7 | 177.5 | 170.8 | 1.6 | 13.4 | 45.9 | 106.4 | 5.57x | 655 |
| 16 | 0 | 22.7 | 20.8 | 19.0 | 0.0 | 20.6 | 0.0 | 1.5 | 1.00x | 655 |
| 16 | 2 | 35.3 | 32.3 | 45.6 | 0.4 | 15.3 | 5.7 | 13.3 | 1.56x | 655 |
| 16 | 4 | 53.3 | 51.9 | 50.2 | 0.8 | 15.5 | 11.3 | 25.2 | 2.35x | 655 |
| 16 | 8 | 87.8 | 84.5 | 84.2 | 1.1 | 14.7 | 22.7 | 48.7 | 3.87x | 655 |
| 16 | 16 | 176.6 | 168.7 | 167.8 | 2.7 | 16.8 | 46.3 | 110.2 | 7.79x | 911 |

Root CandidateFacts CPU wall (baseline work, not budget): 11 us at batch 1, 89 us at batch 8, 147 us
at batch 16, reported separately in the JSON. Cold first calls (JIT/autotune), separated from the
steady state above: 1.52 s for the very first call, then 0.06 to 1.12 s for each new (batch, budget)
shape. Only 5 repetitions per cell: the B0 reference itself varies between batches (30 ms at batch 8,
23 ms at batch 16), so the "wall vs B0" ratios are indicative, not precise.

### Reading the curve

- **The exact query is not the cost.** CPU exact queries are 0.4 ms of 176 ms at batch 1 / B16 and
  2.7 ms of 177 ms at batch 16 / B16 (at most 1.5% of wall). Sustained B8 at batch 16: 2.4% of wall.
- **The cost is many small GPU steps.** Planner + selector is about 62 to 67% of the wall at B16 and costs
  roughly 6 to 7 ms per round at every batch size (46 to 49 ms at B8 for batch 1, 8 and 16 alike): the
  work is launch-bound, not compute-bound, so per-round time does not grow with the batch. Batching
  positions is therefore the throughput lever: B16 costs about 168 to 177 ms per call whether the
  batch is 1 or 16, that is about 176 ms per position at batch 1 and about 11 ms per position at
  batch 16.
- **B16 is not "16x compute".** Measured wall of B16 over B0 is 16.1x at batch 1 but 5.6x at batch 8
  and 7.8x at batch 16, and the B0 cost is the heavy root encoder while each query costs far less.
  Wall, rows executed and VRAM are reported; no compute multiplier is claimed.
- **Health checks** (finite and RMS guards read back to the host each round) are noise-level in most
  cells and up to about 35% in a few; the no-health columns are reported so they are separable.
- **FIXED versus ACTIVE** cost about the same (same heavy path); the selector forward still runs for
  FIXED.

### GPU utilization

Sustained B8, batch 16, 10 s back-to-back: 224 positions per second; `nvidia-smi` utilization
(500 ms samples, any-kernel-running fraction) averaged 39% over busy samples with a maximum of 47%.
So low and sporadic utilization is **structural** for this workload at these batch sizes: many small
kernels and one host read per round because the next exact query is a CPU operation. It is not a
fault. Larger batches raise useful work per launch; any fusion, graph capture or shape bucketing is a
separate execution-only experiment with a parity proof and was not attempted.

### Dynamic queried-state widths

48 distinct positions at batch 1, B8, run twice: 47 distinct width sequences and 27 distinct padded
action widths. The first pass mean is 1.22x the second-pass mean; the worst first-pass call took
0.50 s against a second-pass median of 0.088 s (5.6x). So first-seen shapes cost a one-off JIT or
autotune stall of up to about 0.4 s, bounded and not repeated, and steady state is unaffected. No
bucketing was introduced: this is reported, not optimized.

### Training path (teacher-forced engineering schedule, batch 8, real AdamW updates)

| budget | sec / update | peak VRAM (MB) | gradient coverage |
|---|---|---|---|
| B2 | 0.22 | 1457 | 234 tensors, all covered except masked STOP |
| B4 | 0.18 | 1585 | 234 tensors, all covered except masked STOP |
| B8 | 0.27 | 1649 | 234 tensors, all covered except masked STOP |

### VRAM

- **One resident model, many calls (what a science run does): plateau.** 150 consecutive B8
  inference calls at batch 16: VRAM flat at 2001 MB across all 16 samples. 40 consecutive B4 training
  updates at batch 8: flat at 1489 MB after the first page.
- **Repeated model build/drop: about 13 to 16 MB of growth per cycle, reproduced on V2.5.** With no
  forward pass at all (`build_then_drop`), with B0, with B8, with and without an explicit
  `memory_cleanup`, VRAM climbs about 13 to 16 MB per build/drop cycle (one allocator page of 32 MB
  every two cycles). An explicit `memory_cleanup` releases a large one-off chunk (about 700 MB) but
  not this slope. The control, the historical `CandidateV25Model` built and dropped under the identical
  loop and the same cleanup, climbs at 13 to 16 MB per cycle too, so this is behaviour of the pinned
  Burn/CubeCL stack on this machine, not something V3 introduced, and it does not occur inside a run
  that builds its model once. It was investigated to that point and not fixed.

### Not claimed

Nothing here validates science, learned selection, conversion or extrapolation. BF16 and TF32 were
not run (V3.0 is FP32). CUDA bit-exactness is not required or claimed (the checkpoint comparison
happened to be exactly 0.0 on this device).

## P3 report: findings and suggestions bearing on the scientific contract

None of these changes a frozen rule. They are recorded so the owner can decide before P4.

1. **No change needed to the architecture or the query heads.** Four heads showed no correctness
   failure or pathology; the head count stays frozen.
2. **Live queries are cheap; caching StatePackets is not needed for speed.** The exact CPU query is
   at most 2.4% of end-to-end wall. The optional training-time packet cache (spec section 28) would
   save little; suggest building it only if P4 measures a reason, which keeps one fewer versioned
   contract.
3. **Compute reporting.** Because B16/B0 wall ratios range from 5.6x to 16.1x depending on batch, and
   B0 is the heavy root encoder, Gate IV and the external-search comparison (control G) should report
   measured wall, rows executed and VRAM, and must not describe budgets as compute multiples. This is
   already the rule; the measurements show why it matters.
4. **Throughput plan for P5 to P7.** Planner and selector are launch-bound at about 6 to 7 ms per
   round independent of batch. Training and evaluation should use the largest batch that fits, and the
   ACTIVE versus FIXED comparison should use identical batch shapes. Fusion, graph capture or width
   bucketing are optional execution-only experiments needing a parity proof; none is needed for
   correctness.
5. **Gate VI "stable lifecycle/VRAM" is best read as the resident-model plateau**, which passed. The
   rebuild slope is a pre-existing stack behaviour. A process that evaluates many checkpoints in turn
   (for example P8) should build one model per process or accept about 16 MB per build.
6. **Inherited inert key biases** (about 640 per V2.5 board block, 256 per candidate or query block)
   are kept for contract `v25_root_encoder_v1`. A later contract version could remove them
   deliberately; the impact is negligible.
7. **Input validity.** `GameState::from_fen` accepts adjacent kings and an attacked opponent king,
   after which `candidate_facts` panics. It is unreachable from legal play or solver-verified data;
   V3 TUNE generation must validate its positions (the P4 audit does so through the independent solver
   path). No core change was made.
8. **Open for P4, unchanged:** the ideal-certificate budget coverage `C_8(KQRvK M3)` and the frozen
   feasibility rule. Nothing measured in P3 bears on that number.
