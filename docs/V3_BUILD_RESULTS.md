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
