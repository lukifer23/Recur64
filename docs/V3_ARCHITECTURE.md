# Recur64 V3 — Architecture (`active_search_v3`)

Status: P0 frozen specification. Dimensions marked *(measure)* are targets; the measured values
are recorded in `V3_BUILD_RESULTS.md` once built.

## 1. Question the architecture answers

Does the SAME weight set improve as its exact state-query budget grows (B0→2→4→8→16), because it
learns which unresolved future states to inspect and integrates what comes back?

Not claimed: recurrence is useful; more FLOPs help; exact facts solve chess.

## 2. Data flow

```
root GameState
  ├─ ObservationV1 ─► RootEncoder (V2.5 CF board, 640w/10h/8 blocks) ── runs EXACTLY ONCE
  ├─ CandidateFactsV1 (root only; accounted separately, free of query budget)
  └─► RootCandidateTokens M_i (256d, one candidate block)        [i = root legal moves]
                      │
   B0 ────────────────┴──────────────► RootPolicy (sparse legal softmax)
                      │
   B>0:  repeat B times (forced budget; STOP masked)
        frontier F = legal, not-yet-queried edges of all discovered non-terminal nodes
        Selector(F | workspace, branch memory, parent/edge emb, depth, parity, remaining) ─► edge e
        StateQuery(parent(e), action(e)) ─► StatePacketV1           [CPU, exact, 1 unit]
        QueryStateEncoder(packet.obs)  ─► square feats, pooled node, raw-action embs
        Planner(Z, Branch[root(e)], e, node, M_root(e), depth/parity) ─► Z', Branch'
                      │
                      └──────────────► RootPolicy reads M_i, Branch[i], Z
```

## 3. Identities (every one enters the scientific identity hash and the checkpoint)

| Component | Identity |
|---|---|
| architecture | `active_search_v3` |
| root encoder | `v25_root_encoder_v1` |
| root candidate tokens | `candidate_token_v3_root_v1` |
| query tool | `state_query_v1` |
| queried-state encoder | `query_state_encoder_v1` |
| frontier | `frontier_v1` |
| search memory | `branch_workspace_v1` |
| selector | `active_selector_v1` |
| planner/update | `active_planner_v1` |
| query trace (training only) | `proof_trace_v1` |
| budget training | `budget_0_2_4_8_v1` |
| root policy readout | `root_policy_v3_v1` |

Checkpoints refuse cross-loading against `probe_v1`, `candidate_v25`, `legacy_facts_v25` and any
Chimera architecture, and refuse any V3 contract/config mismatch (visible error, no fallback).
Config fields added to `ModelConfig` are skipped when absent so historical hashes are unchanged.

## 4. Root encoder and candidate tokens

V2.5 CF geometry, unchanged. The board encoder executes once per decision at every budget
(asserted by a counter). Candidate tokens are first-class (from/to square features, pooled board
context, promotion, CandidateFactsV1 through the fact MLP with ordinary non-zero initialization).
No `logit += gain * facts`.

### Query accounting rule
CandidateFacts are a root-only common baseline feature. They are **never** computed for any
queried descendant: they would apply every child move and silently expand states outside the
budget. Root CandidateFacts time/work is reported separately from query budget.

## 5. StateQueryV1 (crate `recur64-statequery`)

`query(parent_node, legal_action) -> StatePacketV1`. One successful call = exactly one legal edge
transitioned = exactly one budget unit. Legal-move generation at the returned child is part of the
call's cost.

Dependency boundary: depends only on `recur64-core` (+ serde, sha2). It cannot import
`recur64-runtime` (solver), `recur64-model` or `recur64-search`. A test and the qualification
script check `cargo tree`.

The manager keeps an authoritative `GameState` per node (full history), so castling, en passant,
repetition and fifty-move are exact.

### StatePacketV1 fields (whitelist; a test freezes this set)
`node_id`, `parent_id`, `incoming_action`, `ply_from_root`, `observation` (ObservationV1 of the
child), `legal_actions` (complete, never truncated), `side_to_move`, `in_check`, `terminal`,
`terminal_reason`, continuation fields (`castling`, `ep_square`, `halfmove_clock`,
`repetition_count`), `semantic_id` (semantic state identity, below).

`StatePacketV1::content_digest()` (a method, not a field) is a SHA-256 over the complete packet
content including the observation and `semantic_id`. It proves that cached and live packets are
field-equivalent; it is not the identity of the state.

### Semantic state identity (`STATE_IDENTITY_VERSION` 1)
`semantic_id` is SHA-256 over everything Rules Profile V1 needs to decide future legality and
termination, so it never merges states whose futures can differ:

- side to move, castling rights, en-passant square (the position key: FEN without clocks);
- the halfmove clock (fifty-move rule);
- the repetition history: the multiset of position keys of every earlier position that can still
  recur, i.e. the last `halfmove_clock + 1` positions clipped to the recorded history (positions
  before the last irreversible move cannot recur), because a later position's threefold count
  depends on it;
- the administrative ply cap and, only when a cap is set, the ply count.

Excluded: the fullmove number, the move order that led to the state, node ids, and anything the
model observes but the rules ignore (the 8-frame observation window). Board-placement-only hashing
is never used for node or transposition identity. The identity is conservative: some states with
the same placement are treated as distinct. V3.0 records transpositions by equal `semantic_id` but
does not merge them. Tested by fixtures for castling, en passant, halfmove clock, side to move,
repetition history, reversible-history differences, irreversible-move reset and the ply cap.

### Prohibited in any tool output
forced-win flag, mate-in-N, DTM/DTZ, tablebase value, proof status, solver result, best move,
PUCT/MCTS visits, neural value, downstream mate count, downstream checking-move count,
"number of winning replies", continuation-quality score, any aggregate tactical summary, anything
derived from the exact mate solver.

### Refusals (visible errors)
illegal action; parent terminal; duplicate edge; unknown node; legal list > 256; depth beyond the
declared range.

## 6. QueryStateEncoder (`query_state_encoder_v1`)

Input: the child `ObservationV1`. Width 256, 2 blocks, FFN 512 (heads 4 — measure 4 vs 8 at
qualification), parameters shared across every queried node and query step. Outputs: square
features; one pooled node vector; raw legal-action embeddings from (from-square feature,
to-square feature, promotion embedding) — built from the node's own squares only, never from a
successor. Parameter count is independent of budget.

Ceiling: total model ≈ low 30M, ≤ ~35M unique parameters unless measured evidence forces change.
V2.5 CF is 27.47M; the active components add the query encoder, planner, selector and readout.

## 7. Search memory (`frontier_v1`, `branch_workspace_v1`)

Explicit records outside the neural latent, per discovered node: node id, parent id, incoming
ActionId, root candidate/branch id, ply depth, root-relative turn parity, state embedding,
check/terminal flags, frontier membership. Neural memory: one branch token per **root candidate**
(256d) and a global workspace of K = 8 tokens (256d). All descendants of root move i update
Branch[i]. No hand-written minimax backup exists in the learned model (classical search is a
separate control).

The tree is a tree, not a graph: transpositions are recorded if detected but not merged.

## 8. Planner (`active_planner_v1`)

One shared-weight gated update per successful query:
`(Z, Branch[r]) ← Planner(Z, Branch[r], edge, node, M_r, depth, parity)`; state update
`S' = RMSNorm((1-u)·S + u·proposal)`, `u = sigmoid(…)`. No "think again" step on identical
information in V3.0. No absolute query-step embedding (extrapolation to B16): query-count and
remaining-budget enter as normalized / Fourier features. Depth embedding has an explicit safe
range (error, not clipping).

Health guards (error): non-finite workspace/branch/gate; workspace RMS above a configured
ceiling.

Per-query diagnostics: workspace RMS/delta, branch-memory RMS/delta, gate mean/std, root branch
selected, depth selected, frontier size, terminal discovery, selector entropy, selector margin,
remaining budget.

## 9. Selector (`active_selector_v1`)

Scores every frontier edge and additionally a STOP logit. The primary comparator schedule
`fixed_bfs_actionid_v1` (frozen in `V3_RESEARCH_PLAN.md`) is model-independent and is not part of
the learned model. **Primary experiment masks STOP and
forces the full budget.** An edge is never queried twice. Selection is teacher-forced from proof
traces in V3.0 (one pre-registered DAgger-style rescue allowed; no RL).

## 10. Root policy (`root_policy_v3_v1`)

Sparse softmax over root legal candidates; reads `M_i`, `Branch[i]`, workspace context (and
optionally a pooled summary of i's discovered descendants). The same weights at B0/2/4/8/16.
B0 = same checkpoint, zero queries, zero query-state encoder runs, zero planner updates. WDL head
stays neutral/compatible; V3.0 is policy-primary.

## 11. Budget semantics and accounting

B0 = 0 queries; B2/B4/B8/B16 = exactly that many successful queries in forced mode.
Reported per inference: requested budget, successful queries, STOP calls, transitions, legal moves
generated, query depths, unique nodes, terminals, root encoder runs, query-encoder runs, planner
updates, CPU query wall, GPU root wall, GPU query-state wall, planner/selector wall, end-to-end wall,
peak VRAM, FLOP estimates only where reliable. No "16x compute" claims without measurement.
All latency/compute-frontier numbers use LIVE StateQuery; cached training packets are digest- and
version-checked and must be field-equivalent to live packets.

## 12. Parameter breakdown *(measure)*

| Part | Expected |
|---|---|
| Root encoder + candidate path + heads (V2.5 CF) | ~27.5M |
| QueryStateEncoder (256w, 2 blocks) | ~2M |
| Planner + branch/workspace init | ~1–3M |
| Selector + readout | <1M |

## 13. Precision

FP32 for all V3 science. No silent precision change. No CPU/FP32 substitution when CUDA is
requested.

## 14. Out of scope for V3.0

Visual CNN, diffusion, retrieval, WASM, recurrent root transformer, capacity sweeps, Gumbel,
world model, tablebase, engine distillation, self-play, RL selector, transposition merging,
learned STOP as a primary confound, process-head proliferation, dense 20,480-way policy.
