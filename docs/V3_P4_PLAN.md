# Recur64 V3 - P4 plan (data, ProofTraceV1, frozen feasibility measurement)

Status: PRE-REGISTERED. This file and the code it describes are committed **before** any
`ProofTraceV1` is generated for the P25_DATA_V1 TRAIN split and before `C_8` is computed. Results
are recorded afterwards in `V3_EXPERIMENTS.md`, `docs/evidence/v3/` and `V3_P4_RESULTS.md`; this file
is not edited after the measurement.

P4 answers four questions and nothing else:

1. Can we build an exact, set-valued training process target for selective search?
2. How many exact state queries does an ideal adversarial proof certificate need?
3. Does enough of the frozen primary stress cell fit inside B8 for the planned experiment to be
   interpretable?
4. Can we build a clean V3-specific TUNE set without contaminating HOLDOUT_C?

It does **not** answer whether a neural model learns search. There is no model training and no
model evaluation in P4. P5 is not authorized by this phase.

## Frozen inputs (unchanged by P4)

`active_search_v3` and its 11 contracts, root-only CandidateFacts, one exact edge = one query, B16
absent from training, STOP masked, `fixed_bfs_actionid_v1`, primary stress cell `KQRvK M3`, Gate II
(`ACTIVE_B8 - B0 >= +0.10`), Gate III (`ACTIVE_B8 - FIXED_B8 >= +0.05`), the feasibility threshold
`C_8(KQRvK M3) >= 0.25`, FP32. The meaning of "frozen" is the single definition in
`V3_RESEARCH_PLAN.md`.

## Facts read during custody, before any trace exists

Custody is read-only (V3-E9). Read from the files, not hard-coded:

- P25_DATA_V1 TRAIN: 44,332 positions, content digest
  `3b25dc8549dd2fc9d47c30e294c273b3306aecb3eba91b964715326ddf74f2e6`, every position passes the
  existing independent label audit, 44,332 unique exact FENs and canonical classes, zero overlap with
  HOLDOUT_C.
- Its cells (family, mate depth, n): KQQvK M1 5000, M2 5000, M3 1831; KQRvK M1 5000, M2 5000,
  **M3 5000**; KRRvK M1 5000, M2 5000, M3 5000; KQvK M1 246, M2 462, M3 862; KRvK M1 153, M2 426,
  M3 352. There are no M4 or M5 positions in the split.
- HOLDOUT_C's content digest equals the frozen value
  `4ab951c6edd8dd4f531bb87d2f4373895d1fddb70efdf052b24c09a1b71d87d5`; it has not been evaluated.

## ProofTraceV1 (contract `proof_trace_v1`)

Training-side only. It lives in `recur64-runtime::proof::trace*`, depends on the exact mate solver, and is
imported by neither the query tool nor the model. `StatePacketV1` is unchanged; a test pins its 15
fields and rejects any field name suggestive of proof information.

### Proof graph

For a position of minimal mate depth `D` (attacker moves; the solver's definition, unchanged):

- **OR node** `(board, n)`: an attacker decision with `n` attacker moves left including the one
  played. Its alternatives are every legal attacker move that forces mate within `n`: an immediate
  checkmate (a leaf, cost 1) or a move after which the defender is lost within `n - 1`
  (cost `1 + AND`).
- **AND node** `(board, k)`: a defender decision, `k` attacker moves left. Its alternatives are
  **every** legal defender reply, each leading to the OR node `(next, k)`.
- A defender reply that stalemates or leaves a dead position cannot occur inside an AND node,
  because such a position is not a forced loss; a stalemating attacker move is simply not a winning
  alternative.

### Cost

`OR(b,n) = min over winning m of [1 + (0 if m mates else AND(after, n-1))]`,
`AND(a,k) = sum over every reply r of [1 + OR(next, k)]`,
`Q*(p) = OR(root, D)`.

`Q*` counts exact query edges of the cheapest complete certificate of a correct root move. It is
computed exactly by this recursion over the stored graph. No heuristic, beam, PUCT, network score,
principal variation, average reply count or sampled reply is involved. Refutation records do not
enter `Q*`.

### Incorrect root moves

For each incorrect, non-terminal root move, the set of defender replies after which the attacker
cannot force mate within `D - 1` (all replies when `D = 1`) is stored as a set. It enters the
selector target (`A_refute`) and never `Q*`.

### Edge identity

V3 does not merge transpositions, so a trace edge is identified by its **root-relative action path**
(the canonical ActionId of each ply). The stored graph shares `(board, n)` nodes internally for
compactness, but admissible edges are always reported as paths. Transient `NodeId`s are never part of
a trace. A test finds a node reached by two different paths and checks that the two remain distinct
edges.

### Target `A(S)`

For a queried-edge set `S` (a set of paths): `r(T,S) = |T \ S|` over certificates `T`;
`T*(S)` are the certificates minimising it (all ties kept); `A_proof(S)` are the frontier edges lying
in at least one of them; `A_refute(S)` are the unqueried refuting replies of every queried incorrect
root move; `A(S) = A_proof(S) union A_refute(S)`. It is computed by dynamic programming over residual
costs rather than by enumerating certificates, preserving every tie, and it is a function of the SET
`S` only. A complete proof contributes no `A_proof` edges. The selector target is uniform over `A(S)`;
no efficiency weighting exists in V1. Tests compare the DP with a brute-force enumeration of every
certificate on hundreds of random queried sets, check order-invariance through the live
`QueryManager` and `Tree`, and check that following admissible edges completes a proof in exactly
`Q*` queries whichever admissible edge is chosen.

### Independent audit

`audit_trace` re-derives every claim through `GameState::apply`, `legal_actions()` and the existing
independent solver memo of the label audit, and calls nothing from the generator. It checks source
identity, the root legal list and correct set against `ProofTargets`, that each OR node's alternatives
are exactly the winning attacker moves (none missing, none extra, mate flags right), that each AND
node contains every legal reply, that a node shared between paths is the same `(position, n)`, the
refutation sets, recomputed costs and `Q*`, acyclicity and absence of orphans. Ten independent
corruptions are each rejected by a test. Every trace is audited.

### Sharding and determinism

Shard `i` is positions `[i * 500, (i + 1) * 500)` of the source order. A shard is a pure function of
its positions (contiguous chunks per thread, re-joined in order), so output does not depend on the
thread count; a command compares sample shards on 1 and N threads against the stored ones. A shard is
reused only after it is re-validated (schema, trace contract, source digest and split, index, range,
ids, content digest); a corrupted or mismatched shard is a hard error. Audit shards are resumable the
same way. The feasibility table is refused unless a complete, failure-free audit covers every shard of
the same trace manifest.

## `v3_tune_v1` (frozen before generation)

| field | value |
|---|---|
| identity | `v3_tune_v1` |
| purpose | V3 P5/P6 model selection and information-sufficiency work only |
| families | KQRvK, KRRvK |
| depths | M1, M2, M3 |
| target | 750 unique positions per cell, 6 cells, **4,500** positions |
| seed | `0x7A130004` (2048065540) |
| history | `fresh_no_history_v1` |
| filters | exactly those of every proof set: legal, live, pawnless, fraction `<= 0.15` of legal moves correct and CandidateFacts ambiguity for depth `>= 2`; never relaxed |
| selection | eligible canonical classes from the exact pool, minus exclusions, seeded shuffle with tag `mix(0x3A710000 + family_index*16 + depth)`, first 750 |
| ids | `v3tune-...` |

If any cell cannot supply 750 eligible, disjoint positions, generation stops before accepting a
partial set and the pool accounting is reported to the owner; 750 is never lowered.

Exclusion: hard-disjoint, by exact FEN and symmetry-canonical class, from every `proof-*.json` dataset
on the workstation: `runs/v25/proof`, `runs/v25/proof-v2`, `runs/v25/p25/data` and
`runs/v25/p25/holdouts` (P25_DATA_V1 TRAIN, every replacement and retired TRAIN/TUNE/CONFIRM,
HOLDOUT_A, HOLDOUT_B, HOLDOUT_C). The manifest records each dataset's path, split, positions, content
digest and contributed canonical classes, and a deterministic exclusion-manifest digest. HP, X1 and X2
exact datasets are **not** available locally in a compatible form, so disjointness from them is
**not** verified and is stated as a limitation, not claimed.

Audit and verification before acceptance: every position through the independent label audit; no
duplicate FEN or canonical class inside the set; no overlap with any inventoried dataset (checked
directly and pairwise); the saved file reloads with the same digest; and an independent regeneration
from the seed reproduces the digest.

## HOLDOUT_C custody

Allowed: `ProofTargets::load` (schema, contracts, digest), comparison with the frozen digest,
canonical-class and exact-FEN exclusion, and writing the seal. Integrity checking is custody, not
evaluation, and is **not** logged as an exposure. Not allowed in P4: any model evaluation, any
ProofTrace, any feasibility number, any top-1, any per-cell or per-family performance analysis, any
selector target, training, threshold inspection, or printing positions or labels.

The seal (`v3_confirm_seal_v1`) records the expected and verified digest, the path, the split, the
position count, `verified`, `sealed` and `evaluated = false`. Guards, all tested: the tracer refuses
every non-TRAIN/TUNE split and the frozen digest; the working-split loader refuses `holdout_c` and its
digest; and the only way to obtain the set for evaluation is `load_sealed_confirmation`, which needs a
`ConfirmAuthorization` that can only be created for phase `V3-P8` with a non-empty owner approval
reference, refuses a seal that records a prior evaluation, and appends an exposure line before
returning data. A digest mismatch is a hard stop; the set is never repaired or regenerated.

## The feasibility measurement (frozen procedure)

Data: the **P25_DATA_V1 TRAIN split only**, every position, no sampling. The tool refuses the primary
role on any other split. Order of operations: generate all TRAIN traces, audit all of them, then
compute the table; the table is refused if the audit is incomplete or has any failure.

Per (family, mate depth) cell: `n`, `Q*` minimum, median, p90, p95, maximum (nearest-rank
quantiles), counts and fractions `C_k` for `k` in 2, 4, 8, 16 with `C_k = (# positions with
Q* <= k) / n`; pooled per depth and overall; the unweighted macro mean of per-cell `C_k`. Values are
machine-readable and unrounded.

Primary quantity: `C_8(KQRvK M3)`. The rule is evaluated in exact integer arithmetic,
`count_le_8 * 4 >= n`, so no float rounding can flip it.

Classification (exact text):

- `C_8(KQRvK M3) >= 0.25`: **SCIENTIFICALLY QUALIFIED FOR THE B8 PRIMARY EXPERIMENT**. This means only
  that B8 can contain complete ideal certificates often enough for the later neural experiment to be
  interpretable. It does not mean the model learns them, that B8 beats B0, that ACTIVE beats FIXED, or
  that V3 succeeds.
- otherwise: **NOT SCIENTIFICALLY QUALIFIED / BUDGET MIS-SPECIFIED**. Then P5 to P8 are not run, Gates II
  and III are not lowered or re-read, HOLDOUT_C stays sealed, nothing is trained through B16, and any
  longer budget is a new experiment identity with a new preregistration. This is not evidence that
  active search is false; it means this budgeted experiment cannot cleanly test it.

After the measurement P4 stops and reports to the owner. No experiment parameter is changed after
seeing the result.

`v3_tune_v1` gets its own audited traces and `C_k` diagnostics afterwards. They are recorded
separately, labelled diagnostic, and never enter the rule.

## Runtime and storage rules

No job may exceed two hours wall-clock without owner approval; trace generation and audit are sharded
and resumable for that reason. Traces and datasets are written to git-ignored run storage; only
compact manifests, digests and summaries are committed. No StatePacket cache is introduced: P3 showed
the live exact query is at most 2.4% of end-to-end wall.

## Out of scope for P4

Model training or evaluation, P5 to P11, RL, adaptive STOP, self-play, transposition merging, extra
CandidateFacts, any answer-adjacent query output, any change to the neural execution path.
