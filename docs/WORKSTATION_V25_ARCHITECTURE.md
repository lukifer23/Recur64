# Workstation V2.5 — Candidate Transformer (`candidate_v25`)

Branch `experiment/workstation-v25`, base main `fef1ffcf9c38381d4adc671e5e2c5ead9f141e33`
(safety tag `main-pre-workstation-v25-fef1ffc`). Main stays the conventional control line;
the HP branch stays the separate learned-planner line. Nothing is merged from HP; only
generic semantics are re-implemented, and every ported semantic is listed here.

Question: can a substantially stronger ONE-PASS model learn conversion technique when
legal moves are first-class tokens, exact per-candidate facts are inputs, and exact
near-mate policy supervision precedes self-play? No recurrence, planner, world model,
visual path, retrieval, external engine, material reward or contempt.

## Identity contract
- `ModelConfig.architecture` ∈ {`probe_v1` (default), `candidate_v25`}. The default and
  the absent `candidate` geometry are never serialized, so historical scientific hashes
  and `check_model` values are unchanged (guarded by the frozen P4.5 hash test and
  `probe_identity_is_unchanged_by_the_architecture_field`).
- Checkpoint metadata records architecture id, head version, CandidateFacts version,
  candidate-token contract, candidate-block contract and full geometry. Loads across
  architectures are refused explicitly by id, not by tensor shape, in both directions.
- `candidate_v25` requires recurrence 1 and refuses anything else.
- C0 (facts ablated) and CF are distinct identities (`candidate.facts_enabled`), with the
  same parameter count.

## Geometry (primary)
Board: Linear(119→640) + learned square embeddings, 8 unique geometry-aware blocks
(10 heads × 64, FFN 1280), final RMSNorm. Candidate: dim 256, 4 heads, FFN 512, 1 block,
facts encoder 8→64→256 (nonzero init), policy scorer 256→128→1 (small nonzero init),
WDL = zero-initialised Linear(mean(B)→3). Target 26–29M parameters; the exact count and
breakdown are reported from the implementation in `WORKSTATION_V25_BUILD_RESULTS.md`.

## CandidateFactsV1 (eight fields, legal-action order, all in 0..1)
0 mate · 1 check · 2 capture · 3 captured_value/9 · 4 attacked_after · 5 promotion ·
6 promotion_gain/8 · 7 stalemate. `attacked_after` = the opponent has a LEGAL reply
landing on the destination square. Semantics follow the HP branch's
`CandidateFactsV1` (read-only inspection of `origin/experiment/hp-r15-h3-integration`
@ `79ffd14`); the implementation here is independent and computed from the authoritative
`GameState`, never from a lossy observation.

## Not in V2.5
Recurrence, latent scratchpads, Chimera planner, WorldModelV2, WASM world model, visual
CNN, diffusion, retrieval, Stockfish/Leela/Syzygy, opening books, human PGNs, material
reward, contempt, handwritten evaluation, dense 20,480-way policy head, TF32, fusion,
autotune, candidate buckets by default. Future only: deeper mate bands, trajectory
distillation, DTM head, successor features, planner integration.


---

## Addendum: legacy_facts_v25 (LF) and the final architecture inventory
| id | params | policy | facts | contract |
|---|---:|---|---|---|
| probe_v1 (L geometry 640/10/1280/8) | 26,809,944 | legacy head v2 (source/dest/promotion) | none | head v2 |
| candidate_v25 C0 | 27,469,204 | candidate tokens + 1 candidate block | zeroed input | token/block 1 |
| candidate_v25 CF | 27,469,204 | candidate tokens + 1 candidate block | token encoder 8->64->256 | token/block 1, facts 1 |
| legacy_facts_v25 LF | 26,810,584 | legacy head v2 + fact delta | `Linear(64->1, no bias)(GELU(Linear(8->64)))` added to the logit | fact-delta 2 |
Historical F10 (384/12/768/8) is 9,805,672 parameters. LF wraps the unmodified `ProbeModel`; its
fact-delta final layer has no bias because a per-row constant is cancelled by the softmax (contract 1,
which had one, was an engineering-only dead end). Every architecture is one pass, recurrence 1, FP32.
