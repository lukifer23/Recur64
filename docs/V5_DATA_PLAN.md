# V5 HP-native exact data plan

PRE-REGISTERED before native generation. Owner amendment 2026-10-03.
P25_DATA_V1 will not be transferred and is removed as a V5 prerequisite.
This is a new lineage, not recreation of P25. No V5 learning result exists.
No historical P25 digest/count is the scientific identity of the new run.
Legacy loaders remain locked pending measured DATA-B and recipe-v2 integration.

Family: `v5_hp_exact_endgames_v1`.
Selection: `v5_hp_dataset_select_v1`. Audit: `v5_hp_independent_audit_v1`.
Manifest: `v5_hp_dataset_manifest_v1`. Future data contract: `v5_hp_data_v1`.
Future recipe: `v5_stage_recipe_v2`; adopted only after measured artifacts.

| Identity | Seed | Families | Depths | Per cell | Total |
|---|---|---|---|---:|---:|
| V5_HP_TRAIN_V1 | 0x7A50_1001 | KQQvK,KQRvK,KRRvK,KQvK,KRvK | M1,M2,M3 | 2000 | 30000 |
| V5_HP_DEV_V1 | 0x7A50_1002 | KQRvK,KRRvK | M1,M2,M3 | 750 | 4500 |
| V5_HP_CONFIRM_V1 | 0x7A50_1003 | KQRvK,KRRvK | M1,M2,M3 | 750 | 4500 |

Generation order TRAIN, then DEV excluding TRAIN, then CONFIRM excluding both.
Both exact-FEN and authoritative `proof::generator::canonical_key` identities
are excluded. Canonical representative uses existing `fen_from_canon`.
Exact labels reuse `MateSolver`, `legal_cozy_moves`, and `ProofTargetsV1`;
independent labels reuse `audit_position` through GameState/apply/termination.
Fresh white-to-move pawnless roots with only a black king, no history,
no castling, zero halfmove clock, nonterminal and complete sorted legal actions.
Existing depth>=2 fraction<=0.15 and CandidateFacts ambiguity filters retained.
No external engine, learned labels or second chess equivalence implementation.

Deterministic selection freezes the exhaustive eligible canonical pool per family
via existing `enumerate_pool`, independent of thread scheduling. Before any split
selection, measure the full KRvK M1 capacity as an early mandatory feasibility
check. This uses max depth 1 and counts every legal/live canonical placement,
without exclusions, giving the largest possible supply for that cell.
If fewer than 2000 classes exist, STOP and retain the exact census; no partial
TRAIN, lower count, duplicate symmetry or replacement family is accepted.
This capacity check generates no dataset and evaluates no confirmation set.

Sort each eligible cell by SHA256(selection-contract bytes, zero separator,
seed little-endian u64, canonical identity bytes), canonical tie-break; take N.
Sorted scientific records order by family, depth, canonical identity. Use IDs
bound to split role and canonical identity. No arrival-order or first-N sampling.
Exhaustive pools can be persisted in deterministic first-king-square ordinal
shards 0..64, merged in canonical order; resumptions must verify source/config
and shard content. No process may exceed two hours. Long enumeration must use
bounded deterministic shards rather than relax counts or time constraints.

Audit every accepted record independently, including canonical/material identity,
legal list, correct indices, minimal mate depth, clocks/history and terminal
status. One unresolved failure invalidates the entire dataset. Regenerate at
least one full split from the same contract before acceptance.

Manifests bind source SHA, seeds/contracts, cell/counts, generation code/config,
canonical sorted record content digest, exact-FEN/canonical-set digests, exclusions
and independent audit results. Hashes exclude machine clock, local paths, usernames, hostnames and operational
times. Content includes authoritative scientific FEN clocks.
Raw artifacts are ignored at:
`runs/v5/data/v5-hp-train-v1.json`, `v5-hp-dev-v1.json`, `v5-hp-confirm-v1.json`.
Compact manifests, audits and seal go under `docs/evidence/v5/data/`.
Custody must read local bytes, never infer existence from a manifest.

Inventory locally available historical raw proof data and exclude identities
where practical. Cross-disjointness from unavailable workstation-only raw
datasets was NOT verified. Those sets are not used as V5 evaluation artifacts.
V4_TUNE_V1 and HOLDOUT_C remain historical UNEVALUATED and are not prerequisites.

After complete measured generation/audit/disjointness, seal CONFIRM with
`evaluated=false`. Ordinary training and DEV commands must refuse CONFIRM;
no authorization to evaluate it exists here. DATA-A is committed/pushed before
capacity/generation; DATA-B measured identities is committed/pushed before
scientific integration. No guessed digest is bound into code.

Future integration trains both stages on all 30000 TRAIN positions, all 15 cells;
DEV is independent 4500, six equal 750 cells, primary KQRvK M3 n=750.
The 24-position drill is four stable-ID/hash TRAIN positions per heavy-family
M1/M2/M3 cell, Q8/R4, reader only, <=200 updates, peak LR 1e-3, unchanged loss
reduction gate and conditional Q16. Stage A/B hyperparameters, 18 conditions,
FP32 and practical thresholds unchanged. No Stage A/B, DEV evaluation, query
controller, self-play, replication or confirmation evaluation authorized.

Current-source CPU/CUDA exact qualification, provenance, all datasets/audits,
TRAIN/DEV custody and sealed CONFIRM custody must all pass before the disposable
drill. Even a passing drill ends this authorization before Stage A.

## Continuation capacity census (not a data-contract amendment)

The full M1/M2/M3 census additionally finds KQvK eligible capacities
306/576/1076 and KRvK 189/532/438. All six light-family quotas are infeasible,
not only KRvK M1. Heavy-family capacities suffice for their original quotas.
See `evidence/v5/data/capacity-df261ab-all-families-m3.json`.
No lower count, reallocation, generation identity or recipe has been adopted.