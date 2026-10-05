# V5 HP native data V2 plan

Preregistered owner amendment, before accepted generation. The V1 plan and
measured light-family capacity failure remain historical and infeasible.
No V1 record is auxiliary training data. No P25 identity is adopted.

## Frozen identities and quotas

Family: `v5_hp_heavy_endgames_v2`; data contract: `v5_hp_data_v2`.
Selection: `v5_hp_dataset_select_v1`; audit: `v5_hp_independent_audit_v1`.
The future scientific recipe is `v5_stage_recipe_v3`, bound only after measurement.

| Split | Seed | Families | Per M1/M2/M3 cell | Total |
|---|---|---|---:|---:|
| V5_HP_TRAIN_V2 | 0x7A50_2001 | KQQvK, KQRvK, KRRvK | 3000 | 27000 |
| V5_HP_DEV_V2 | 0x7A50_2002 | KQRvK, KRRvK | 750 | 4500 |
| V5_HP_CONFIRM_V2 | 0x7A50_2003 | KQRvK, KRRvK | 750 | 4500 |

Generate TRAIN, then DEV excluding TRAIN, then CONFIRM excluding both. Both FEN
and authoritative canonical identities are excluded. Require zero within-split
duplicates and zero pairwise intersections. No quota reduction or family/depth
substitution is permitted. DEV primary KQRvK M3 has exactly 750 records.

## Exact semantics and deterministic enumeration

Reuse proof::generator canonical_key/fen_from_canon, exact MateSolver,
ProofTargets action/label semantics, and existing candidate filters unchanged.
White to move, bare black king, no castling/en-passant, halfmove zero/fullmove one,
fresh history. Minimal attacker mate depth; all correct root actions retained.
Complete legal list in ActionId order. Depth >=2: correct fraction <=0.15 and
CandidateFacts ambiguity; M1 exempt. No engine/tablebase/learned labels.

Exhaustive family pools are eight fixed shards, white-king square ranges
[0,8),[8,16),...,[56,64), each classifying depths 1..3. This is enumeration,
not RNG sampling. Shards bind all three selection seeds, source/config, range,
family, raw/live/canonical/eligible counts and candidate content. SHA256 of
canonical serde serialization with digest empty and elapsed wall zero binds the
scientific shard; elapsed wall is operational metadata. Identical overlapping
canonical candidates merge through a BTreeMap; conflicting labels are refused.
Require complete, nonduplicated range coverage. Completion order cannot affect
selection. Existing verified shards resume; invalid/interrupted files are
quarantined without overwriting. Each invocation is bounded below two hours.

Selection key is SHA256(selection contract bytes || zero separator || u64 split
seed little endian || canonical identity). Sort by (key, canonical identity),
take exactly quota after exclusions. Output records sort lexicographically by
family, mate depth, canonical identity. IDs bind split/family/depth/canonical
identity. Per-cell checkpoints bind selection/exclusions/source/config and
records; resumed cells are independently audited again. Do not overwrite them.

## Audit, hashes, custody, sealing

Every selected record is independently audited via proof::audit GameState
AND/OR traversal, not trusted solver serialization. Check material/canonical
identity, legal order, complete correct set/index alignment, minimal depth,
nonterminal root, fresh history and frozen filters. Memo resets every 16 records
bound memory; parallel audit completion does not alter scientific output.
Zero failures required. Reconstruct all three selections and labels in separate
regeneration output paths; compare bit-identical scientific digests. Reusing
verified exhaustive inputs is declared; this is not an independent pool census.

Raw artifact schema `v5_hp_dataset_artifact_v2`; compact manifest schema
`v5_hp_dataset_manifest_v2`. Scientific record-content digest is SHA256 of the
compact serde_json serialized sorted Vec<ProofPosition>. Existing ProofTargets
digest additionally binds target contracts/filters/split/seed. Set digests hash
sorted unique UTF-8 strings, each followed by newline. No paths, clocks, user or
host identity occur in scientific hashes. Producer source SHA is frozen at the
DATA-V2-A implementation commit; later consumer source qualification is separate.
Production must bind exactly that measured producer manifest, not mistakenly
invalidate immutable data when consumer code changes.

Raw paths: runs/v5/data/v2/v5-hp-{train,dev,confirm}-v2.json, ignored by Git.
Compact measured manifests/audit/disjointness/regeneration and CONFIRM seal are
committed under docs/evidence/v5/data/v2/ before production integration.
Custody verifies actual local raw bytes against committed scientific identities.
CONFIRM is generated/audited/sealed with evaluated=false. Custody does not run a
model. Ordinary training/evaluation refuses CONFIRM, without a bypass flag.

Cross-disjointness from unavailable workstation-only raw datasets was NOT
verified. Local historical inventory is informational. V4_TUNE_V1/HOLDOUT_C are
not used and remain unevaluated.

## Scientific integration and final gate

Only after DATA-V2-B commit/push, bind TRAIN-only and DEV-only loaders to measured
manifests. Nine equal TRAIN cells feed unchanged Stage A and each of 18 Stage B
condition samplers. Retain seed5301, FP32, AdamW, LR3e-4/warmup80, Stage A1200
updates/batch64/Q0 and Stage B800/batch36/Q2,4,8 x R1,2,4 x two schedules. Record
cell exposure. No training is authorized now. DEV total4500, six cells750;
bootstrap primary KQR M3 n750; practical thresholds unchanged.

Require fresh current-source full/focused release tests, changed-file rustfmt,
V5 Clippy, serial pinned CUDA build, CPU/CUDA qualification including unchanged
exact D9, all nine shapes, 50 resident Q8/R4 updates and checkpoint/resume, graph
provenance, all audits/disjointness/seal and all three local custody PASS before
one disposable reader drill. Existing stable-ID-hash selection chooses four TRAIN
records per KQR/KRR M1/M2/M3 cell. Q8/R4 <=200 updates, LR1e-3/warmup20, >=20%
mean correct-set loss reduction unless start<0.05. Q16 only under the existing
informative Q8-failure rule. Then STOP. No Stage A/B, DEV science, pilot,
controller, replication, self-play or confirmation evaluation.

## Post-result closure note ? population discrepancy (2026-10-05)

The original primary KQR M3 n750 bootstrap statement above is preserved.
The frozen implementation actually bootstrapped all4500 positions after
within-position schedule averaging. Its published NO_SIGNAL report is retained,
not relabelled as that original primary analysis. [Closure](V5_CLOSURE.md) records
this discrepancy and an explicitly retrospective supplementary750 analysis from
serialized records only. Future V6 primary code must filter and verify750 IDs.
This note does not amend the historical preregistration after measurement.
