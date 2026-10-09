# Recur64 V69 — T1 amendment A1 (partitioning)

Amends `T1_CONTRACT.md` §1 (frozen; unedited) after the first T1 generation attempt showed — before any row, label, partition file or model existed — that the contract's **component-level partitioning cannot supply the validation/test quotas**:
with ~67% of fresh roots excluded for prior-data overlap, the kept pool's connected components **percolate** (32,563 kept roots collapsed into 61 components), so almost everything lands in the single giant component (forced to train) and held-out supply stays ~4.6k examples short for any pool size (same mechanism as the G1 first attempt). No rows, labels or models were produced; the generator output was discarded (the same drawn seed is reused: it has been exposed only to generation statistics).

## A1 changes
1. **Partition assignment is per root** (label-blind): keyed hash of the canonical root key under stream `t1/partition` → 80% train / 10% val / 10% test. The size-based component rule is dropped.
2. **Cross-partition isolation of what models see:** partitions are filled test → val → train with one global taken-set of (canonical child, budget), so **no identical child position occurs in two partitions** and no root spans partitions. Unselected children are never model inputs.
   *Stated limitation:* two roots in different partitions may share an *unselected* child, and positions near each other in position space can fall in different partitions (as in any random split of this domain); the G1 evaluation panel and gen-001 remain identity-disjoint from all T1 data (identity-level exclusion unchanged).
3. **Reporting/bootstrap groups** = connected components (shared canonical children) among the *contributing roots of each partition*, formed after selection, per partition (reproducible from the stored contributing pool; audited by rebuild).
Everything else (quotas, caps, exclusion against gen-001 + G1 pool, audit incl. exact oracle label invariance under the 8 board transforms, grid, selection rule, decision rules, limits) is unchanged. Frozen in `t1/frozen_protocol_a1.json` before generation resumes.
