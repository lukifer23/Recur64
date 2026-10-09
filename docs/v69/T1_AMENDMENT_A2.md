# Recur64 V69 — T1 amendment A2 (audit-found specification defects)

The A1 dataset (28,608 examples; seed unchanged) failed its pre-training audit (`audit_pass=false`, 4 failures; output preserved in `t1/attempt2_audit_failed_A1/`). No model was trained and no label was used for anything but the audit. Causes, both mine:
1. **Taken-set key (A1 §2 text defect).** A1 keyed the cross-partition taken-set on (canonical child, budget); the same canonical position can be a child of an M2 root (budget 1) and an M3 root (budget 2) and so appeared in two partitions. The contract (§1) requires that no canonical child identity span partitions. **A2: the taken-set is keyed on the canonical child key alone** (stricter; one example per canonical child overall).
2. **Audit defect:** the nested-scale check expected 12·2·k rows; train has 12 cells, so the correct count is 12·k (3,000/12,000/24,000 were correct).
Everything else is unchanged. Same seed re-used (exposed only to generation/audit statistics, no labels used for modelling). Regenerated before any training; the strict audit is re-run unchanged otherwise.
