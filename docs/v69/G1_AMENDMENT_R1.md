# Recur64 V69 — G1 amendment R1 (owner-approved)

Amends `G1_CONTRACT.md` (which stays frozen and unedited) after the first G1 attempt stopped at data construction (`RESULTS_G1.md`): the contract's whole-component exclusion could not supply the KQQvK remaining-budget-2 quota
(best round 100 examples short; ~40% of fresh roots overlap gen-001 directly and connected components percolate). The owner approved revision **R1** in the session and delegated the seed policy to the agent.
Frozen in `g1r1/frozen_protocol.json` **before the R1 seed is drawn**. Everything not listed here is unchanged from `G1_CONTRACT.md` and `g1_config.json` (candidates, quotas 128 per cell, per-root cap, selection, audit, controls, metrics, bootstrap, decision rules, limits, access roles).

## R1 changes

1. **Exclusion (contract §3):** a new root is removed iff its canonical root key, or the canonical key of any of its immediate children (all legal moves, terminal included), occurs in the gen-001 exclusion index (gen-001 accepted roots and **all** their immediate children; full 65-byte identities, never hashes).
   Connected groups are then formed **among the surviving roots** (shared canonical children) for deduplication and the cluster bootstrap. The component-level removal of the original contract is dropped.
   *Guarantee:* no kept G1 root or child canonical identity occurs in gen-001 (audited exhaustively with full keys). *Not guaranteed (stated limitation):* a G1 group may sit adjacent to, though never identical with, gen-001 positions; candidates were trained only on the 768 gen-001 fitting rows.
2. **Seed policy — fresh draw (agent's choice, owner-delegated):** a new 32-byte master seed from the OS CSPRNG at `g1r1/seed/g1_master_seed.hex`, recorded before generation, write-once. Reason: R1 was selected after observing the first seed's generation statistics, so reusing it would leave the argument that the revision was fitted to that seed's outcome;
   a fresh draw removes the question. The first seed (`2fef6b12a6cc5faa`, drawn 2026-10-09T02:38:17Z) is **abandoned**: it was exposed only to generation statistics, produced no rows/labels/panel and will never be used.
3. **Namespace:** all R1 artifacts live under `artifacts/v69/g1r1/` (append-only receipts; the attempt-1 `g1/` directory is preserved untouched and hashed in `g1r1/g1_attempt1_manifest.json`; the preservation receipts cover it). The evaluator is re-verified at the amended source digest with a new receipt.
4. Audit text is unchanged except that the exclusion check is the identity-level one; group ids are rebuilt from the kept pool.

Disclosure: the abandoned-seed diagnostics included per-root-variant availability counts (statistics only). No G1 example, label or model output existed before R1 was frozen.
