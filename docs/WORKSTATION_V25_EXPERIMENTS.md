# Workstation V2.5 — experiment ledger

Each entry: QUESTION / HYPOTHESIS / PRE-REGISTERED RULE / CONFIG / DATA DIGEST / WALL /
PEAK VRAM / MEASURED / INFERRED / DECISION / NEXT ACTION. Rules are written BEFORE the
run. Entries are appended; they are never rewritten after results are seen.

## Pre-registered gates (fixed before any training)
- P0: all 20 architecture-correctness items must pass. No science if P0 fails.
- P0.5: if CandidateFacts costs >20% of end-to-end self-play evaluation wall, optimize
  without changing semantics.
- P0.6: training layout must be finite with peak VRAM <= 12 GB; failed cells stay visible.
- P1: CF, seed 1, LR in {3e-5, 7.5e-5, 1.5e-4, 3e-4}, ~75–100 updates, selection on TUNE
  policy CE / correct-move mass / stability (never train loss alone).
- P2 (L, C0, CF × 2 seeds, 400 updates, policy-only):
  - Q1 FACTS PATH GO iff CF M1 top-1 >= 0.95 AND CF > C0 paired 95% CI entirely > 0 AND both seeds positive.
  - Q3: CF M2 top-1 >= 0.75 and M3 top-1 >= 0.55 (both seeds at/near floors with pooled CI clear of chance).
  - One extension only: 400 → 800 updates if train/tune still improve with no overfit gap and healthy gradients. Then CONFIRM once more; if still short, STOP (no self-play).
  - Thresholds may be amended only BEFORE CONFIRM is seen.
- P3 (frozen conversion suite, argmax from ply 0, no noise, solver off): proof-trained CF
  beats corrected F10 by >= 10 wins/128 TARGET starts (searched), paired bootstrap 95% CI > 0,
  no catastrophic HEAVY regression; M3 CONFIRM playout conversion >= 80%.
- P3.5: one M4/M5 extension only if M1–M3 strong, M3 playout strong, transfer weak.
- P4: only if P3 passes; 2-cycle smoke at most; curriculum conversion < 70% for two
  consecutive cycles is a hard stop.

---

## E-DATA-1 — ProofTargetsV1 scale (PRE-REGISTERED before any dataset was generated)

QUESTION: how large can the hard-disjoint M1/M2/M3 datasets be under the pre-registered
filters (correct <= 15% of legal moves and CandidateFacts ambiguity for M2/M3; canonical
dedup; hard-disjoint splits)?

HYPOTHESIS: the requested ~1000/100/100 per (family, band) is not available for the
two-piece families.

MEASURED (exact exhaustive enumeration of every placement, symmetry-canonical classes that
pass the filters; `docs/evidence/v25/proof/proof-pool-*.json`):

| Family | M1 | M2 | M3 |
|---|---:|---:|---:|
| KQvK | 306 | 576 | 1,076 |
| KRvK | 189 | 532 | 438 |
| KQQvK | 95,649 | 174,163 | 4,409 |
| KQRvK | 111,273 | 306,595 | 211,215 |
| KRRvK | 41,612 | 122,082 | 108,086 |

Six of fifteen cells are below the 1,200 needed for 1000/100/100 (KQvK M1/M2/M3, KRvK
M1/M2/M3). The generator benchmark gate measured ~185–15,000 positions/s by band with a
projected full-scale wall under one minute, so the limit is pool size, not time.

PRE-REGISTERED RULE (all filters kept; nothing relaxed):
- A cell whose eligible pool P >= 1200 uses TRAIN/TUNE/CONFIRM = 1000/100/100.
- A cell with P < 1200 uses its whole pool split 80/10/10: TUNE = CONFIRM = floor(0.10 P),
  TRAIN = P - 2 floor(0.10 P). No position is reused across splits.
- Splits are selected from the exact pool by independent seeded shuffles (CONFIRM, then
  TUNE from the remainder, then TRAIN from the remainder), so they are hard-disjoint by
  exact FEN and canonical class by construction, and re-verified.
- Family x depth balance is therefore approximate, not exact. Results are reported pooled
  by depth AND by family; per-family confirm counts for KRvK M1 are small and are flagged.

EXPECTED SIZES: TRAIN 11,501, TUNE 1,208, CONFIRM 1,208.
DECISION: proceed. This is a scale decision inside the "approximately" latitude, not a
change to the exactness contract; if it should be treated as a material contract change the
owner can regenerate under a different rule in minutes (generation is cheap).
