# H3.6 root cause — arena truncation (MEASURED)

**Replay.** The cycle-2 parent arena was re-run with binary `a0e874c`. That binary
adds only the diagnostic `final_fen` field; seeds, scoring and identity are unchanged.
- Setup: candidate `snapshot-002` (`d0ee3ced…`) vs parent `snapshot-000`
  (`990e5e54…`), seed offset 2, the run's own `config.toml`.
- It reproduced the pilot's arena **game for game**: 32/32 move digests match,
  W/D/L/T = 12/12/4/4 (`c2-parent-arena-replay/eval-arena.json`).

The truncated games at the 400-ply cap:

| game | candidate | final material (White vs Black) | final FEN | winning side |
|---|---|---|---|---|
| g0 | White | K vs K+Q+N+N | `1n6/8/5n2/8/7k/2q5/8/6K1 w - - 23 204` | parent (can't mate a lone king) |
| g13 | Black | K+B+N vs K | `8/8/8/8/4K3/B7/5N2/5k2 w - - 86 204` | parent (KBN mate not found) |
| g15 | Black | K+B+N vs K | `8/7k/8/1N6/8/8/B7/1K6 w - - 0 204` | parent (KBN mate not found) |
| g19 | Black | K+B vs K+N | `6k1/2B5/8/6n1/8/8/8/K7 w - - 17 204` | none (dead draw) |

## Findings

**MEASURED:**
- 3 of 4 truncations are **won-but-unconverted** positions for the parent.
  One is a dead draw.
- Across all five H3.6 arenas, every one of the 17 truncated games reached
  the cap while its color-swapped partner ended normally.

**MEASURED consequence:** `candidate_score` excludes truncated games, so unconverted
wins silently drop out of the score. Promotion robustness under alternative
truncation scoring:

| arena | as played | T = draw | T = all parent wins (worst) | decision robust? |
|---|---|---|---|---|
| c0 vs parent | 0.517 | 0.516 | **0.484** | **no**: promote → hold in the worst case |
| c1 vs parent | 0.464 | 0.469 | 0.406 | yes (hold) |
| c2 vs parent | 0.643 | 0.625 | 0.562 | yes (promote) |
| c1 vs reference | 0.574 | 0.562 | 0.484 | — |
| c2 vs reference | 0.733 | 0.719 | **0.688** | robust above 0.5 |

**INFERRED mechanism:**
- The search is 32 simulations with root noise ε = 0.25 on every move. The
  near-uniform policy has raw-policy entropy close to uniform, and the value
  head is still small (mean |v| = 0.027).
- That combination can't find the multi-move mates KQNN-vs-K and KBN-vs-K
  need. The fifty-move counter keeps resetting (captures/pawn moves) or never
  reaches 100 before the cap.
- Self-play has the same failure: 2/2/1 truncated games per cycle, whose 400/800
  positions are **untrainable**. So the value head is never trained on exactly
  these won-but-unconverted positions.

## Next steps

These are **not** executed in this pass; they are R15-entry items.

1. **Truncation-aware arena scoring.** Pre-register it before any R15 strength
   comparison:
   - report T-as-draw and a simple material adjudication at the cap;
   - promotion should read a score that cannot improve by failing to convert.
2. **Truncated self-play games.** Decide explicitly (ADR) whether to adjudicate
   truncated games into WDL targets or keep discarding them. Either way the
   loss of these positions is a known bias of the value data.
3. **Arena ply cap / mate conversion.** This is measure-first. Evaluate whether
   the arena should end dead-material positions (K+B vs K+N) as draws, and
   whether a higher cap helps at all given the conversion failure.
