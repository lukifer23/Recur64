# Recur64 — Rules Profile V1

Version: `RULES_PROFILE_VERSION_V1 = 1`. Standard chess only (no Chess960).

## Draw convention

**Auto-claim on the current position** for threefold repetition and the 50-move
condition: the engine declares the draw on the first position where the condition
holds, without a claim. This is an **engine-training convention**, not a model of
optional human claim strategy. It is used identically in future self-play and
arenas. FIDE's automatic fivefold/75-move conditions are subsumed (auto-claim
fires earlier) and are not implemented separately.

## Termination and precedence

Checked in this exact order:

1. **Checkmate** — side to move in check and has no legal moves. *Decisive.*
2. **Stalemate** — side to move not in check and has no legal moves. *Draw.*
3. **InsufficientMaterial** — recognized dead position (below). *Draw.*
4. **ThreefoldRepetition** — FIDE repetition count ≥ 3. *Draw.*
5. **FiftyMoveRule** — halfmove clock ≥ 100. *Draw.*
6. **Truncated** — administrative ply cap reached. **Not a chess result.**
7. **Aborted** — external cancellation. **Not a chess result.**

Checkmate and stalemate are evaluated before any draw condition, so a draw rule
can never overwrite a mate. `Truncated`/`Aborted` yield `outcome() == None` and
must not be converted into draw WDL targets.

## Repetition identity

Positions are compared with cozy-chess `Board::same_position`, which implements
the FIDE definition:

- same board placement, same side to move, same castling rights;
- the en-passant square is **ignored when no legal en-passant capture exists**,
  and is part of the identity when a legal capture exists.

Halfmove and fullmove counters are excluded. A board-placement-only hash is
**not** used. The observation's EP feature is FEN-style and is deliberately
looser than this repetition key (see `REPRESENTATIONS.md`).

## Insufficient material (exact recognized set)

No pawns, rooks, or queens on either side, and one of:

- K vs K;
- K + exactly one minor (bishop or knight) vs K;
- bishops-only, where **all** bishops (both sides) stand on one square color.

**Not recognized** (deliberately): K+N vs K+N, opposite-colored bishops,
K+N+N vs K, and unusual blocked positions. Missing a dead position is safe (the
game ends by another rule); falsely declaring a draw is not. This is a
conservative material test, not a complete FIDE dead-position solver.

## Administrative cap

A configurable ply cap (e.g. 512) is `Truncated`, **not** a draw. Truncated
games carry no terminal-value supervision and may be persisted/continued instead.

## Perft reference provenance

Perft counts used in tests are from the Chess Programming Wiki "Perft Results"
page (<https://www.chessprogramming.org/Perft_Results>): the standard start
position and Positions 2–6. These are an independent oracle, not cozy-chess's
own tests.

## Independent differential oracle

`shakmaty` (GPL-3.0-or-later) is an **optional dev-only** oracle behind the
`oracle` feature (off by default). It compares legal move sets and
checkmate/stalemate. Draw adjudication is intentionally excluded because
Recur64's auto-claim profile differs from a library default.
