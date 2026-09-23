# Recur64 — Representations (Phase 1)

Durable contract for the chess layer. Version constants live in
`recur64-core::schema`: `OBSERVATION_VERSION_V1 = 1`, `ACTION_VERSION_V1 = 1`,
`RULES_PROFILE_VERSION_V1 = 1`.

## Squares, colors, pieces

- Indexing is `a1 = 0`, **rank-major** (`a1, b1, …, h1, a2, …, h8 = 63`), matching
  cozy-chess exactly.
- `Color ∈ {White, Black}`; `Piece ∈ {Pawn, Knight, Bishop, Rook, Queen, King}`.
- Files are fixed; ranks increase toward White's back rank.

## Canonical perspective

The **current side to move** determines canonicalization for the entire
observation and for action indices.

- White to move → identity.
- Black to move → **rank reflection** `square XOR 56` **and** color swap.
- Files are fixed, preserving kingside/queenside meaning.
- The transform is an involution: `canonical(canonical(s)) == s`.

## Observation V1 (`[64, 119]`)

Per square, 119 floats, laid out `[square][feature]` (square-major):

| Feature indices | Width | Meaning |
|---|---:|---|
| `0..13` | 13 | frame 0 piece one-hot |
| `13` | 1 | frame 0 validity |
| `14..27` | 14 | frame 1 (same layout) |
| … | | frames 2..6 |
| `98..111` | 14 | frame 7 |
| `112..115` | 4 | castling: own kingside, own queenside, opp kingside, opp queenside |
| `116` | 1 | en-passant target square indicator |
| `117` | 1 | `min(halfmove_clock, 150) / 150` |
| `118` | 1 | `min(repetition_count, 5) / 5` |

Arithmetic: `8 * 14 + 4 + 1 + 1 + 1 = 119`; total `64 * 119 = 7616`.

**Piece channel order** (canonical; "own" = side to move):

`0` own pawn, `1` own knight, `2` own bishop, `3` own rook, `4` own queen,
`5` own king, `6` opp pawn, `7` opp knight, `8` opp bishop, `9` opp rook,
`10` opp queen, `11` opp king, `12` empty.

Under Black canonicalization, "own" pieces are physically Black.

### History frames

- Frame `0` = current position; frame `k` = position `k` plies ago.
- Frames beyond available history are **unavailable**: all 13 piece channels `0`
  **and** validity `0`. An unavailable frame is **not** an empty board — the
  empty channel (12) is not set.
- The current side-to-move transform is applied to **every** frame, not to each
  frame's own mover.

### Metadata

- **Castling:** current rights only, broadcast to all squares; `own`/`opp` are
  relative to the side to move.
- **En passant:** FEN-style. The indicator is set on the EP target square after
  any double pawn push, whether or not a legal EP capture exists. It is
  reflected (`XOR 56`) under Black canonicalization. (The repetition key uses a
  stricter FIDE notion — see `RULES_PROFILE.md`.)
- **Halfmove clock:** cozy-chess caps the clock at 100, so under this profile the
  value is in `[0, 100]` and the feature is `clock / 150`.
- **Repetition count:** FIDE count of the current position over the **full**
  authoritative history (not just the visible frames), capped at 5.

## Action V1

- `ActionId = ((from * 64 + to) * 5 + promotion_code)`, space `20,480`.
- Promotion codes: `0 = none`, `1 = N`, `2 = B`, `3 = R`, `4 = Q`.
- Action indices use the **same canonical orientation** as the observation.
- This is an identity/storage space; the policy head is sparse (gather legal
  candidates only), never a dense 20,480-wide layer.

## Standard move convention (castling)

Recur64 has exactly one external convention: **standard chess / UCI**, where
castling is the **king destination** (`e1g1`, `e1c1`, `e8g8`, `e8c8`).

cozy-chess stores castling internally as **king-captures-rook** (`e1h1`, …). The
`recur64-core::uci` layer converts between the two; the internal form never
leaks into actions, observations, or replay. Conversion is cross-checked against
`cozy_chess::util::{parse,display}_uci_move`.

## UCI / FEN

- `StandardMove` ↔ UCI strings (`e2e4`, `e7e8q`, `e1g1`, en passant, promotion,
  underpromotion). Malformed or non-canonical strings fail visibly; trailing
  junk is rejected.
- FEN is an **interchange/debug** format only (CLI, fixtures, tests). It is never
  the hot-path internal state and never the canonical persisted training format.
