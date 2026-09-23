# Recur64 — Replay V1 (Phase 2)

Compact, versioned, checksummed replay. No dense observations and no 20,480-wide
policy vectors are stored.

## Schema

A `GameRecord` stores:

- `game_id`, `seed`, `start_fen`;
- `search` settings (simulations, `c_puct`, temperature, recurrence);
- `plies`: for each ply, the selected canonical `ActionId`, the sparse search
  target `Vec<(ActionId, prob)>`, total visits, and the side to move;
- `termination` (Rules Profile V1 label) and `outcome`
  (`0` white win, `1` draw, `2` black win; `None` for truncated/aborted).

A `ReplayHeader` records the replay schema version, observation/action/rules
profile versions, `run_id`, `model_id`, git revision, backend, and precision.

**Positions are reconstructed, not stored.** A game stores its start FEN and
selected moves; replaying them reproduces every `GameState` (history included)
and therefore every Observation V1. Legal actions are regenerated on read
(deterministic sorted order) and the sparse target is mapped onto them, so an
illegal or misaligned target is a hard error.

## Shard format

```
magic "R64S" | schema_version u32 | payload_len u32 | crc32 u32 | payload
```

`payload` is a bincode-encoded `Shard { header, games }` (bincode standard
config). A shard is written to `<name>.tmp`, flushed, then atomically renamed to
`shard-NNNNNN.r64shard`. Only complete shards appear in `manifest.json`, which is
itself written atomically. A leftover `.tmp` is never read.

## Audit

`recur64 replay-audit --input <dir>` verifies per shard and record:

- magic, schema version, declared length, and CRC;
- header contract versions match the current ones;
- every game replays legally move-for-move (history continuity);
- each selected action is legal in its position;
- every target action is legal; probabilities finite, non-negative, sum ≈ 1;
- an outcome is present **iff** the termination is decisive/drawn (never for
  truncated/aborted);
- checkmate outcome perspective matches the losing side to move;
- provenance present; no duplicate `game_id`.

A bad shard fails visibly with a non-zero exit and a machine-readable report.

## Versioning

`REPLAY_SCHEMA_VERSION = 1`. Contract versions come from
`recur64-core::ContractVersions`. A mismatch is refused on open/parse.
