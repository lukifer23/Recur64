# Recur64 — Search (Phase 2)

PUCT is the transparent control search. Gumbel, MCTS variants, transposition
merging, and neural caches are deferred.

## Value perspective (the highest-risk contract)

Every value is from the **side-to-move perspective** of the position it belongs
to.

- Terminal: checkmate → `-1` for the side to move; stalemate/insufficient/
  threefold/fifty → `0`. Terminal nodes **never** call the evaluator.
- Neural: `value = P(win) - P(loss)` from the softmax over the WDL head.
- Backup flips the sign every ply: a child value `v` (from the child's
  perspective) is added to the parent edge as `-v`.

`recur64-search` contains a dedicated perspective suite (mate-in-1, terminal
win/loss/draw children, alternating backup over 1/2/3 plies, "neural +1 at child
becomes −1 at parent").

## Formula

```
Q(s,a) = w / n                 (Q = 0 when n = 0)
U(s,a) = c_puct * P(s,a) * sqrt(N_total) / (1 + n(s,a))
select  a* = argmax_a [ Q(s,a) + U(s,a) ]
```

Ties are broken deterministically by higher prior, then lower `ActionId`.
`c_puct` is explicit config (Phase 2 pilot default `1.0`, untuned).

## Budget

`search(root, n_simulations)` performs **exactly** `n_simulations` traversals for
a non-terminal root. Traversal 0 expands the root (one neural evaluation); each
later traversal expands exactly one new leaf (one neural evaluation) or stops at
a terminal leaf (no neural evaluation). A terminal root performs zero traversals
and produces no target. This is asserted by tests (traversal count and
evaluator-call count).

## Target

`pi(a) = n(a) / sum_b n(b)` over root edges. If there were no visits (degenerate
budget) the prior distribution is used; if priors are degenerate, uniform.
Terminal roots produce no target.

## Evaluator interface

`recur64-search::Evaluator` is a narrow, blocking, single-position interface:

```rust
fn evaluate(&self, request: EvalRequest<'_>) -> Result<EvalResult, EvalError>;
```

`EvalRequest` carries the canonical observation, the canonical legal actions
(sorted), and the side to move. Implementations:

- `BatchedEvaluator` (runtime): submits to the single inference owner and blocks;
  the batcher coalesces across game threads.
- `SyncEvaluator` (runtime): one model, used by the arena.
- `FixedEvaluator` / `ScriptedEvaluator` (search): deterministic test evaluators.

Search never depends on Burn or on `recur64-model`.

## Play

`play_game_from` / `play_game_seeded` (in `recur64-search`) play complete games:
one search per ply, move selection by temperature over visit counts
(`temperature = 0` → argmax), terminations from Rules Profile V1. Truncated and
aborted games carry no outcome.

## Deferred

Gumbel, transposition merging, virtual loss, parallel selection within a tree,
root reuse, and neural evaluation caches.
