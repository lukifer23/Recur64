# Recur64 — F10 + PUCT Baseline (Phase 3)

The first real feed-forward control baseline. Self-play only, random init, no
Gumbel / R10 / diffusion / geometric bias / auxiliary heads / external labels /
BF16. This is the control for all later research.

**Status:** system works end-to-end; learning health is **not yet good enough for
a 24h run**. Decision: **CONDITIONAL GO** (see the decision package).

## Architecture

| | |
|---|---|
| Model | F10: width 384, heads 12, FFN 768, input 0 / core 8 / output 0 |
| Unique params | 9,805,288 |
| Recurrence | 1 (feed-forward execution) |
| Precision | FP32 |
| Backend | Burn 0.21.0, CUDA (user-space CUDA 12.9.1) |

## Training contract

- AdamW, lr 3e-4, weight decay 1e-4, gradient clipping (norm) 1.0.
- LR schedule: linear warmup then cosine decay, spanning the whole pilot
  (`planned_updates = cycles × max_updates`).
- Effective batch via accumulation: physical `train_batch` × `accumulation_steps`.
- Loss: policy CE over legal candidates + WDL CE (truncated/aborted excluded).
- Per-update metrics: total/policy/WDL loss, grad norm, LR, policy entropy.
- Checkpoint v2 with chess contract versions, `model_id` content hash, and the
  schedule step; F10 resume is bit-exact on CPU (verified).

## Search contract

- PUCT only, `c_puct = 1.0` (retained; no pathology check found it clearly wrong).
- Simulations/move: **frozen at 64** for the baseline (Stage A measured 16/32/64).
- Value perspective flips every ply; terminal nodes never call the network.

## Self-play exploration

- Sample proportional to root visit counts at **temperature 1.0 for all plies**.
- **No Dirichlet noise, no Gumbel.**
- Standard start. `active_games` = concurrent games (one per thread).

## Replay policy

- Replay V1, uncompressed, checksummed, atomic shards.
- Capacity `replay_max_positions = 100_000`; oldest shards archived (never
  deleted).
- On-demand streaming sampler (recency-biased), so memory does not grow with
  capacity.
- Reuse target `2.0` (`examples_consumed / new_positions_inserted`).

## Hardware

Windows 11, Intel Core Ultra 9 285K (24c), 63 GB RAM, NVIDIA RTX 2000 Ada 16 GB,
native Burn CUDA (FP32).

## Stage A — systems profiling (measured)

- Batcher now coalesces: batch mean 20–34 (was 1.0 before the concurrency fix);
  VRAM ≤ 2.4 GB.
- Search budget vs throughput (active 64, batch 64, ply_cap 100):

| sims/move | positions/s |
|---:|---:|
| 16 | 47.0 |
| 32 | 33.5 |
| 64 | 17.5 |

- Warmup (F10 forward, batch sizes 1–64) ≈ 2.5–5 s, recorded separately from
  steady state.

## Stage B — short learning smoke (2 cycles)

- 2 cycles completed; both candidates promoted (arena 0.5).
- Loss unstable within a cycle (1.86→0.89, then 1.14→6.05); grad norm high.
- Replay reuse ≈ 0.09–0.16 (far below the 2.0 target).
- Raw policy vs random: 1.0 then 0.5 (noisy, tiny sample).

## Stage C — bounded pilot (4 cycles, ~62 min)

Command: `recur64 pilot --config configs/f10-stage-c.toml --run-dir runs/f10-stage-c-1`.
Status: `budget_exhausted` after 4 cycles. ~44,232 positions.

| cycle | positions | updates | reuse | loss (first→last) | grad norm | policy entropy | arena | raw vs random |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | 14,085 | 8 | 0.07 | 1.93→2.23 | 44.1 | 0.60 | 0.475 | 0.231 |
| 1 | 13,818 | 8 | 0.07 | 1.29→3.66 | 88.2 | 0.44 | 0.500 | 0.188 |
| 2 | 7,866 | 8 | 0.13 | 1.44→3.30 | 70.3 | 0.73 | 0.475 | 0.321 |
| 3 | 8,463 | 8 | 0.12 | 2.20→1.80 | 69.2 | 0.86 | 0.450 | 0.167 |

Throughput: ~44k positions/hour at sims 32, 64 games/cycle plus a 20-game arena
and a 20-game raw match; ~15 min/cycle.

### Data health
- Games are generated legally; **zero illegal actions**; audit passes.
- Searched games are dominated by **repetition / fifty-move draws** (e.g.
  cycle 1 arena: 18 threefold, 2 fifty-move, 0 decisive) — the untrained value
  function does not make progress, so PUCT shuffles into repetition. This makes
  the policy targets repetitive.
- Raw (no-search) games are decisive (mostly checkmate), so the environment is
  not itself degenerate; the search is.
- Truncation rate low.

### Learning health
- **Loss is unstable** (often rising within a cycle) and gradient norms are high
  (44–88).
- Policy entropy 0.44–0.86 (not collapsed, but low).
- **Replay reuse 0.07–0.13**, far below the 2.0 target: the learner under-consumes
  the data it generates (`max_updates` is too small relative to new positions).
- Raw policy vs random is **below 0.5** (0.17–0.32): the network is not yet
  better than random, and may be degrading.

### Evaluation
- Arena candidate-vs-reference is uninformative (near-all draws by repetition).
- Raw-policy vs random is the more informative signal at this stage, and it is
  poor.

## Failure cases / limitations

- The searched-play arena cannot discriminate untrained candidates because the
  games are repetition draws.
- The learner's update count is not yet tied to the reuse target, so it
  under-trains per cycle.
- Training stability (loss/grad) is not yet controlled at this scale.
- No Elo/strength claim is made or implied. Numbers are single-run on one
  workstation.

## Decision package — Stage C → long run

- **System health:** OK. No crashes/deadlocks/corruption; audit passes; checkpoints
  and F10 resume work; memory bounded; replay healthy.
- **Data health:** mixed. Legal games, low truncation, but searched self-play is
  repetition-dominated and the learner under-consumes it.
- **Learning health:** poor. Unstable loss, high grad norm, raw policy ≤ random.
- **Throughput:** ~44k positions/h; ~15 min/cycle at sims 32.
- **Evaluation:** arena uninformative; raw policy poor.

**Recommendation: CONDITIONAL GO.** Before a ~24h baseline, fix a small set of
specific issues and re-run a bounded pilot:

1. **Tie updates to the reuse target** — compute `max_updates` from
   `new_positions × reuse_target / effective_batch` so the learner actually
   consumes data (target reuse ≈ 2).
2. **Training stability** — lower the peak LR or lengthen warmup; keep grad
   clipping; confirm loss decreases within a cycle and across cycles.
3. **Repetition-dominated search** — investigate whether the untrained value
   function plus `c_puct = 1.0` causes shuffling; consider a lower search
   temperature late-game and/or a value-informed move selection, and re-measure
   the decisive-game fraction.
4. **More informative evaluation** — require a minimum decisive-game fraction
   before an arena is considered meaningful.

No ~24h run will be started without explicit owner approval after a successful
re-run.
