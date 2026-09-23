# Recur64 — Runs (Phase 2)

`recur64 run` executes a bounded vertical slice:

```
COLLECT → AUDIT → TRAIN → EVALUATE → REPORT
```

It is bounded by the config (games, updates, arena games) and by
`run_budget_minutes`. There are no indefinite loops.

## Run directory

```
runs/<run-id>/
    config.toml        # resolved config (no hidden defaults)
    metadata.json      # status, versions, device/precision
    logs/
    replay/            # manifest.json + shard-NNNNNN.r64shard
    checkpoints/       # reference/ and candidate/
    eval/              # arena output
    report/            # report.json + report.md
```

`RunDir::create` refuses to reuse an existing directory unless `--force` is
given.

## Phases

1. **COLLECT** — a fresh Micro reference checkpoint is saved; self-play games run
   through the single inference owner; replay shards are written. The reference
   model is immutable for the whole phase.
2. **AUDIT** — `replay-audit` must pass; a failure aborts the run visibly.
3. **TRAIN** — the reference weights are loaded into a training model, the
   learner runs a bounded number of updates, and a candidate checkpoint is saved
   atomically. If no completed game produced a result, the reference is published
   as the candidate and no training is reported.
4. **EVALUATE** — a paired-color arena compares candidate vs reference with the
   same rules profile, search budget, and recurrence.
5. **REPORT** — `report/report.json` and `report/report.md` are written and the
   metadata status is updated.

Only one accelerator is used: the inference owner is shut down before training
and arena, so self-play inference never competes with training.

## Interruption / recovery

Ctrl+C sets a cancellation token.

- Workers stop after the current game; in-flight inference requests are answered
  (or drained with `Shutdown`); only fully written shards are published.
- Training finishes the current update and saves a recoverable checkpoint.
- `metadata.json` records `interrupted`; a run is never reported as `completed`
  if it was interrupted.
- A pre-cancelled run saves the reference checkpoint and leaves no candidate.

## Arena

The arena uses one synchronous evaluator per model (candidate and reference),
paired colors, a deterministic opening (standard start unless a config overrides
it), fixed search budget, fixed recurrence, and deterministic move selection
(temperature 0). It records W/D/L, truncations, termination distribution, model
ids, and the candidate score. This is a systems comparison, not an Elo claim.

## Metrics

The run report records self-play/inference metrics from the batcher: requests
submitted/completed/errors, batch count and size (mean/p50/p95/max), queue wait
(mean/p50/p95), and forward latency. See `docs/BENCHMARKS.md` for measured
baselines.
