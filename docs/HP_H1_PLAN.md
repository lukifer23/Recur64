# HP H1 — Experimental harness hardening + F15 qualification

Plan only. No code, commits, training, self-play, or GPU work in this step.
Implementation starts only after explicit approval.

Branch inspected: `experiment/hp-r15` at `89d5ccba4c986dbe5148bbe0f0ffd7f5c212c7b1`.
Findings below are from the source and from committed run artifacts. Tests were not re-run. No CUDA job was started.

## Verdict on the fifteen suspected issues

All fifteen are confirmed in the current source. Qualifications matter; the fix follows the code, not the suspicion alone.

| # | Claim | Verdict |
|---|---|---|
| 1 | Pilot bypasses `SelfPlayMetrics` | Confirmed. `coordinator::run` and `collect_only` share `collect_parallel` and record metrics. `pilot::collect_games` is a second copy and records none. `sweep::run_cell` is a third copy. |
| 2 | `active_games` is both the total and the thread count | Confirmed in `collect_parallel` and `collect_games`. The sweep already separates them (`cell.active_games` vs the `games` argument). |
| 3 | `cpu_workers` does not limit the pilot | Confirmed for every current collect path. It is copied into `SelfPlayMetrics` and printed. The thread pool size is `active_games`. |
| 4 | `replay_reuse_target` does not set the update count | Confirmed. The pilot always passes `cfg.max_updates`. The ratio it logs is `examples_consumed / all new plies`, and nothing compares it to 2.0. |
| 5 | Adam state resets while the LR step continues | Confirmed on the pilot path. `load_training` and the CPU `f10_resume` test can preserve moments for an in-process `train_step` save/load. The pilot never calls `load_training`. It loads weights and builds a fresh AdamW. `cumulative_updates` also advances when the candidate is not promoted. |
| 6 | Lineage can name the candidate as its own parent | Confirmed on promotion only. `snapshot_model_id` is overwritten before `LineageRecord` is built, so `parent_model_id` and `replay_model_ids` become the candidate. A non-promoting cycle still records the real parent. |
| 7 | Lineage git revision stays `None` | Confirmed. `RunMetadata` does record `RECUR64_GIT_SHA` / branch from `build.rs`. `LineageRecord`, `ReplayHeader::new`, and `CheckpointMeta::new` hardcode `None`. |
| 8 | Draw-only 0.5 promotes because 0.5 ≥ 0.35 | Confirmed. An all-truncated arena is worse: `decided == 0` forces `candidate_score = 0.5`, which also clears 0.35. The arena is always candidate vs the frozen reference, including after a promotion, so it is not a parent comparison. |
| 9 | Bad openings fall back to startpos | Confirmed on the pilot and the arena. `load_openings` turns a missing suite into an empty list. `run_arena` and `raw_policy_vs_random` replace an unparseable FEN with startpos. An invalid self-play `start_fen` skips that game and continues. The `eval-policy` CLI already errors if the suite file fails to load; invalid FENs inside a loaded suite still fall back. |
| 10 | Accumulation metrics are the last microbatch | Confirmed. `components` is overwritten each microbatch. Burn 0.21 `GradientsAccumulator::accumulate` sums (`grad.add(new)`); it does not average. |
| 11 | One hash mixes science and scheduling | Confirmed. `config_hash` hashes the whole `RunConfig`. |
| 12 | `peak_vram_mb` is not a peak | Confirmed for `sweep.rs`: one sample before the cell and one after. The 1,953 / 2,113 MiB figures in `docs/HP_EXPERIMENT.md` come from the separate 0.5 s `nvidia-smi` logs (`runs/gpu-monitor-f15.csv` contains 1953), not from this field. |
| 13 | Pilot concurrency 32 is the wrong HP default | Confirmed as a configured lie, not as a measured optimum. `configs/hp/f15-pilot.toml` sets `active_games = 32` and `cpu_workers = 8`. Today that starts 32 threads and ignores 8. The 334.6 eval/s point is real (`runs/hp-concurrency-sweep/sweep.json`, cell `ag32-cw12-mb64-to2000-r1`) but it was pre-merge: `cpu_workers = 12` was the thread count, games were short (mean 56.9 plies, 24/32 truncated), and batch p50 was 11 under a cap of 64. It is a prior for the new sweep, not a value to freeze. |
| 14 | F15 and R15-R1 are different functions | Confirmed from `forward_r`. The parity test does not claim otherwise. |
| 15 | No per-batch recurrence schedule | Confirmed. `LearnerConfig.recurrence` is one `usize` for the whole segment. This is a next-phase gap, not an H1 code change. |

## 1. Current branch state

| Item | Value |
|---|---|
| Branch | `experiment/hp-r15`, tracking `origin/experiment/hp-r15` |
| HEAD | `89d5ccba4c986dbe5148bbe0f0ffd7f5c212c7b1` — HP docs: record Phase 3 merge, updated priors, and revised next steps |
| Merge | `fa66c3297332f083b3f6236bc45a8af6b0e7df5d` — parents `21834dad` (HP) and `5ac291c6` (Phase 3) |
| Phase 3 mainline | `5ac291c6736b6037703522c49db62f2296c20bde` is an ancestor |
| Fork point | `78be2052612236547f6b5232b417175c2ccdcfc9` is an ancestor |
| Working tree | `Grok-plan.md` only. HEAD's copy is an empty file. The working copy is the 1,561-line H1 brief and is uncommitted. |
| Toolchain | Rust 1.97.1, Burn `=0.21.0`, user-space CUDA 12.9.1, FP32 gate refuses BF16/FP16 |

Direct F15 CUDA numbers in `runs/hp-f15-cuda/bench.json` match the brief: warm inference about 639 / 627 / 632 ex/s at batches 16 / 32 / 64, training batch 64 about 220 ex/s. Those are direct-model benchmarks, not self-play.

## 2. Merge review

Phase 3 (`5ac291c`) brought in the pieces this branch now actually runs:

- `RunConfig` fields for cycles, replay capacity, `replay_reuse_target`, warmup, accumulation, snapshot policy, opening suite, and a single `config_hash`.
- `recur64 pilot` (`pilot.rs`): collect, audit, train, evaluate, conservative promotion.
- Streaming `ReplayStore`, capacity archiving, learner accumulation, warmup + cosine, raw-policy eval, frozen openings, arena CI.
- `bench-runtime` and its workstation grid (concurrency 32–256).
- The F10 workstation result in `docs/F10_BASELINE.md`: reuse 0.07–0.13 versus a target of 2.0, unstable loss, repetition-dominated search, arena scores near 0.5. Those numbers are the RTX 2000 Ada workstation. They are not HP measurements.

The HP side of `fa66c32` kept:

- F15/R15 geometry and `f15_r15_parity.rs`.
- `SelfPlayMetrics` (terminations, W/D/L, truncation, `peak_in_flight`) on the coordinator path only.
- Build-time git SHA and branch on `RunMetadata`.
- `hardware_profile` / `model_profile`.
- The precision gate on the run paths.
- HP configs under `configs/hp/` and `configs/hardware/hp-home.toml`.

What the merge did not do: point the pilot at `collect_parallel`, split game count from concurrency, make reuse control updates, restore Adam state, or freeze an HP schedule. `docs/HP_EXPERIMENT.md` still says the branch does not depend on Phase 3 and is reproducible from `78be205` alone. That sentence is false after `fa66c32`.

## 3. Confirmed bugs

### B1 — Two learning collectors, and the pilot drops health metrics

- Severity: P0. A learning result from the pilot cannot be audited for data health.
- Where: `pilot.rs::collect_games` (about lines 87–174) versus `coordinator.rs::collect_parallel` (about 138–211). `CycleReport` has games, positions, train, arena, raw, and a reuse ratio. It has no `SelfPlayMetrics` and no `MetricsSnapshot`.
- Evidence: both functions set `concurrency = active_games` and `games_total = active_games`, spawn that many threads, and swallow `play_game_from` errors with `if let Ok`. Only the coordinator snapshots inference metrics and builds termination counts.
- Impact: the F15 pilot would not report W/D/L, checkmate, threefold, fifty-move, truncation, batch p50, queue wait, or inference errors.
- Fix: one collector. Pilot, `run`, `collect_only`, and the sweep call it. The pilot report embeds its metrics.

### B2 — `active_games` couples throughput to sample size

- Severity: P0.
- Where: the two collect functions above. Defaults in `config.rs`: `active_games = 32`, `cpu_workers = 8`.
- Evidence: the same integer is the thread count and the number of games fetched from the atomic queue.
- Impact: "collect 64 games with 12 in flight" cannot be expressed. Raising the sample size also raises thread count.
- Fix: `games_per_cycle` and `concurrent_games`, with the legacy mapping in section 9.

### B3 — `cpu_workers` is a dead knob

- Severity: P0. It reads as a limit and does nothing.
- Where: stored on `RunConfig` and `SelfPlayMetrics`; accepted by `--cpu-workers` in `phase2.rs`. No collect path reads it to size threads.
- Evidence: pre-merge, the sweep's `peak_in_flight` matched `cpu_workers` (the 334.6 cell has `peak_in_flight: 12`). Post-merge the pool follows `active_games`. The CLI flag still exists, so a rerun of `hp-concurrency-sweep.ps1` would not reproduce that cell.
- Fix: section 9. Do not leave the field looking live.

### B4 — Reuse target is a comment

- Severity: P0. This is the failure F10 already measured.
- Where: `config.rs` documents `examples_consumed / new_positions_inserted` and defaults the target to 2.0. `pilot.rs` sets `LearnerConfig.max_updates` from `cfg.max_updates` (200 in `configs/hp/f15-pilot.toml`). Nothing reads `replay_reuse_target` except parsing and the hash.
- Impact: a cycle that generates thousands of positions still takes a fixed 200 updates, or whatever cap was copied from F10. Achieved reuse is an accident of how many positions showed up.
- Fix: section 10.

### B5 — Promoted weights, fresh Adam, advanced LR

- Severity: P0. Multi-cycle loss curves would not be one optimizer trajectory.
- Where: `pilot.rs` inside the cycle loop: `model_io::load` then `let mut optim = adamw()`. `start_update: cumulative_updates`. `cumulative_updates += report.updates` happens before the promotion decision.
- Evidence: `save_training` writes `optimizer.mpk` and the checkpoint meta already has `update_counter` and `lr_schedule_step`. `load_training` restores both. The next cycle never loads them. `model_io::load` restores weights only.
- Burn detail, verified in burn-core 0.21: loading a module restores each `ParamId` from the checkpoint record, and the optimizer record is a `HashMap` keyed by those ids. `load_training` is the path that keeps weights and moments matched. A fresh `adamw()` discards the map.
- Extra defect: a rejected candidate still advances `cumulative_updates`, so the next attempt on the old weights uses a later cosine step and empty moments.
- The existing `f10_resume_preserves_optimizer_and_schedule` test proves a same-process `train_step` save/load. It does not pass through `run_pilot`.
- Fix: section 11.

### B6 — Promotion rewrites the parent id before lineage is written

- Severity: P0.
- Where: `pilot.rs` promotion block assigns `snapshot_model_id = candidate_model_id.clone()`, then `append_lineage` sets `parent_model_id` and `replay_model_ids` from that variable.
- Impact: a promoted cycle claims the candidate generated its own training data. The checkpoint that actually generated the replay is lost from that row.
- Fix: section 13. Capture `parent_snapshot_id` before training and do not overwrite it.

### B7 — Git SHA never reaches lineage, replay, or checkpoints

- Severity: P0 for lineage. Same one-line omission in replay and checkpoint metadata.
- Where: `pilot.rs` `git_revision: None`. `ReplayHeader::new` sets `None`. `CheckpointMeta::new` sets `None`. `RunMetadata::new` is the only caller that uses `option_env!("RECUR64_GIT_SHA")` and `RECUR64_GIT_BRANCH`.
- The sweep artifact shows the hole: `runs/hp-concurrency-sweep/.../manifest.json` has `"git_revision": null` and `"run_id": "hp-f15-selfplay"`.
- Fix: the runtime, which owns the build-time env, fills all three. Do not move `build.rs` into the model crate.

### B8 — Promotion accepts an uninformative arena

- Severity: P0.
- Where: `pilot.rs` promotes when `audit.ok() && updates > 0 && arena.candidate_score >= promotion_score_floor` (default 0.35). `arena.rs` sets the score to 0.5 when every game is truncated, and to 0.5 when every decided game is a draw.
- F10 already promoted on arena 0.5. `configs/hp/f15-pilot.toml` and `configs/hp/r15-pilot.toml` still set the floor to 0.35.
- The comparison target is always `reference_ckpt`, the random init, not the parent snapshot.
- Fix: section 12.

### B9 — Silent opening substitution

- Severity: P1, and it blocks a frozen evaluation. Treated as required before the smoke.
- Where: `pilot.rs::load_openings` (`unwrap_or_default`). `arena.rs` line 120 and `eval_policy.rs` line 119 (`unwrap_or_else` startpos). Collect paths `continue` on a bad `start_fen` after the game index has already been consumed, so the run can finish with fewer games and no error.
- Fix: section 12. Standard start remains the path when no suite and no `start_fen` were requested.

### B10 — Last-microbatch loss, summed gradients, per-tensor clip

- Severity: P1, required before any loss number is interpreted.
- Where: `learner.rs` `run_updates`. `train.rs::adamw` sets `GradientClippingConfig::Norm(1.0)`.
- Burn 0.21, read from the pinned crate:
  - `GradientsAccumulator` sums parameter gradients.
  - `policy_ce` and `wdl_ce` are already means inside the microbatch. Summing those backward grads multiplies the step by the microbatch count relative to a mean over the effective batch, until clipping binds.
  - Clipping is per parameter tensor, inside `SimpleOptimizerMapper::map_float`, before Adam sees the gradient. It is not a global L2 clip.
  - `global_grad_norm` runs on the summed grads before `optim.step`. The logged `grad_norm` is a pre-clip global norm of the sum. F10's 44–88 figures are that quantity. They are not the norm Adam applied, and they are not a global-clip threshold.
- Fix: section 10's sibling in section 7 (ticket H1-08). Mean-reduce so the configured LR refers to the effective batch. Label pre-clip global norm, and record how many parameter tensors the per-tensor clip actually touches. Do not rewrite Burn's clip into a global clip in H1.

### B11 — One config hash

- Severity: P1.
- Where: `RunConfig::config_hash` serializes the entire struct, including `active_games`, batch caps, timeout, device, and `hardware_profile`.
- Fix: section 7, ticket H1-09.

### B12 — Boundary VRAM labeled as peak

- Severity: P1 for the sweep. Do not quote `peak_vram_mb` from the current `bench-runtime` as a peak.
- Where: `sweep.rs::sample_vram_mb`, called once before `thread::scope` and once after it, then `max`.
- Fix: a sampler thread during the cell. If `nvidia-smi` fails, the field is absent and the report says so.

### B13 — Play errors disappear

- Severity: P0, alongside the collector. Not in the original list; it is in the same functions.
- Where: `if let Ok(mut g) = play_game_from(...)` in both collectors, and `Err(_) => continue` / `break` on a bad FEN.
- Fix: a configured invalid FEN aborts the run. A search or inference error increments a visible failure count and fails the cycle if it is non-zero. Do not drop games and still report success.

### B14 — `train_from_store` reports `games_skipped = 0`

- Severity: P1 telemetry.
- Where: `learner.rs` `train_from_store` passes `0` for skipped games and `store.total_games()` as games used. Truncated games are excluded from `sampleable()` and then omitted from the report.
- Fix: report sampleable positions, games with a result, and games skipped.

## 4. High-confidence risks

These are not counted as bugs to fix inside H1 unless a ticket below says so.

- R1. Mean gradient reduction changes training dynamics relative to every F10 number. F10 summed microbatch grads and then per-tensor-clipped them, so the clip was usually active. After a mean reduction the clip may go quiet. That is the point of making LR mean something, and it means F10 grad norms are not a baseline for H1 curves. No HP learning run exists yet, so nothing already collected on this machine is invalidated.
- R2. The sampler is recency-biased over the whole active store (`weight = shard_index + 1`). `examples_consumed / new_trainable_positions` is an accounting ratio, not a promise that each new position is seen twice. Report the fraction of consumed examples whose game id was generated this cycle.
- R3. Untrained PUCT on F10 was repetition-dominated, and deeper search made targets worse rather than better. The same thing can happen on F15. The search sweep exists to observe it. A repetition-dominated result is a gate outcome, not a reason to raise the simulation cap.
- R4. F15's `alpha` is nearly inert. With zero input blocks the injection is `RMSNorm((1+α) x)`. Burn's RMSNorm divides by the RMS, so a positive scale cancels apart from epsilon. On R15, `h` is two blocks away from `x`, and `h + αx` is a real residual. Do not "fix" this inside H1. It is part of why F15 is a different family.
- R5. `forward_r(..., deep_supervision = false)` is what the learner calls. One readout, after the output blocks. Deep supervision stays off. Turning it on would change the loss and the executed-block count (R15 R4 would be 26 blocks, not 20).
- R6. Arena CI is a per-game normal interval. Games are paired by opening (`i / 2`) and color (`i % 2`). The interval ignores the pairing. Acceptable as a diagnostic label in H1. Not acceptable as the uncertainty on a future recurrence headline.
- R7. Replay archiving moves shard files and then rewrites the manifest. A power loss in between leaves the old manifest pointing at files that now live in `archive/`. P2. Not a smoke blocker. See section 8.
- R8. The 334.6 eval/s prior used short, mostly truncated games. A standard-start sweep at ply cap 400 will be slower and must be allowed to say so.
- R9. Cross-process optimizer resume depends on Burn restoring `ParamId`s from the checkpoint. The 0.21 loader does that (`Param::from_item` / `transform_for_load`). The H1 test has to go through `load_training`, not through a cloned in-memory module alone, so a future Burn change cannot hide behind the old test.
- R10. `configs/hardware/hp-home.toml` still says gradient accumulation is not implemented. It is implemented. The file is a stale template, and nothing merges it into a run config.

## 5. Scientific confounds

### F15 is not R15 at R=1

`forward_r` does this:

```
x = embed(board)                          // input projection + square embedding
h = input_blocks(x)
for each of R iterations:
    h = core_blocks(inject_norm(h + α x))  // α = sigmoid(alpha_logit), init 0.1
y = output_blocks(h)                       // once, unless deep supervision
```

F15 is `input_blocks = 0`, `core_blocks = 8`, `output_blocks = 0`, R = 1:

```
y = Core[8](inject_norm((1+α) x))
```

Eight distinct blocks, all after the injection norm. `α` is almost invisible to the forward pass because RMSNorm removes the scale.

R15 at R = 1 is `2 + 4 + 2`:

```
h = Input[2](x)
y = Output[2](Core[4](inject_norm(h + α x)))
```

Two blocks run before the injection. The injection mixes their output with the raw embedding. Four shared core blocks follow. Two output blocks are outside the loop. No assignment of F15's eight core weights onto R15's eight blocks makes these the same function, because the injection sits at a different depth.

R15 at R = 2 or 4 is not "R1 repeated". Each extra iteration re-injects `α x` and reruns the four shared core blocks. Output blocks still run once. Executed blocks are `2 + 4R + 2`: 8, 12, 20. That matches the parity test's accounting and is the right cost model. It is iterative refinement with a persistent input skip.

`f15_r15_parity.rs` proves three things, and the comments already say so:

- both store 15,154,120 unique parameters
- executed-block formulas
- `R15.forward_r(R=1)` equals `R15.forward_control` on that same R15 module

It does not compare an F15 module to an R15 module. H1 will add a short comment in that test stating the non-claim, so a later reader does not promote the parameter match into a functional match. No weight-tying experiment is added to force them together.

### What each comparison answers

Keep these apart. They are different questions.

| Comparison | Question it answers |
|---|---|
| F15 vs R15 trained at R=1 | Does the recurrent-family layout differ from a matched-parameter feed-forward stack, under the same data and optimizer? |
| R15 trained separately at R=1, R=2, and R=4 | Does training at a higher recurrence adapt the shared core in a way that shows up in play? This is the clean recurrence comparison. |
| One trained R15 checkpoint, evaluated at R=1, 2, and 4 | Does extra iteration at inference time refine a network that was not retrained for that depth? |
| One R15 network trained with R sampled per batch from {1, 2, 4} | Can one set of weights learn to use a variable amount of internal compute? The learner cannot do this yet. |
| F15 or R15-R1 with more search, versus R15-R2/R4 with less search | Does internal recurrence beat spending the same move-time or the same neural compute on external search? |

H1 does not run any of these. H1's job is to make an F15 learning run something those later comparisons can stand on. The recurrence control for the later claim is R15 R1 vs R15 R2 vs R15 R4, trained as separate conditions. F15 stays the feed-forward architecture control beside that ladder, not a stand-in for R15 at R=1.

## 6. P0 fixes before any F15 learning

All of these land, with CPU tests, before the scheduling sweep. The sweep must measure the collector the smoke will use.

1. One collector. `games_per_cycle` and `concurrent_games`. `cpu_workers` stops being a silent no-op. Pilot reports `SelfPlayMetrics` and inference metrics. Play failures are visible.
2. Update count comes from new trainable positions and `replay_reuse_target`, with a cap, a floor, and a written reason when the target is missed.
3. The next cycle loads the promoted checkpoint's weights, Adam moments, and schedule step. A rejected candidate does not advance the parent's schedule.
4. Lineage records parent, replay generator, and candidate as three ids captured before promotion. Git SHA and branch are filled.
5. Promotion cannot succeed on a draw-only or all-truncated arena. Parent comparison and reference comparison are separate results.
6. A requested opening suite or `start_fen` that is missing or illegal fails the process.
7. Accumulation loss is the whole effective batch. Gradients are mean-reduced. Clip scope is labeled.
8. `scientific_config_hash` and `resolved_config_hash` both exist.

Files and tests are in the tickets. Existing Phase 0/1/2 tests stay enabled and must pass.

## 7. P1 qualification improvements

- HP sweep grid and a real VRAM sampler, on the unified collector. Section 14.
- Pre-learning search budget on frozen random F15. Section 15.
- Raw-network candidate vs parent, no tree. Keep raw-vs-random as the weak anchor.
- Post-hoc search-target health from the stored visit distribution: mean entropy, top-1 share. This walks replay records after the game. It does not add work inside PUCT.
- Median game length, repetition share, draw share, on the same pass.
- `games_skipped` reported from the store.
- `timeout_flushes` included on `MetricsSnapshot`. The counter already exists and is not exported.
- Document the arena CI as an independent-game interval.

Per-batch recurrence sampling is not in this list.

## 8. P2, before any 24-hour run

Not required for the smoke or the qualification pilot.

Replay archive is crash-unsafe. `enforce_capacity` renames oldest shards into `replay/archive/` and only then calls `write_manifest_atomic`. A crash after the first rename and before the manifest rename leaves `manifest.json` naming files that are no longer beside it.

Replacement order:

1. Copy each doomed shard into `archive/` and fsync it. The original stays in place.
2. Write the new manifest (kept shards only) via the existing temp-file plus rename, and fsync.
3. Delete the originals that the new manifest no longer names.

A crash before step 2 leaves the old manifest and the original files. A crash during step 3 leaves extra copies in `archive/` and a manifest that only names the kept files. The reader never points at a missing shard. Add a test that simulates a crash after the copy and before the manifest swap, and one after the swap and before the delete.

The long-run gate, all of it required and none of it scheduled in H1:

- optimizer resume tested through the pilot path
- the crash-safe archive above
- bounded disk use (capacity already bounds the active set; archive growth still needs a policy)
- sustained thermals and VRAM from a periodic sampler, not a boundary sample
- remote recovery and Windows restart recovery actually tried
- atomic checkpoints at a defined interval
- a qualification pilot that already passed

24-hour run during H1: no.

## 9. Collection and concurrency

New request, one function, used by `run`, `collect_only`, `run_pilot`, and `run_cell`:

- `games`: how many games to try to finish
- `concurrent_games`: how many game threads
- `deadline`, `cancel`
- model snapshot directory (the caller loads it and owns the inference runtime)
- search config and the `SearchRecord` stamped into replay
- `game_id_base` and seed
- start position, already validated

Return value:

- replay records
- `SelfPlayMetrics` (terminations, W/D/L, truncated, mean and median plies, draw share, repetition share, failures)
- `MetricsSnapshot` (submitted, completed, errors, batch mean / p50 / p95 / max, queue wait p50 / p95, forward latency, `peak_in_flight`, timeout flushes)
- target-health summary computed from stored `PlyRecord.target` (entropy and top-1 share)

`repeated positions per game` is a FEN walk over the recorded moves after the game, on the CPU, outside search.

### Config resolution

New fields: `games_per_cycle: u32`, `concurrent_games: u32`.

`active_games` and `cpu_workers` become `Option`. Omitted means unset, not "32" and "8".

`RunConfig::resolve_collection()`:

- Both new fields set: use them. If `active_games` or `cpu_workers` is also set and disagrees with the new fields, the process errors and names the disagreement.
- Neither new field set, `active_games` set, `cpu_workers` unset: legacy coupling. Both the game count and the thread count become `active_games`. The resolved report records `collection_resolution = "legacy_active_games_coupled"`.
- `cpu_workers` set while `concurrent_games` is unset: error. The message says the field no longer sizes a thread pool and that `concurrent_games` is required. This is the explicit compatibility behavior. The HP pilot config hits this error until it is rewritten, which is what we want.
- Neither legacy nor new fields set: error. There is no silent default of 32 threads.

In-repo configs and the two PowerShell scripts are updated in the same ticket. `scripts/hp-concurrency-sweep.ps1` currently treats `--cpu-workers` as the thread count. After this change that script would be wrong; it is replaced by the HP grid on `bench-runtime` (section 14), not patched to keep the old meaning alive.

## 10. Replay and reuse

Denominators, all reported every cycle:

| Name | Definition |
|---|---|
| `new_positions` | Plies written this cycle, including truncated and aborted games |
| `new_trainable_positions` | Plies that belong to games with an outcome. Truncated and aborted games contribute zero. This is the reuse denominator. |
| `active_replay_positions` | Positions still named by the manifest |
| `active_trainable_positions` | `ReplayStore::sampleable()` |
| `examples_consumed` | Examples the learner actually ran forward/backward on |
| `new_game_example_fraction` | Share of consumed examples whose `game_id` was generated this cycle |

```
effective_batch = train_batch * accumulation_steps
desired_examples = new_trainable_positions * replay_reuse_target
desired_updates = ceil(desired_examples / effective_batch)
scheduled_updates = clamp(desired_updates, min_updates, max_updates)
```

`max_updates` becomes the safety cap. The HP smoke sets it high enough to hold the target and low enough that a bug cannot train for hours. `min_updates` defaults to 1 when `new_trainable_positions > 0`, otherwise 0.

The learner accepts the cycle deadline and stops between updates. Stopping early sets `reuse_shortfall_reason = "time_budget"`. Hitting the cap sets `"hit_max_updates_cap"`. Zero trainable positions sets `"no_trainable_positions"` and schedules 0 updates. A clean hit of the target sets the reason to empty and `reuse_target_met = true`.

`achieved_reuse = examples_consumed / new_trainable_positions` when the denominator is positive. Also log `examples_consumed / new_positions` so the truncated mass stays visible. A miss is a field in the cycle report, never a silent smaller `max_updates`.

`replay_reuse_target` stays 2.0 for the smoke, as an exposed config value, matching the F10 contract. It is not retuned to make the gate pass.

### Gradient reduction in the same learner change

Burn's accumulator sums. The H1 learner mean-reduces before the optimizer step:

- Each microbatch backward grad `g_i` is the mean grad of that microbatch.
- Accumulate `n_i * g_i`, then divide by `N = sum n_i`, where `n_i` is the microbatch example count.
- Equal-sized microbatches reduce to "divide the sum by the microbatch count".
- The loss, policy loss, WDL loss, and policy entropy logged on the update are the same `n_i`-weighted means. The last microbatch is not the update's loss.

Clip reporting, without changing Burn's clip:

- `grad_norm_pre_clip`: global L2 of the mean-reduced grads, before `optim.step`.
- `parameters_clipped`: how many parameter tensors have pre-clip L2 above 1.0, which is the condition under which Burn's per-tensor norm clip fires.
- `clip_scope = "per_parameter_l2"` and `clip_threshold = 1.0` on every update record.

The scientific LR is the LR of this mean-reduced effective batch. Physical batch size stays a hardware choice.

## 11. Optimizer and checkpoint continuity

The live trajectory is the last promoted checkpoint. The frozen reference directory is written once at the start and never replaced.

Cycle start:

- Load that checkpoint with `load_training` (weights, Adam record, `update_counter`, `lr_schedule_step`).
- Cycle 0 loads the reference, whose optimizer is the fresh AdamW saved at step 0. That reset is correct.
- Train `scheduled_updates` steps. `start_update` is the loaded schedule step, not a counter that includes rejected cycles.
- Save the candidate with `save_training`, including the optimizer and the new step.

Promotion copies the candidate directory onto `snapshot-CCC` and that directory becomes the next parent. Rejection leaves the parent directory untouched. The candidate directory stays on disk for the report. The next cycle loads the parent again, so the rejected Adam state and the rejected schedule steps are not part of the trajectory.

Test, CPU, deterministic Flex, through the pilot load path:

- Run A: N learner updates, one optimizer, no save.
- Run B: K updates, `save_training`, `load_training` the way cycle K+1 will, then N−K updates.
- Final loss and a recorded core weight agree within the tolerance the existing resume test already uses (loss 1e-4, weight 1e-5) or tighter if the run is bit-exact. Report which one happened. Do not loosen the existing test to make this pass.
- A second test: a cycle that does not promote leaves the parent's `lr_schedule_step` unchanged.

`f10_resume_preserves_optimizer_and_schedule` stays. It is necessary and not sufficient.

## 12. Promotion and evaluation

Two searched arenas per cycle, same openings, same budget, paired colors:

- Candidate vs parent. This is the only arena the promotion rule may read.
- Candidate vs frozen reference. Longitudinal. Recorded. Ignored by promotion.

Raw games, no search:

- Candidate vs uniform random. Kept. Weak anchor. Run once on the reference before cycle 0, and once on the candidate after every cycle.
- Candidate vs parent. New. W/D/L, score, termination counts. Plus a position diagnostic on the opening FENs: move agreement at temperature 0, policy entropy, top-1 overlap. No Stockfish, no LC0 labels, no human games.

### When an arena counts

`ArenaResult` gains `decisive_games` and `informative`. `candidate_score` becomes `Option<f64>`. It is `None` when no game has a result. The current code's substitution of 0.5 in that case is removed. The same change applies to raw-policy scores.

An arena is informative only when `decisive_games >= promotion_min_decisive_games`. That minimum is config. The smoke default is 4, written in the config file with a comment that it is a diagnostic floor, not an Elo sample. Below the floor the decision reason is `arena_uninformative`.

### Promotion rule

Every line is required:

- audit passed
- inference errors = 0 and collector failures = 0
- losses and pre-clip grad norm finite
- `updates > 0`
- reuse target met, or the shortfall reason is recorded and `allow_promotion_on_reuse_shortfall` is false (default false, so a shortfall does not promote)
- checkpoint reloaded by the continuity test's same function succeeds for this candidate
- parent arena is informative
- parent-arena score is greater than 0.5

`promotion_score_floor = 0.35` is removed from the HP configs. A score of 0.5 is a tie with the parent and does not promote. The floor field, if still present for old F10 configs, cannot by itself promote: the informative-arena requirement and the strict improvement over the parent are mandatory in the pilot. F10 toml files on this branch get a comment that they are workstation history; their numeric floor is not the HP rule.

Decision strings: `promote`, `hold`, `stop`. `hold` carries a reason enum the report prints: `arena_uninformative`, `score_not_above_parent`, `reuse_shortfall`, `non_finite`, `audit_failed`, `no_updates`.

Opening rule:

- `opening_suite = Some(path)`: file loads, version parses, every FEN parses, every position has a legal move list. Any failure aborts before cycle 0.
- `opening_suite = None`: evaluation uses the standard start. That is the only startpos fallback.
- `start_fen = Some`: one parse at startup. Failure aborts. It is not a per-game `continue`.
- Tests: invalid FEN errors, missing suite errors, malformed suite errors, omitted suite uses startpos.

## 13. Lineage and provenance

Before the cycle trains, freeze:

- `parent_snapshot_id` — checkpoint whose weights played the games
- `replay_generator_id` — same id, stored under its own name so a later change cannot collapse them
- `candidate_id` — assigned after the candidate save, from that checkpoint's `model_id`

The lineage row is written from those three, after the promotion decision, and also stores `promoted_snapshot_id` (the candidate id if promoted, otherwise the parent id) and `snapshot_decision`.

`git_revision` and `git_branch` come from the same env `RunMetadata` already uses. `ReplayHeader` and `CheckpointMeta` get the same values at construction time in the runtime. Schema versions stay. New lineage fields are optional with serde defaults so older `lineage.jsonl` rows still parse.

A unit test builds a two-cycle fixture that promotes on cycle 0 and checks `parent_snapshot_id != candidate_id`, `replay_generator_id == parent_snapshot_id`, and `git_revision` is `Some` when the env is present.

## 14. HP batching sweep

Do not run `grid()` or `grid(small)`. Those start at 32 concurrent games and go to 256. Add `grid_hp()` and a `--grid hp` flag so the workstation grid stays available for the other machine.

The sweep calls the unified collector. Each cell records terminations, not only batch stats.

### Matrix

Fixed during this sweep, so scheduling is the only thing that moves:

- F15, CUDA FP32, R = 1
- standard start
- `c_puct = 1.0`, temperature 1.0
- simulations held at 32 (a scheduling probe, not the learning budget)
- ply cap 80, 8 games per cell
- per-cell wall cap 90 seconds; a cell that hits it is `time_capped`, not a throughput winner

Grid:

| Axis | Values |
|---|---|
| `concurrent_games` | 6, 8, 12, 16, 24 |
| `max_inference_batch` | 8, 16, 32, 64 |
| `batch_timeout_us` | 500, 1000, 2000 |

60 cells. At the cap that is at most about 90 minutes. Coarse pass first if a full pass is too long: concurrent {6, 12, 24} × batch {16, 32, 64} × timeout {1000, 2000} (18 cells), then one ring around the leader.

VRAM: a thread samples `nvidia-smi memory.used` every 250 ms for the life of the cell and reports the max as `peak_vram_mb`. Sampler failure stores null and `vram_sample = "unavailable"`. GPU util and temperature are sampled the same way and stored beside throughput. They are not the objective.

### Stop conditions

- Another compute process is on the GPU (`nvidia-smi` compute apps): do not start.
- `peak_in_flight <= 1` or batch p50 <= 1 on a cell with `concurrent_games >= 6`: that cell is `FAILED`. If every cell in the coarse pass fails this way, stop the sweep. Do not pick a winner.
- CUDA OOM: that cell is `oom`. Do not retry it at a smaller silent batch. Keep going with the remaining cells.
- A temperature sample ≥ 90 °C: stop the sweep and write the rows collected so far.
- The process is cancelled: partial CSV/JSON is kept and marked `interrupted`.

### How a winner is chosen

Useful throughput is evaluations per second on cells that:

- finished without `oom`, `FAILED`, or `time_capped`
- have batch p50 ≥ 4
- have true peak VRAM ≤ 3,500 MiB (headroom under 4,096 for the later training step)
- prefer `concurrent_games <= 16` unless 24 beats the best ≤16 cell by more than 15% eval/s

`max_inference_batch` is a ceiling. If the leader's batch p95 is 12 under a cap of 64, the frozen cap is the smallest grid cap that is ≥ p95, not 64. "Batch 64 fits" is not a reason to select it. Direct F15 inference was already flat from batch 16 (639, 627, 632 ex/s).

Freeze the winner into `configs/hardware/hp-home.toml` and into the smoke config. Record the sweep directory, the cell name, and both config hashes in the smoke report. Until that freeze, the smoke config is not runnable: its preflight requires `scheduling_frozen = true` set by this step, not by hand in advance.

## 15. Search-budget sweep

After the scheduling freeze. Before any training. Frozen random F15, the frozen scheduling config, standard start, ply cap 400.

Budgets: 32, 64, 128. Same game count at each budget (16). Per-budget wall cap 20 minutes. A budget that hits the cap is reported as time-capped and is not selected just because it searched deeper.

Measure, from the unified collector: positions/s, evals/s, games completed, checkmate, stalemate, insufficient material, threefold, fifty-move, truncated, mean length, target entropy, top-1 visit share, batch p50 / p95, true peak VRAM.

Selection rule, written into the report before looking at a favorite:

- The goal is a budget whose visit targets are not degenerate, at a positions/s that leaves room for two smoke cycles inside 30 minutes.
- Degenerate means top-1 visit share above 0.9 on average, or threefold-plus-fifty-move share above 0.8, or target entropy near zero. Thresholds are written in the sweep report as the rule used, and a budget that trips them is `degenerate`.
- Among non-degenerate budgets, pick the one with the best positions/s. Do not pick the largest simulation count.
- If every budget is degenerate, freeze 32, mark the search gate `CONDITIONAL`, and say the smoke is a pathology check. Do not add 256. 256 is allowed only when 128 is non-degenerate, finishes inside the cap, and its positions/s still leaves a 30-minute smoke possible. That bar is unlikely; the default plan does not include 256.
- The chosen integer is written into the smoke config before the smoke process is started. Changing it later is a new `scientific_config_hash`.

## 16. F15 learning smoke

Starts only when sections 6, 14, and 15 are done and the search gate is GO or the explicit CONDITIONAL described above. Wall clock is a backstop. The real bounds are cycles and positions.

| Knob | Value |
|---|---|
| Model | F15, R = 1, CUDA FP32 |
| Start | Standard start. No `start_fen`. |
| Search | Frozen sims, `c_puct = 1.0`, temperature 1.0 for every ply, no Dirichlet, no Gumbel |
| Scheduling | Frozen concurrent games, batch ceiling, timeout |
| `games_per_cycle` | `clamp(round(10 min × measured positions/s / mean plies from the search sweep), 8, 24)` |
| Cycles | 2 |
| `position_budget` | 12,000 |
| `run_budget_minutes` | 30 |
| Ply cap | 400 |
| Replay | streaming store, capacity 100,000, reuse target 2.0 |
| Effective batch | 128, as physical 32 × accumulation 4, unless the frozen VRAM note says physical 32 does not fit beside the optimizer. Then physical 16 × accumulation 8, and the scientific hash still sees effective batch 128. |
| `max_updates` | 64, as a cap |
| `min_updates` | 1 |
| LR | 3e-4, warmup and cosine over `planned_updates` large enough for the pilot that might follow (set `planned_updates` to the qualification-pilot horizon, not to one smoke cycle, and record it) |
| Arena | 20 games, `configs/openings-v1.toml`, both opponent roles |
| Raw | 20 games vs random, and the parent raw match |
| Seed | 1 |
| Snapshot | Conservative, under the section 12 rule |

Before cycle 0 the run writes reference raw-vs-random and the two config hashes. Cycle 0's collection is the baseline self-play health sample, taken with the random init, and it is in the report before anyone reads the loss.

### Smoke GO

System: no illegal moves, no inference errors, no collector failures, no NaN/Inf, checkpoint reloads with `load_training`, the continuity relation from section 11 still holds on a CPU test that was run before this smoke, process RSS and the periodic VRAM sampler show no unbounded climb across the two cycles.

Data: both cycles finish at least the requested games or hit the position budget cleanly; truncation fraction is reported; termination counts are present; replay audit passes; target entropy and top-1 share are present. A repetition share above 0.8 with no written mechanism is NO-GO, not a pass.

Training: achieved reuse within 15% of 2.0, or a shortfall reason that is `time_budget` or `hit_max_updates_cap` and is discussed before anyone says GO. Policy loss and WDL loss are separate columns. Grads finite. `parameters_clipped` reported. A core weight in the candidate differs from the parent. Loss is finite at every update.

Evaluation: raw eval finishes. Each searched arena is marked informative or not from its decisive count. No strength sentence. An uninformative arena is acceptable for the smoke GO on evaluation health and is not acceptable as a promotion. Promotion may legitimately `hold` for the whole smoke. A `hold` is not a NO-GO. A `promote` on an uninformative arena is a NO-GO, because it means the gate is still wrong.

Anything else: NO-GO. Diagnose. Do not open the qualification pilot.

## 17. F15 qualification pilot

Only after a smoke GO. Same `scientific_config_hash` as the smoke. Same frozen sims and the same scheduling. A change to either is a different experiment and stops this pilot.

| Knob | Value |
|---|---|
| Cycles | 4 |
| `position_budget` | 40,000 |
| `run_budget_minutes` | 75 |
| `games_per_cycle` | the smoke value |
| Replay cap | 100,000 |
| Optimizer | continuous across promotions, as in section 11 |
| Purpose | The learning trajectory is stable enough that a later recurrence comparison would be interpretable |

Not a strength run. Not 24 hours. A repetition collapse, a non-finite update, a reuse miss with no reason, or a promotion that ignores the gate is NO-GO for R15 entry.

## 18. R15 entry gate

H1 does not train R15 and does not start an R15 self-play pilot. `configs/hp/r15-pilot.toml` stays unused.

At the end of H1, one of these labels is written into the smoke/pilot report. The label is about readiness to design the recurrence runs, not permission to launch them inside H1.

**R15 GO** requires all of:

- F15 smoke GO and qualification-pilot GO
- data health not degenerate, or the degeneracy is confined to a measured search budget that the frozen choice avoided
- reuse under the controller, with shortfalls explained
- loss and gradients stable enough that a curve can be read (finite, both heads present, weights moving, clip behavior reported)
- raw-vs-random not in a monotone collapse across the pilot
- parent vs reference evaluations both present and the promotion rule matched the section 12 code
- lineage parents checked on the real run's `lineage.jsonl`, not only in the unit test
- the section 5 interpretation written next to the result: F15 is the feed-forward control; the recurrence contrast is R15 at R=1, R=2, and R=4

**R15 CONDITIONAL GO** is only for: systems, optimizer, lineage, and eval machinery all passed, and the remaining issue is a single named data-quality problem with one proposed next measurement. CONDITIONAL GO does not authorize R15 training.

**R15 NO-GO** when any system gate failed, when the smoke was not run, or when raw policy collapsed. The interestingness of recurrence is not a reason to override this.

## 19. Future recurrence experiment

Not implemented in H1. The learner stays at one `recurrence: usize`.

Ladder, in order, each one isolated:

1. **Matched-data, separate training.** Freeze one self-play corpus. Train F15, R15-R1, R15-R2, and R15-R4 from controlled inits, same example count, same effective batch, same AdamW family, same scientific schedule. Several seeds when the machine can afford them. This is the first recurrence result. R15-R1 vs R15-R2 vs R15-R4 is the recurrence contrast. F15 vs R15-R1 is the architecture-class contrast.
2. **Same checkpoint, several depths.** Take the R15-R1 checkpoint and evaluate it at R=1, 2, and 4 with no further training. That measures inference-time refinement only. Do not describe it as the training result from step 1.
3. **Multi-R training.** Only after 1 and 2 exist. Sample `R ∈ {1, 2, 4}` per physical batch from a declared distribution (uniform is a choice, not a default buried in code). The schedule is deterministic given the seed, and every update record stores the R it used. This needs a `RecurrenceSchedule` on `LearnerConfig` and provenance in `TrainReport`. Deep supervision stays a separate flag, default off, because it changes both the loss average and the block count.
4. Online self-play per condition comes after the frozen-data ladder, because online play confounds the net with the data it generates.

Distribution weights, if step 3 is ever run, are an experimental config with their own scientific hash. There is no hidden default.

## 20. Matched-data control

Before any online F15-versus-R15 loop:

1. Generate one Recur64 self-play corpus with the frozen F15 (or, if the question is within R15, with a frozen R15-R1 generator — pick one generator and do not mix).
2. Freeze the directory: manifest checksum, game count, trainable-position count, generator model id, git SHA, scientific hash.
3. Train each condition from that directory only. The training code path for this phase must be able to refuse to start self-play.
4. Same examples, same effective example count, same optimizer family, same LR schedule.
5. Initialization: either a shared seed per paired run, or an explicit mapping. Do not copy F15 weights into R15. They are not the same function.
6. Seeds: one seed is a pilot of the ladder; claims wait for more than one seed.

H1 only writes this design. It does not build the frozen-corpus trainer unless a P0 fix needs the replay loader, which it does not. The existing `train_from_store` is the tool a later ticket will point at a frozen directory.

## 21. Matched-compute control

The central later question: does internal recurrence beat spending the same compute on search?

Primary fairness: matched wall clock per move.

Secondary, reported separately: matched neural compute.

Account, per move, for F15, R15-R1, R15-R2, and R15-R4:

- GPU inference time
- executed transformer blocks (`2 + 4R + 2` for R15 final-output, 8 for F15)
- neural evaluations
- CPU time inside search
- total move wall time

Equal simulation counts are the wrong fairness criterion. R15-R4 costs 20 blocks per evaluation against F15's 8, so the same node budget gives R15-R4 more neural work. The search-budget sweep's positions/s and evals/s are the prototype of this accounting. H1 records the fields. H1 does not run the comparison.

## 22. Test plan

CPU tests, real collector and real learner, no mocks in place of those paths. Existing tests stay enabled.

| # | Test |
|---|---|
| 1 | `games_per_cycle = 4`, `concurrent_games = 2` collects 4 games with `peak_in_flight` reflecting 2 threads, on the CPU micro model |
| 2 | Pilot's collect call is the shared function. A grep-level assertion is not the test; the pilot report contains `SelfPlayMetrics` produced by that function |
| 3 | Those metrics survive into the serialized pilot report, including terminations and inference batch p50 |
| 4 | Reuse arithmetic: 100 new trainable positions, target 2.0, effective batch 32 → 7 updates. Cap 4 → 4 updates and reason `hit_max_updates_cap`. Zero trainable positions → 0 updates and `no_trainable_positions`. Truncated plies do not enter the denominator |
| 5 | Optimizer continuity through `load_training`, section 11 |
| 6 | LR at update K+1 after a save/load equals `lr_at` of the continuous run. A non-promotion does not move the parent's schedule step |
| 7 | Promoting cycle: `parent_snapshot_id != candidate_id`, replay generator equals the parent |
| 8 | Lineage `git_revision` and `git_branch` are `Some` when the build env is set |
| 9 | Draw-only arena (score 0.5) does not promote. All-truncated arena (`candidate_score = None`) does not promote |
| 10 | Invalid opening FEN returns an error |
| 11 | Missing suite path returns an error. Malformed suite returns an error. Omitted suite evaluates from startpos |
| 12 | Two microbatches: logged loss equals the weighted mean, not the second loss. Mean-reduced grad norm is about half the summed grad norm on equal batches |
| 13 | Changing only `concurrent_games` or `batch_timeout_us` keeps `scientific_config_hash` and changes `resolved_config_hash` |
| 14 | Changing `simulations_per_move` or `c_puct` changes the scientific hash |
| 15 | Parity test comments and assertions match section 5: parameter count, block accounting, R15-vs-itself at R=1. No assertion that F15 equals R15 |
| 16 | Cancel during collect: status `interrupted`, reference checkpoint still loads, no partial manifest names a missing shard |
| 17 | `cargo test --workspace` for the existing Phase 0/1/2 suites, plus fmt and clippy, after the changes |

Also: `cpu_workers` set without `concurrent_games` is an error. Legacy `active_games` alone still runs and records the coupling. A bad `start_fen` aborts.

GPU smoke and the two sweeps are not unit tests. They are gated runs with the stop conditions in sections 14–16. Their logs are kept. A failed cell stays in the table.

## 23. Documentation changes

- `docs/HP_EXPERIMENT.md`: the branch contains Phase 3 via `fa66c32`. `78be205` is the fork point, not the current base. Remove the "CUDA smoke in progress" sentence; the status log already says it passed. Describe F15 and R15-R1 as different functions, in the words of section 5. Mark the 334.6 cell as a pre-merge, short-game prior. Point the next step at this plan instead of "run `configs/hp/f15-pilot.toml`" as if that file were ready.
- `docs/HP_CHANGES.md`: add an H1 entry when the code lands. Until then, do not claim the pilot shares the collector.
- `docs/F10_BASELINE.md`: add a banner that every number in it is the workstation RTX 2000 Ada F10 run. Do not edit the measurements.
- `docs/STATUS.md`: the Phase 3 section needs the same banner. HP F15 learning is not started.
- `README.md`: the experimental-branch note should say the branch was forked at Phase 2 and now contains the Phase 3 merge. The linker sentence still describes the workstation `rust-lld` setup; the HP machine uses MSVC. Say which machine the README's linker line applies to.
- `configs/hardware/hp-home.toml`: delete the claim that accumulation is unimplemented. Fill it only after the sweep, and only with measured values.
- `Grok-plan.md`: HEAD's file is empty and should not stay as an empty tracked placeholder. The working copy is this review brief, and it is not an accident. On implementation, move that text to `docs/HP_H1_BRIEF.md` if it is going to live in the repo, or leave it uncommitted. Do not delete the working copy to "clean up" the empty blob.

No new architecture document, and no edits to the master specification or the Phase 0 kickoff.

## 24. Risk register

| ID | Risk | If it happens |
|---|---|---|
| R1 | Mean reduction makes H1 curves look unlike F10 | Expected. Report clip counts. Do not retune LR during the smoke to chase F10's shape. |
| R2 | Reuse ratio looks healthy while new games are rarely sampled | Read `new_game_example_fraction`. A low fraction with a high ratio is a CONDITIONAL, not a silent pass. |
| R3 | All search budgets are repetition draws | Freeze 32, search gate CONDITIONAL, smoke is diagnostic. No 256. No R15. |
| R4 | Someone treats F15 as R15-R1 | Section 5 is the interpretation that ships with the result. The parity test comment repeats it. |
| R5 | Deep supervision gets switched on to "use the loss" | Out of H1. It changes block counts. |
| R6 | Arena CI quoted as a recurrence result | H1 reports are diagnostics. Paired bootstrap is a later ticket. |
| R7 | Power loss corrupts replay on a long run | P2 archive order before any 24 h run. Smoke is short and still audits. |
| R8 | Sweep winner copied from the 334.6 cell | Forbidden. The new grid produces its own winner. |
| R9 | Pilot "resume" tested only via the old in-memory test | Ticket H1-04 adds the load path the pilot uses. |
| R10 | Stale HP toml used for the smoke | Preflight requires `scheduling_frozen` and `search_frozen`. |
| R11 | GPU busy with another process | Sweep and smoke refuse to start. |
| R12 | Promotion gate weakened so the smoke "succeeds" | A hold is a valid smoke. Changing 0.5-does-not-promote is a plan change, not an implementation detail. |

## 25. Implementation tickets

### H1-01 — One collector, split game count from concurrency

- OBJECTIVE: `run`, `collect_only`, `pilot`, and `sweep` play games through one function. Game count and thread count are different fields. `cpu_workers` cannot sit in a config and be ignored.
- RATIONALE: B1, B2, B3, B13.
- FILES: `crates/recur64-runtime/src/coordinator.rs`, `pilot.rs`, `sweep.rs`, `config.rs`, `inference.rs` (export `timeout_flushes`), `crates/recur64-cli/src/phase2.rs`, `phase3.rs`, `configs/hp/*.toml`, `configs/smoke.toml`, `configs/smoke-cuda.toml`, the f10 configs that set the old fields, `scripts/hp-concurrency-sweep.ps1` (retire its worker loop or make it call `--grid hp` only after H1-11).
- IMPLEMENTATION: section 9. Shared `collect` returns records, health, and inference metrics. Resolution rules for the legacy fields. Invalid `start_fen` aborts. `play_game_from` errors increment failures and fail the cycle when nonzero.
- TESTS: rows 1, 2, 3, 16 of section 22. Legacy coupling test. `cpu_workers`-without-`concurrent_games` error test.
- METRICS: `peak_in_flight`, games requested vs games written, failure count.
- DEPENDENCIES: none.
- FAILURE MODES: a second collect loop left in the pilot; a default of 32 restored by `#[serde(default)]` on the old field.
- GO CONDITION: the tests pass and a text search shows one game-thread spawn site.
- STOP CONDITION: if unifying the sweep would change workstation `grid()` results' meaning, stop and keep `grid()` behavior, adding `grid_hp` in H1-11 instead. Do not silently change the 32/64/128 grid.

### H1-02 — Pilot report carries self-play health

- OBJECTIVE: `CycleReport` includes terminations, W/D/L, truncation, mean and median length, draw share, repetition share, inference batch stats, queue wait, forward latency, errors, target entropy, top-1 share.
- RATIONALE: B1. The smoke gate is unenforceable without these fields.
- FILES: `pilot.rs`, `coordinator.rs` (metric struct), report writer.
- IMPLEMENTATION: embed the collector result. Target entropy is a CPU pass over `PlyRecord.target`.
- TESTS: row 3. A fixture game with a known visit distribution produces the expected entropy.
- METRICS: the fields above, present in `pilot.json`.
- DEPENDENCIES: H1-01.
- FAILURE MODES: entropy computed inside PUCT and slowing search. Keep it post-hoc.
- GO CONDITION: a CPU pilot writes every field.
- STOP CONDITION: if median or repetition requires storing new replay fields, stop and compute them from the existing ply list. Replay V1 stays.

### H1-03 — Reuse controls the update count

- OBJECTIVE: `scheduled_updates` is the formula in section 10. Misses carry a reason.
- RATIONALE: B4. F10 measured 0.07–0.13 against a target of 2.0 because `max_updates` was the workload.
- FILES: `config.rs`, `learner.rs`, `pilot.rs`.
- IMPLEMENTATION: section 10. `max_updates` is the cap. Deadline checked between updates. Report every denominator.
- TESTS: row 4.
- METRICS: requested reuse, achieved reuse, both denominators, shortfall reason, `new_game_example_fraction`.
- DEPENDENCIES: H1-01 for the new-position counts.
- FAILURE MODES: using all new plies, including truncated games, as the denominator. Using the cap as the workload again.
- GO CONDITION: the arithmetic tests pass and the pilot log shows `scheduled_updates` derived, not copied.
- STOP CONDITION: if the sampler cannot say which game an example came from without a replay-schema change, report the fraction from `game_id` already on `GameRecord`. Do not bump `REPLAY_SCHEMA_VERSION`.

### H1-04 — Optimizer trajectory survives promotion

- OBJECTIVE: a promoted cycle continues weights, Adam moments, and the LR step. A rejected cycle continues the parent, not the rejected step count.
- RATIONALE: B5.
- FILES: `pilot.rs`, `model_io.rs` only if a helper belongs there. `checkpoint.rs` load/save already exist; call them.
- IMPLEMENTATION: section 11.
- TESTS: rows 5 and 6.
- METRICS: `optimizer_step_start` and `optimizer_step_end` match the loaded and saved meta.
- DEPENDENCIES: none strictly. Lands before the smoke, after or beside H1-03.
- FAILURE MODES: loading the optimizer onto a freshly built module that did not take its `ParamId`s from the file. The test must use `load_training`. Keeping the old `f10_resume` test and calling the job done.
- GO CONDITION: continuous N updates match K + save + load + (N−K) within the stated tolerance, through the pilot's load function.
- STOP CONDITION: if Burn 0.21 cannot restore moments across `load_training` on a module built with `ProbeModel::new`, stop and report that. Do not invent a side-channel moment format.

### H1-05 — Lineage chronology and git SHA

- OBJECTIVE: parent, generator, and candidate are distinct facts. Git SHA and branch show up on lineage, replay headers, and checkpoints.
- RATIONALE: B6, B7.
- FILES: `pilot.rs`, `run_dir.rs`, `replay/schema.rs` (fill the existing `Option`, no version bump), call sites that build `CheckpointMeta`.
- IMPLEMENTATION: section 13.
- TESTS: rows 7 and 8.
- METRICS: the three ids and the SHA on the JSONL row.
- DEPENDENCIES: H1-04, because the candidate id comes from the saved checkpoint.
- FAILURE MODES: writing the row after `snapshot_model_id` is reassigned. Bumping the checkpoint schema for a field that already exists (`git_revision`).
- GO CONDITION: a promoting fixture has parent ≠ candidate, and the SHA is `Some`.
- STOP CONDITION: none anticipated.

### H1-06 — Promotion gate and split evaluations

- OBJECTIVE: draw-only and all-truncated arenas cannot promote. Parent and reference are different matches.
- RATIONALE: B8.
- FILES: `pilot.rs`, `arena.rs`, `config.rs`, `configs/hp/f15-pilot.toml`, `configs/hp/r15-pilot.toml`.
- IMPLEMENTATION: section 12, except the raw-vs-parent match, which is H1-07.
- TESTS: row 9.
- METRICS: `informative`, `decisive_games`, `candidate_score` as `Option`, decision reason.
- DEPENDENCIES: H1-02 for the report shape.
- FAILURE MODES: leaving 0.35 as a sufficient condition. Using the reference arena for promotion after cycle 0.
- GO CONDITION: the two unit cases refuse promotion, and a decisive win against the parent still can promote when the other health bits are set in the test.
- STOP CONDITION: if `Option<f64>` breaks too many downstream readers, add `informative: bool` and stop emitting 0.5 for the empty case. Do not keep the fake 0.5.

### H1-07 — Openings fail visibly, and raw candidate vs parent exists

- OBJECTIVE: bad suite, bad FEN, missing file: process error. New raw match: candidate vs parent, no search.
- RATIONALE: B9. Raw-vs-random cannot carry a later recurrence claim.
- FILES: `openings.rs`, `arena.rs`, `eval_policy.rs`, `pilot.rs`, `phase3.rs`.
- IMPLEMENTATION: section 12 opening rule, plus `raw_policy_match` between two evaluators. Position diagnostic on the suite FENs: agreement, entropy, top-1 overlap.
- TESTS: rows 10 and 11. One raw match on a fixture where both sides are the same net and the score is a draw-heavy result without a startpos substitution.
- METRICS: raw W/D/L both ways, agreement, entropy.
- DEPENDENCIES: H1-06 for the report slot.
- FAILURE MODES: `unwrap_or(startpos)` left on one of the three call sites. Stockfish or an external label introduced to make the metric "better".
- GO CONDITION: the three negative tests error, and the omitted-suite test uses startpos.
- STOP CONDITION: if the position diagnostic needs stored policy vectors the replay does not have, run it live on the opening FENs only. Do not widen Replay V1.

### H1-08 — Effective-batch metrics and mean reduction

- OBJECTIVE: the logged loss describes the optimizer step. The step sees the mean grad of the effective batch. Clip scope is written down.
- RATIONALE: B10.
- FILES: `learner.rs`, `train.rs` only for comments on `adamw` if the clip note belongs next to `GradientClippingConfig::Norm(1.0)`.
- IMPLEMENTATION: section 10, gradient paragraph. Do not replace Burn's per-tensor clip with a global clip.
- TESTS: row 12.
- METRICS: weighted losses, `grad_norm_pre_clip`, `parameters_clipped`, `clip_scope`, `clip_threshold`.
- DEPENDENCIES: none.
- FAILURE MODES: dividing by `accumulation_steps` even when the last microbatch was short. Logging the pre-clip norm of the sum and calling it post-clip.
- GO CONDITION: the half-norm test and the mean-loss test pass.
- STOP CONDITION: if a weighted mean cannot be formed because Burn grads are opaque, stop and report that, and do not claim the LR is an effective-batch LR.

### H1-09 — Scientific hash vs resolved hash

- OBJECTIVE: scheduling changes do not look like a new experiment. Science changes do.
- RATIONALE: B11.
- FILES: `config.rs`, report metadata.
- IMPLEMENTATION: `scientific_config_hash` covers model geometry, recurrence, simulations, `c_puct`, temperature, `argmax_after_ply`, ply cap, precision, LR, warmup, `planned_updates`, effective batch (the product), reuse target, replay capacity, seed, snapshot policy, promotion settings, opening-suite path, and the AdamW constants the code actually uses (weight decay 1e-4, clip threshold, per-tensor scope). `resolved_config_hash` remains the hash of the full resolved config. Hardware-only fields: device, `concurrent_games`, `games_per_cycle`, `max_inference_batch`, `batch_timeout_us`, physical `train_batch`, `accumulation_steps`, `hardware_profile`. Physical batch is hardware because the product is what the scientific hash sees.
- TESTS: rows 13 and 14.
- METRICS: both hashes in `metadata.json` and in every lineage row (`config_hash` becomes the scientific one; `resolved_config_hash` is added).
- DEPENDENCIES: H1-01 so the new field names exist.
- FAILURE MODES: hashing `hardware_profile` into the scientific id. Hashing the physical batch into the scientific id and thus calling a VRAM split a new experiment.
- GO CONDITION: the two tests pass.
- STOP CONDITION: none.

### H1-10 — HP scheduling sweep

- OBJECTIVE: measure the RTX 2050 envelope on the unified collector and freeze a scheduling profile.
- RATIONALE: B12, issue 13. The pre-merge 12-worker cell is a prior.
- FILES: `sweep.rs` (`grid_hp`, periodic VRAM sampler), `bench_runtime.rs` (`--grid hp`), `configs/hardware/hp-home.toml`, smoke config preflight.
- IMPLEMENTATION: section 14.
- TESTS: CPU test that `grid_hp()` equals the stated matrix and that `grid(false)` is unchanged. The sampler is tested by a fake clock only if that can be done without hiding `nvidia-smi`; otherwise the GPU log is the test, and a unit test covers "sampler failure stores null".
- METRICS: evals/s, positions/s, wall, batch p50/p95/max, peak VRAM, temperature, `peak_in_flight`, terminations.
- DEPENDENCIES: H1-01, H1-02, H1-09. GPU idle.
- FAILURE MODES: selecting batch 64 because VRAM fit. Treating a boundary sample as a peak. Running while another process owns the GPU.
- GO CONDITION: a winner exists under the section 14 rule, or the sweep stops on a stated stop condition with the table intact.
- STOP CONDITION: section 14 stop list. Also stop if `peak_in_flight` never exceeds 1.

### H1-11 — Search-budget qualification

- OBJECTIVE: freeze 32, 64, or 128 (256 only under the written bar) on random F15 before learning.
- RATIONALE: deeper search can amplify a bad value prior. F10 already saw that.
- FILES: a small driver, or `bench-runtime` cells with a sims axis. `configs/hp/f15-pilot.toml` receives the frozen sims only after the result. `scripts/hp-search-budget.ps1` is rewritten to the unified flags and the 32/64/128 list, or deleted if the Rust driver replaces it.
- IMPLEMENTATION: section 15.
- TESTS: the degeneracy rule has a pure function test: a fixture distribution is classified `degenerate` or not from entropy, top-1, and draw share, with the thresholds passed in.
- METRICS: section 15 list.
- DEPENDENCIES: H1-10. GPU idle.
- FAILURE MODES: adding 256 by default. Freezing 128 because it is larger. Changing sims again after the smoke has started.
- GO CONDITION: a frozen integer and a written classification of each budget.
- STOP CONDITION: section 15. If all three are degenerate, freeze 32 and mark CONDITIONAL. Do not keep searching for a non-degenerate budget in this phase.

### H1-12 — F15 smoke

- OBJECTIVE: the bounded run in section 16, then a GO or NO-GO.
- RATIONALE: the first F15 learning numbers that the harness can support.
- FILES: `configs/hp/f15-smoke.toml` (new), pilot preflight.
- IMPLEMENTATION: section 16. Preflight requires `scheduling_frozen`, `search_frozen`, CUDA, FP32, F15, and an idle GPU.
- TESTS: preflight unit tests for the refusal cases (not frozen, wrong precision, suite missing). The smoke itself is the run.
- METRICS: the full cycle report.
- DEPENDENCIES: H1-03 through H1-09, H1-11.
- FAILURE MODES: extending the budget because the machine is free. Promoting on 0.5. Claiming chess strength from 20 games.
- GO CONDITION: section 16 GO list.
- STOP CONDITION: any NO-GO item. Stop the process on NaN/Inf, audit failure, or OOM. Do not restart with a weaker gate.

### H1-13 — F15 qualification pilot

- OBJECTIVE: section 17, only after a smoke GO.
- RATIONALE: stability of the learning behavior, still bounded.
- FILES: `configs/hp/f15-pilot.toml` rewritten from the frozen smoke config. The current file is not a starting point for numbers; its 32/8 coupling and 0.35 floor are known-bad.
- IMPLEMENTATION: section 17.
- TESTS: scientific hash equals the smoke hash. A test can construct both configs and compare.
- METRICS: same cycle report, four cycles or a budget stop.
- DEPENDENCIES: H1-12 GO.
- FAILURE MODES: retuning sims or batch size between smoke and pilot. Letting it run past 90 minutes.
- GO CONDITION: section 17 purpose met, gates still hold.
- STOP CONDITION: smoke was not GO. Any non-finite update. Wall clock 75 minutes.

### H1-14 — Docs

- OBJECTIVE: section 23.
- RATIONALE: the branch currently describes itself as independent of Phase 3, and F10 workstation numbers sit where an HP reader will see them.
- FILES: the docs named in section 23. Move the brief only if it is being kept.
- IMPLEMENTATION: edits after the code tickets, so the docs describe the code that landed. The F15-vs-R15 paragraph can be written as soon as H1-05's parity comment lands.
- TESTS: none beyond reading the diff. No number in `F10_BASELINE.md` changes.
- METRICS: none.
- DEPENDENCIES: the tickets whose behavior the docs describe.
- FAILURE MODES: deleting the uncommitted brief. Pasting HP bench numbers into the F10 document.
- GO CONDITION: a reader can tell which numbers were measured on which machine.
- STOP CONDITION: do not revise the master spec.

### H1-15 — Crash-safe replay archive (P2)

- OBJECTIVE: section 8 order: copy, fsync, manifest swap, then delete.
- RATIONALE: R7. Required before a 24 h run. Not required for the smoke.
- FILES: `replay/sampler.rs` `enforce_capacity`, `replay` tests.
- IMPLEMENTATION: section 8.
- TESTS: crash-window simulations using a temp directory.
- METRICS: none.
- DEPENDENCIES: none, but it is sequenced after the pilot decision so it cannot delay the smoke.
- FAILURE MODES: deleting originals before the new manifest is durable.
- GO CONDITION: both crash-window tests leave a readable manifest.
- STOP CONDITION: do not start this ticket during the smoke.

### H1-R — Next phase, not H1

- OBJECTIVE: record the recurrence ladder and the missing `RecurrenceSchedule`. Do not write the scheduler.
- RATIONALE: issue 15. Building it now is scope expansion.
- FILES: none in H1. A later ticket will touch `learner.rs` and `TrainReport`.
- IMPLEMENTATION: section 19, when that phase is approved.
- TESTS: none now.
- METRICS: none now.
- DEPENDENCIES: R15 GO from section 18, plus a new approval.
- FAILURE MODES: sampling R inside H1 "so the plumbing exists".
- GO CONDITION: not part of H1.
- STOP CONDITION: any implementation of per-batch R during H1, unless a correctness bug forces it. None does.

## 26. Execution order

1. H1-01 collector and config resolution.
2. H1-02 health metrics, H1-08 mean reduction, H1-09 hashes. These can proceed together after H1-01's config names exist; H1-08 has no dependency and may start immediately.
3. H1-03 reuse budget.
4. H1-04 optimizer continuity.
5. H1-05 lineage.
6. H1-06 promotion gate.
7. H1-07 openings and raw-vs-parent.
8. H1-14 docs for the code that has landed, including the F15/R15 comment and the parity-test note (test row 15).
9. Full CPU `cargo test --workspace`, fmt, clippy. This is the harness GO. Failures stay visible.
10. Stop. GPU work needs a free card. Check `nvidia-smi` for other compute processes. If the card is busy, do not start.
11. H1-10 HP sweep. Freeze scheduling.
12. H1-11 search budgets. Freeze sims.
13. H1-12 F15 smoke.
14. If and only if the smoke is GO: H1-13 qualification pilot.
15. Write the R15 label from section 18. Stop. Do not train R15.
16. H1-15 only when a 24 h run is being considered later. Not in this sequence's critical path.

## 27. Gate tree

```
H1 HARNESS GO
  CPU tests in section 22 pass, including the old Phase 0/1/2 suites.
  Collector, reuse, optimizer, lineage, promotion, openings, hashes, metrics.
        |
        v
HP SYSTEMS SWEEP GO
  Unified collector on the RTX 2050.
  A winner under the section 14 rule, or a stop with the table kept.
  Scheduling frozen. Scientific hash does not contain that choice
  except through effective batch and the other scientific fields.
        |
        v
SEARCH-BUDGET GO
  32 / 64 / 128 classified.
  One budget frozen.
  All-degenerate → CONDITIONAL, freeze 32, no 256.
        |
        v
F15 SMOKE GO
  Section 16. A hold on promotion is allowed.
  A bad gate, non-finite training, or unexplained repetition is NO-GO.
        |
        v
F15 PILOT GO
  Only after smoke GO. Same scientific hash. ≤ 75 minutes.
        |
        v
R15 ENTRY GO
  Section 18. This label ends H1.
  It does not start R15.
```

24-hour run: NO-GO on every branch of this tree.

## 28. Explicitly deferred

- R15 training, R15 self-play, and any R1/R2/R4 learning run
- Per-batch recurrence schedules and multi-R training
- Same-checkpoint R sweeps and the matched-data corpus
- Matched wall-clock recurrence-versus-search trials
- Paired bootstrap or any headline arena statistic beyond labeling the current CI
- Crash-safe archive implementation (designed in section 8, ticket H1-15, not on the smoke path)
- 24-hour runs, remote recovery, Windows-restart recovery
- Global gradient clipping as a replacement for Burn's per-tensor clip
- Gumbel, diffusion, SSRL, geometric attention, Stockfish labels, LC0 labels, human games, BF16, distributed training
- Changes to Observation V1, Action V1, rules profile V1, legal-move semantics, WDL perspective, PUCT backup signs, Replay V1 meaning, or checkpoint schema versions
- Deep supervision turned on
- Forcing F15 and R15-R1 into one function class
- Docker, driver changes, and paid compute

## Contracts that stay

Unless a later bug is demonstrated in them, H1 does not change observation V1, action V1, rules profile V1, legal move generation, WDL perspective, PUCT backup sign, Replay V1 record meaning, or checkpoint schema version. New report fields are additive. `candidate_score` becoming optional is a report-struct change with a test, not a chess-contract change.

## What approval means

Approval of this plan authorizes the CPU tickets H1-01 through H1-09 and H1-14, in the order above, on `experiment/hp-r15`.

It does not authorize starting the sweep, the search-budget run, the smoke, or the pilot until those CPU gates are green and a separate go is given to use the GPU. It does not authorize R15, a 24-hour run, or the deferred list.
