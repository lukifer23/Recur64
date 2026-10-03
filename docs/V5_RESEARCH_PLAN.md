# V5 fixed-graph reader research plan

Status: **PRE-REGISTERED 2026-10-03, before V5 implementation or measurement.**

Current execution status: acquisition correction implemented after the owner's
2026-10-03 delegation to choose next steps; fresh qualification pending. No drill,
training or DEV evaluation has run. The original failure is preserved in
`V5_ROOT_CAUSE.md`; the pre-pilot amendment below changes no training/evaluation
recipe, seeds, loss, practical gates or authorized research scope.

## Question

With identical weights and identical acquired states, does applying the same
relational evidence/hypothesis loop more times improve root decisions? Q is exact
StateQuery transitions and R is reader-loop count. Results are the surface A(Q,R),
not an undifferentiated thought budget.

## Data and custody

Only P25_DATA_V1 TRAIN is accepted: 44,332 positions, digest
`3b25dc8549dd2fc9d47c30e294c273b3306aecb3eba91b964715326ddf74f2e6`.
The inherited `v4_train_dev_v1` canonical partition must reproduce:

- FIT 39,929; sorted-ID digest
  `a01932d7db863fbd0d160bc04bd3589137449d2c1e33d408f8d5d3f3cb45f8f5`
- DEV 4,403; sorted-ID digest
  `f877219bc87d916ad8478a745f1572821c1a051337c3a071f2b37b9a5f83b899`

V4_TUNE_V1 and HOLDOUT_C are refused by every V5 command. DEV is never trained.

## Acquisition

`uniform_frontier_v1` samples the complete sorted unqueried frontier using the
episode RNG. `base_ranked_depth_v1` sorts root moves by frozen z0 descending,
ActionId ties ascending, then performs stable-hash child-ordered DFS to depth five
and at most five acquired edges per root branch. Both stop at Q8 or genuine
exhaustion. Root edges are depth one and consume Q and branch quota.

Evaluation seed is `0x7A50_E001`. Q8 is acquired once; Q2/Q4 are exact prefixes.
Episode identity contains position ID, schedule, run seed and occurrence ordinal,
never R, label, batch order, microbatch, thread scheduling or loop output.

### Pre-pilot depth-contract amendment, 2026-10-03

The original implementation incorrectly applied ranked DFS's depth-five cap to
uniform acquisition. Restore the complete uniform frontier and encode depths
1..16 with explicit non-overlapping one-hot/turn/slot/action fields. Retain ranked
depth five/five edges per branch. The acquired-graph subcontract and manifest are
version two, configuration digest
`d74109e229e49dc9962c348202db3527a5ce4c63da20efcd04a2bbe2577ff937`.
This is a correction before any drill or pilot result, not result-driven tuning.
Q16 remains limited to the conditionally authorized FIT engineering diagnostic;
no Q16 DEV or additional pilot is authorized. A same-weight FP64 numerical unit
reference diagnoses FP32 finite-difference roundoff; production/training and
measured qualification remain FP32. Fresh CPU/CUDA qualification is required.

## Training recipe

Seed 5301 only. FP32, `adamw-v1`, correct-set loss, linear warmup 80 then existing
cosine decay, no WDL objective, no DEV-driven selection.

- Stage A: random initialization, baseline only, 1,200 updates, peak LR 3e-4,
  effective batch 64, physical 2 x accumulation 32, Q0. Evaluate final DEV B0.
- Stage B: graph-free frozen baseline, fresh optimizer, 800 updates, peak LR
  3e-4, effective batch 36. Lexicographic condition order is schedule
  (`uniform_frontier`, `base_ranked_depth`), Q (2,4,8), R (1,2,4), with exactly
  two examples per condition and 18 persistent independent cell-balanced samplers.
  Preferred physical layout is 2 x accumulation 18; the only fallback is
  1 x accumulation 36, frozen before the pilot.

The engineering drill selects four stable-ID-hash FIT positions from each of the
six KQRvK/KRRvK x M1/M2/M3 cells. A frozen random base and disposable reader train
on both schedules (48 graphs/update) for at most 200 updates, Q8/R4, warmup 20,
cosine, peak LR 1e-3. It passes at finite loss and at least 20% mean set-loss
reduction unless initial loss is below 0.05. One fresh, otherwise identical Q16
diagnostic is allowed only after Q8 failure. Both failing stops the pilot.

## Evaluation and gates

Evaluate Stage B at update 0 and 800 only. At update 800 measure B0 and both
schedules at Q{2,4,8} x R{1,2,4} on all 4,403 DEV positions. At Q8/R1 and R4 also
measure returned-payload shuffle, relation-bias removal, no-hypothesis-feedback
and all-payload-null. Shuffle seed is `0x7A50_E002`; complete four-slot groups are
deranged between positions within family, mate depth and observed graph depth,
with documented widening and no self-mapping.

Composition seed is `0x7A50_E003`. Stable path hashes split queried nodes A/B;
an all-one-side split is replaced by hash-sort alternating assignment. Report
neither/A/B/both at R1/R4 on KQRvK M3. Fewer than two acquired nodes is
non-estimable. Q8/R8 is forward-only and runs only if every primary gate passes.

Primary inference averages the two schedule effects per position. Bootstrap:
20,000 SplitMix64 resamples, ranks 499/19,499. Seeds are `0x7A500101` loop,
`0x7A500102` B0, `0x7A500103` shuffle loss, `0x7A500104` interaction.

`PILOT_CANDIDATE` requires exactly the six gates in the owner authorization:
R4-R1 top1 >=.03 with CI >0; R4-B0 >=.03 with CI >0; shuffled-real set loss
>=.01 with CI >0; real-vs-shuffled loop interaction CI >0; neither schedule
negative on contrasts 1-3; and every engineering/integrity/accounting gate.

## Stop rules

No job exceeds two hours. Project longer stages first, then run deterministic
resumable chunks of at most 45 minutes. Missing exact data, invariant/gradient
failure, more than 10M parameters, failed drill, or unexpected execution failure
stops progression. A negative pilot is final for this authorization and is not
retuned.

## Future query controller (design only)

After a replicated reader result, a separate experiment may freeze the reader
and train a budget-conditioned frontier head against bounded downstream decision
improvement
`U_h(e|S) = L_set(S) - E[L_set(after e and h-1 charged queries)]` for horizons
1/2/4 under a fixed continuation policy. Counterfactual probe trajectories remain
isolated and fully charged; deployed state receives only selected returns. No
controller module, utility label, THINK/QUERY/STOP allocator, adaptive halting,
warm start, transposition merge, diffusion, expert system or self-play is built in
V5.0.

The full deferred target/input/isolation/accounting sketch is in
`docs/V5_QUERY_CONTROLLER_MEMO.md`.
