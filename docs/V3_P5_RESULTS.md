# V3 P5 results: clean six-run LR screen and frozen selection

**Status: P5 COMPLETE. P6 NOT RUN - awaiting owner review/approval.** HOLDOUT_C remains sealed and unevaluated.

All numbers below are read from the committed evidence in `docs/evidence/v3/` (six run summaries, the pairing check,
`v3-p5-lr-selection.json`, `v3-p5-selected-recipe.json`). Everything is TUNE-side recipe selection and diagnosis. Nothing
here is a Gate I/II/III result, and none of it is held-out performance.

## 1. What was run

- Frozen contract digest `105ac3133877f954ed00e6ce9caaadabf6d5da1cf78a7ab99d44a195d03009d2`, resolved layout micro 16 x
  accum 8 (effective batch 128, budget sequence [0,2,4,8,0,2,4,8]), 800 updates, warmup 80, objective = mean root-policy
  CE + 1.0 x mean selector NLL, teacher `proof_teacher_seeded_v1`, four `cell_balanced_v1` samplers, `adamw-v1`, FP32 CUDA.
- Six runs: LR {7.5e-5, 1.5e-4, 3e-4} x seed {5101, 5102}, in the preregistered order, each from a fresh empty run directory.
  **All six completed uninterrupted** (`exit 0`, `start_kind = fresh`, `start_update = 0`, zero resumptions, 800 updates each).
  Launched about 11:20, `ALL_DONE` before 13:18 (about 2 h in total, about 16.4 min training + 3.2 min of five TUNE
  evaluations per run).
- The earlier interrupted attempt (LR 7.5e-5, seed 5101, stopped at update 150) is **void and excluded**
  (`docs/evidence/v3/v3-p5-interrupted-attempt1.json`, V3-D20). As an independent integrity signal, the new seed-5101
  update-0 `S_run` (3.5532696635894676) is bit-identical to the voided attempt's update-0 value.
- Pairing integrity check (before selection, never used to rank): for each seed the three LRs' update-0 ACTIVE evaluations
  (`S_run` and all 24 cell CEs) are **bitwise identical** (max |diff| = 0.0; seed 5101 `S_run@0` = 3.5532696636, seed 5102 = 3.5546736513).
- Sampler exposure (persisted per run, validated): every run consumed exactly **25,600 examples at each of B0/B2/B4/B8**
  (800 x 32); the 15 cells of each budget are balanced to within one example. Exposure is reporting evidence, not a
  selection input.

## 2. The six runs and the frozen selection

`S_run` = mean over budgets {0,2,4,8} of the mean ACTIVE CE over the six TUNE cells at update 800 (24 equal values).

| LR | seed | eligible | S_run (update 800) | @0 | @200 | @400 | @600 | train wall | max sampled grad norm | sampled updates with grad>20 | mean policy / selector loss, updates 700-790 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 7.5e-5 | 5101 | yes | **1.7119** | 3.553 | 2.854 | 2.017 | 1.706 | 16.3 min | 118.3 | 8 of 80 | 0.980 / 1.451 |
| 7.5e-5 | 5102 | yes | **1.6810** | 3.555 | 2.962 | 1.968 | 1.747 | 16.2 min | 35.0 | 3 of 80 | 1.045 / 1.440 |
| 1.5e-4 | 5101 | yes | **1.5663** | 3.553 | 2.369 | 1.624 | 1.571 | 16.2 min | 17.4 | 0 of 80 | 0.918 / 1.224 |
| 1.5e-4 | 5102 | yes | **1.5477** | 3.555 | 2.785 | 1.593 | 1.574 | 16.3 min | 51.7 | 3 of 80 | 0.967 / 1.131 |
| 3e-4 | 5101 | yes | **1.5274** | 3.553 | 2.078 | 1.592 | 1.565 | 16.3 min | 28.4 | 1 of 80 | 0.928 / 1.124 |
| 3e-4 | 5102 | yes | **1.4923** | 3.555 | 2.625 | 1.676 | 1.548 | 16.6 min | 5.5 | 0 of 80 | 0.959 / 1.065 |

| LR | S_lr = mean(seed 5101, seed 5102) | eligible |
|---|---|---|
| 7.5e-5 | 1.6964 | yes (both seeds) |
| 1.5e-4 | 1.5570 | yes (both seeds) |
| 3e-4 | 1.5098 | yes (both seeds) |

**Selected by the frozen rule (lowest eligible `S_lr`, exact tie to the lower LR; no tolerance, no override): peak LR
0.0003 (3.0e-4).** Selected-recipe digest (without seed): `a069ba9d18befed65f970aca253b47780365fd7019be38283f270d79d6c1db33`. The rule was applied exactly
once; `v3-p5-lr-selection.json` and `v3-p5-selected-recipe.json` were written by `recur64 v3-p5 select`, which refuses to run twice.

Reading the screen honestly:

- `S_lr` falls monotonically with LR (1.696 -> 1.557 -> 1.510), and the order is the same in both seeds. The selected
  LR is **the largest candidate**, i.e. the edge of the preregistered grid; the screen cannot say whether a larger LR
  would do better, and the grid was not extended after the fact.
- The gap between the two best LRs (about 0.047 mean; 0.039 and 0.055 per seed) is modest next to the seed-to-seed spread
  at one LR (about 0.02-0.035), but it has the same sign in both seeds.
- `S_run` at updates 200/400/600 is not monotone for every run (e.g. LR 3e-4 seed 5102: 1.676 at 400, 1.548 at 600).
  No intermediate checkpoint was used; selection used update 800 only.

## 3. Training stability and throughput

- Every update of every run had finite loss and gradient norm; no health, query or checkpoint error occurred; no run was ineligible.
- Gradient-norm spikes were concentrated at the lowest LR and recovered within a few sampled steps; LR 3e-4 had one sampled spike (28.4, seed 5101). Sampled maxima (every 10th update only, so true peaks may be higher): LR 7.5e-5 seed 5101: 118.3; LR 7.5e-5 seed 5102: 35.0; LR 1.5e-4 seed 5101: 17.4; LR 1.5e-4 seed 5102: 51.7; LR 3e-4 seed 5101: 28.4; LR 3e-4 seed 5102: 5.5. Nothing here is a threshold; it is a TRAIN-side observation.
- Loss curves (policy / selector TRAIN loss, single sampled update each, hence noisy):

| run | u0 | u100 | u200 | u300 | u400 | u500 | u600 | u700 |
|---|---|---|---|---|---|---|---|---|
| LR 7.5e-5 s5101 | 3.44 / 3.84 | 1.78 / 2.12 | 1.61 / 2.03 | 1.41 / 1.89 | 1.14 / 1.62 | 1.27 / 1.51 | 1.08 / 1.66 | 1.08 / 1.42 |
| LR 7.5e-5 s5102 | 3.46 / 3.81 | 1.93 / 2.19 | 1.56 / 1.94 | 1.29 / 1.78 | 1.22 / 1.63 | 1.18 / 1.66 | 1.11 / 1.58 | 1.14 / 1.43 |
| LR 1.5e-4 s5101 | 3.44 / 3.84 | 1.79 / 2.16 | 1.60 / 1.99 | 1.05 / 1.63 | 1.05 / 1.47 | 1.18 / 1.37 | 1.01 / 1.36 | 1.03 / 1.25 |
| LR 1.5e-4 s5102 | 3.46 / 3.81 | 1.86 / 2.07 | 1.48 / 1.80 | 0.96 / 1.49 | 1.06 / 1.32 | 1.10 / 1.33 | 1.02 / 1.26 | 1.05 / 1.19 |
| LR 3e-4 s5101 | 3.44 / 3.84 | 2.42 / 2.95 | 1.38 / 1.77 | 1.06 / 1.49 | 1.02 / 1.30 | 1.15 / 1.19 | 0.98 / 1.25 | 1.02 / 1.11 |
| LR 3e-4 s5102 | 3.46 / 3.81 | 1.65 / 2.04 | 1.45 / 1.73 | 0.98 / 1.30 | 1.08 / 1.19 | 1.09 / 1.28 | 1.01 / 1.20 | 1.02 / 1.11 |

- **Throughput observation (owner follow-up):** about 1.1-1.2 s/update, process CPU about 1.5 cores, whole-machine CPU about
  12%, GPU 28-63% busy (mean about 52%), peak VRAM 2,993 MB. The run looks CPU-bound (per-update teacher simulation and exact
  StateQuery passes), but this was not profiled. Profiling, batching and optimisation should be done as a separate
  owner-approved ticket **before any future long run**; it must not change scientific semantics.

## 4. Selected LR (3e-4) at update 800: ACTIVE results by budget and cell

Each entry is `seed 5101 / seed 5102`. n = 750 per cell, 4,500 pooled. Chance top-1 per cell: KQRvK M1 0.042, M2 0.051, M3 0.060; KRRvK about 0.037-0.05.

### B0

| cell | top-1 | correct mass | CE | entropy |
|---|---|---|---|---|
| KQRvK M1 | 1.000 / 1.000 | 0.997 / 1.000 | 0.371 / 0.368 | 0.390 / 0.366 |
| KQRvK M2 | 0.711 / 0.696 | 0.516 / 0.528 | 1.707 / 1.715 | 1.747 / 1.709 |
| **KQRvK M3** | 0.449 / 0.489 | 0.340 / 0.364 | 2.441 / 2.395 | 2.169 / 2.142 |
| KRRvK M1 | 1.000 / 1.000 | 0.996 / 1.000 | 0.095 / 0.091 | 0.117 / 0.092 |
| KRRvK M2 | 0.712 / 0.727 | 0.572 / 0.605 | 1.548 / 1.495 | 1.416 / 1.350 |
| KRRvK M3 | 0.667 / 0.695 | 0.497 / 0.531 | 1.979 / 1.914 | 1.704 / 1.638 |
| pooled | 0.756 / 0.768 | 0.653 / 0.671 | 1.357 / 1.330 | 1.257 / 1.216 |

### B2

| cell | top-1 | correct mass | CE | entropy |
|---|---|---|---|---|
| KQRvK M1 | 1.000 / 1.000 | 0.999 / 1.000 | 0.370 / 0.369 | 0.371 / 0.365 |
| KQRvK M2 | 0.704 / 0.685 | 0.579 / 0.573 | 1.902 / 1.919 | 1.313 / 1.296 |
| **KQRvK M3** | 0.491 / 0.524 | 0.410 / 0.439 | 2.851 / 2.780 | 1.376 / 1.351 |
| KRRvK M1 | 1.000 / 1.000 | 0.999 / 1.000 | 0.093 / 0.092 | 0.099 / 0.091 |
| KRRvK M2 | 0.716 / 0.732 | 0.639 / 0.669 | 1.832 / 1.806 | 0.855 / 0.753 |
| KRRvK M3 | 0.656 / 0.695 | 0.588 / 0.634 | 2.393 / 2.333 | 0.887 / 0.793 |
| pooled | 0.761 / 0.773 | 0.702 / 0.719 | 1.574 / 1.550 | 0.817 / 0.775 |

### B4

| cell | top-1 | correct mass | CE | entropy |
|---|---|---|---|---|
| KQRvK M1 | 1.000 / 1.000 | 0.999 / 0.999 | 0.370 / 0.370 | 0.376 / 0.369 |
| KQRvK M2 | 0.703 / 0.687 | 0.567 / 0.581 | 1.918 / 1.930 | 1.335 / 1.252 |
| **KQRvK M3** | 0.491 / 0.524 | 0.414 / 0.442 | 2.928 / 2.818 | 1.276 / 1.298 |
| KRRvK M1 | 1.000 / 1.000 | 0.997 / 0.999 | 0.095 / 0.093 | 0.111 / 0.100 |
| KRRvK M2 | 0.716 / 0.732 | 0.627 / 0.671 | 1.861 / 1.819 | 0.880 / 0.733 |
| KRRvK M3 | 0.656 / 0.695 | 0.596 / 0.640 | 2.512 / 2.378 | 0.766 / 0.738 |
| pooled | 0.761 / 0.773 | 0.700 / 0.722 | 1.614 / 1.568 | 0.791 / 0.748 |

### B8

| cell | top-1 | correct mass | CE | entropy |
|---|---|---|---|---|
| KQRvK M1 | 1.000 / 1.000 | 0.999 / 0.999 | 0.370 / 0.370 | 0.372 / 0.368 |
| KQRvK M2 | 0.695 / 0.673 | 0.499 / 0.515 | 1.853 / 1.850 | 1.680 / 1.611 |
| **KQRvK M3** | 0.491 / 0.525 | 0.390 / 0.420 | 2.804 / 2.751 | 1.497 / 1.450 |
| KRRvK M1 | 1.000 / 1.000 | 0.998 / 0.999 | 0.094 / 0.093 | 0.105 / 0.097 |
| KRRvK M2 | 0.712 / 0.731 | 0.564 / 0.625 | 1.811 / 1.727 | 1.162 / 1.003 |
| KRRvK M3 | 0.653 / 0.695 | 0.571 / 0.615 | 2.457 / 2.340 | 0.901 / 0.868 |
| pooled | 0.758 / 0.771 | 0.670 / 0.696 | 1.565 / 1.522 | 0.953 / 0.900 |

Observations (TUNE, diagnostic):

- **KQRvK M3 (the cell the feasibility claim rests on):** top-1 about 0.49-0.52 at B2/B4/B8 versus 0.45-0.49 at B0; correct
  mass 0.39-0.44 versus 0.34-0.36 at B0; **CE is worse with queries (2.75-2.93) than at B0 (2.39-2.44)**.
- **The policy does not improve with budget in CE.** Pooled ACTIVE CE is lowest at B0 and higher at B2/B4/B8 for both
  seeds (macro CE B0/B2/B4/B8 = 1.357/1.574/1.614/1.565 and 1.330/1.550/1.568/1.522). Top-1 and correct mass rise slightly
  (mass 0.65 -> 0.70 -> 0.70 -> 0.67) while CE rises too, which means the budgeted policy is sharper (entropy 1.26 -> 0.82) but
  more confidently wrong on the cells it misses. M1 cells are solved at every budget (top-1 1.0); M2/M3 carry the error.
- This is a recipe-selection screen at 800 updates; it does not decide the B8 primary experiment, but it is not
  evidence that the active-search hypothesis is working yet.

## 5. Learned selector behaviour at update 800 (selected LR; diagnostics, not selection inputs)

Pooled over the 4,500 TUNE positions; per query unless stated. Entry `5101 / 5102`.

| B | proof-admissible | refute-admissible | off-target | first-query correct root (per position) | mean query depth | depth histogram 1/2/3/4 | distinct root branches / position | mean residual decrease / query | fraction of queries reducing residual | final proof residual / position | proof-completion fraction | ideal Q* <= B ceiling | post-completion queries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B2 | 0.560 / 0.566 | 0.092 / 0.096 | 0.348 / 0.338 | 0.761 / 0.773 | 1.33 / 1.33 | 0.67/0.33/0.00/0.00 (5101) | 1.33 / 1.33 | 0.560 / 0.566 | 0.560 / 0.566 | 4.16 / 4.14 | 0.333 / 0.333 | 0.333 / 0.333 | 1500 / 1500 |
| B4 | 0.406 / 0.416 | 0.191 / 0.194 | 0.403 / 0.390 | 0.761 / 0.773 | 1.91 / 1.91 | 0.36/0.40/0.21/0.03 (5101) | 1.45 / 1.45 | 0.406 / 0.416 | 0.406 / 0.416 | 3.65 / 3.61 | 0.486 / 0.487 | 0.579 / 0.579 | 5183 / 5188 |
| B8 | 0.232 / 0.239 | 0.112 / 0.110 | 0.656 / 0.651 | 0.761 / 0.773 | 2.99 / 2.99 | 0.18/0.23/0.23/0.18 (5101) | 1.46 / 1.46 | 0.232 / 0.239 | 0.232 / 0.239 | 3.42 / 3.36 | 0.544 / 0.550 | 0.799 / 0.799 | 14619 / 14688 |

KQRvK M3 only:

| B | proof-admissible | refute-admissible | off-target | first-query correct root | proof-completion fraction | ideal Q* <= B | final residual / position |
|---|---|---|---|---|---|---|---|
| B2 | 0.373 / 0.388 | 0.179 / 0.193 | 0.447 / 0.419 | 0.491 / 0.524 | 0.000 / 0.000 | 0.000 / 0.000 | 10.62 / 10.59 |
| B4 | 0.305 / 0.327 | 0.174 / 0.182 | 0.521 / 0.491 | 0.491 / 0.524 | 0.000 / 0.000 | 0.000 / 0.000 | 10.15 / 10.06 |
| B8 | 0.208 / 0.214 | 0.099 / 0.099 | 0.693 / 0.686 | 0.491 / 0.524 | 0.037 / 0.052 | 0.432 / 0.432 | 9.71 / 9.65 |

Observations:

- **The selector does choose proof-admissible edges, and does reduce the proof residual, but increasingly less so as
  the budget grows.** The proof-admissible share of queries is about 0.56 at B2, 0.41 at B4 and 0.23-0.24 at B8; the
  off-target share rises to about 0.65 at B8. (`mean residual decrease / query` equals the proof-admissible share: the total residual decrease equals the number of
  proof-admissible queries, i.e. one per such query.) A large number of B8 queries (about 14.6k of 36k) were issued after the proof was
  already complete; how the diagnostic classifies those was not separated here, so the off-target share should not be read as
  purely wrong-branch queries.
- **Proof completion lags the ideal ceiling as the budget grows:** B2 matches the ceiling (0.333 vs 0.333), B4 reaches
  0.49 of a possible 0.58, B8 reaches 0.54-0.55 of a possible 0.80. On KQRvK M3 at B8 only 3.7-5.2% of positions are
  completed against an ideal 43%. Residual still shrinks substantially without completion (final residual per position
  about 3.4 at B8 on the pooled set, about 9.7 on M3).
- First-query correct-root rate (0.76-0.77) is identical at every budget and equals the B2 policy top-1 (0.761 /
  0.773), consistent with the first query following the policy's root choice (not verified separately).

## 6. ACTIVE versus teacher versus FIXED (selected LR, update 800)

Teacher = `proof_teacher_seeded_v1` forcing the query sequence, the same teacher the model was trained against.
FIXED = `fixed_bfs_actionid_v1`, the frozen non-selective schedule. Policy metrics, `5101 / 5102`. These are diagnostic
comparisons, **not Gate II/III results**.

| B | mode | pooled top-1 | correct mass | CE | entropy | KQRvK M3 top-1 | KQRvK M3 mass | KQRvK M3 CE |
|---|---|---|---|---|---|---|---|---|
| B2 | ACTIVE | 0.761 / 0.773 | 0.702 / 0.719 | 1.574 / 1.550 | 0.817 / 0.775 | 0.491 / 0.524 | 0.410 / 0.439 | 2.851 / 2.780 |
| B2 | TEACHER | 0.994 / 0.993 | 0.835 / 0.848 | 0.962 / 0.946 | 0.932 / 0.880 | 0.996 / 0.995 | 0.678 / 0.690 | 1.729 / 1.701 |
| B2 | FIXED | 0.750 / 0.764 | 0.614 / 0.639 | 1.378 / 1.347 | 1.468 / 1.398 | 0.439 / 0.479 | 0.297 / 0.326 | 2.445 / 2.404 |
| B4 | ACTIVE | 0.761 / 0.773 | 0.700 / 0.722 | 1.614 / 1.568 | 0.791 / 0.748 | 0.491 / 0.524 | 0.414 / 0.442 | 2.928 / 2.818 |
| B4 | TEACHER | 0.995 / 0.993 | 0.839 / 0.850 | 0.960 / 0.947 | 0.917 / 0.869 | 0.995 / 0.991 | 0.668 / 0.680 | 1.711 / 1.692 |
| B4 | FIXED | 0.748 / 0.763 | 0.617 / 0.641 | 1.378 / 1.350 | 1.445 / 1.381 | 0.440 / 0.476 | 0.301 / 0.329 | 2.445 / 2.407 |
| B8 | ACTIVE | 0.758 / 0.771 | 0.670 / 0.696 | 1.565 / 1.522 | 0.953 / 0.900 | 0.491 / 0.525 | 0.390 / 0.420 | 2.804 / 2.751 |
| B8 | TEACHER | 0.994 / 0.991 | 0.825 / 0.845 | 0.956 / 0.943 | 0.961 / 0.884 | 0.992 / 0.980 | 0.627 / 0.649 | 1.722 / 1.697 |
| B8 | FIXED | 0.748 / 0.765 | 0.618 / 0.643 | 1.379 / 1.351 | 1.441 / 1.371 | 0.435 / 0.483 | 0.301 / 0.330 | 2.448 / 2.412 |

Selector NLL under the teacher (mean per supervised decision), B2/B4/B8: seed 5101 0.909 / 1.228 / 1.487; seed 5102 0.903 / 1.206 / 1.454.

Answers to the interpretive questions:

1. **Does teacher-forced information materially help the policy? Yes, a lot.** With the teacher's queries the policy reaches
   top-1 about 0.99 and CE about 0.95 at B2/B4/B8 (KQRvK M3 top-1 0.98-1.00), against ACTIVE top-1 about 0.76 / CE about
   1.55. **Caveat:** the teacher walks the proof path, so *which* edges it queries also reveals the answer; this diagnostic
   cannot separate "the integrator exploits the returned states" from "the integrator reads the teacher's query pattern".
2. **How far behind teacher is ACTIVE?** Far: about 0.23 top-1 and about 0.6 nats of CE pooled (KQRvK M3: about 0.5 top-1 and about 1.1
   nats).
3. **Does ACTIVE differ from FIXED?** Slightly. ACTIVE has a marginally higher top-1 (0.76-0.77 vs 0.75-0.76; M3 0.49-0.52 vs
   0.43-0.48) and higher correct mass (0.67-0.72 vs 0.61-0.64), but a **worse CE** (1.52-1.61 vs 1.35-1.38) because it is sharper
   and less calibrated. FIXED barely changes between B2, B4 and B8.
4. **Is the selector increasingly choosing proof-admissible edges?** Not with budget (share falls from 0.56 to 0.23); it is
   well above zero and the first query follows the policy's root choice.
5. **Is residual reduction occurring without a complete proof?** Yes: the residual falls by about one per proof-admissible query,
   and the mean final residual stays above zero for the incomplete positions.

## 7. Other learning rates at update 800 (ACTIVE macro CE per budget)

| run | B0 | B2 | B4 | B8 | KQRvK M3 CE at B8 |
|---|---|---|---|---|---|
| LR 7.5e-5 s5101 | 1.474 | 1.733 | 1.846 | 1.794 | 3.245 |
| LR 7.5e-5 s5102 | 1.475 | 1.727 | 1.797 | 1.724 | 3.135 |
| LR 1.5e-4 s5101 | 1.389 | 1.621 | 1.652 | 1.603 | 2.883 |
| LR 1.5e-4 s5102 | 1.378 | 1.593 | 1.632 | 1.589 | 2.944 |
| LR 3e-4 s5101 | 1.357 | 1.574 | 1.614 | 1.565 | 2.804 |
| LR 3e-4 s5102 | 1.330 | 1.550 | 1.568 | 1.522 | 2.751 |

The B0-below-B2/B4/B8 pattern holds at every LR and seed.

## 8. DAgger-style rescue: observation only (not run)

The evidence is closest to: **the integrator can exploit high-quality state acquisition (teacher-forced top-1 about 0.99) but ACTIVE
selection has a large exposure gap** (proof completion well below the ideal ceiling, falling proof-admissible share at larger
budgets, ACTIVE far behind teacher). The selector is trained only on teacher prefixes, so it never sees its own mistakes,
which is exactly what a scheduled-sampling/DAgger-style correction targets. Caveats: the teacher-forced gap is confounded by
the answer being visible in the query pattern, and ACTIVE does not beat B0 on CE at any budget yet. This **suggests a
DAgger-style rescue deserves consideration**; it was not implemented or run and needs owner authorisation.

## 9. Integrity and process notes

- Contract digest unchanged: `105ac3133877f954ed00e6ce9caaadabf6d5da1cf78a7ab99d44a195d03009d2` (re-verified by regenerating the contract with the release binary before launch).
- Summaries were validated by the strengthened `v3-p5 select` (schema, lr/seed, digests, update counts, evaluation set,
  ACTIVE B0/B2/B4/B8 only, recomputed `S_run`, exposure, no HOLDOUT/CONFIRM reference, preregistered ineligibility classes).
- **Validator defect found and fixed between the screen and the selection.** The first `select` invocation refused all
  summaries ("evaluation entry is not ACTIVE B0") because my validator compared the evaluator's `selection` label to `"ACTIVE"` while the
  evaluator writes `"active"`. It stopped before writing any selection or pairing output; I changed the check (and its test
  fixture) to use `EvalSelection::Active.label()`, rebuilt the CUDA binary and re-ran `select`. No data, rule or summary changed.
- CUDA checkpoint resume is not claimed bit-identical (CPU resume only was proven); no resume occurred in the clean screen.
- HOLDOUT_C was never loaded or evaluated; the custody/seal evidence files are unchanged. B16 was not trained or evaluated.
  No Gate I/II/III, DAgger, RL, ALL-INFO or P6 work was done.

**P6 NOT RUN - awaiting owner review/approval.**

