# V420 — results: `STOP_DEPTH_DOES_NOT_HELP`

**Outcome (pre-registered rule, applied once on 2026-11-11): S0 PASS, S4 PASS, S1 FAIL, S2 PASS (moot), S3 reported OK.**
One shared board-transformer block applied K times does NOT improve exactly-labelled endgame accuracy enough to meet the frozen gate.
Per the approved plan V420 ends here: no retry, no retune, option 2 (iterative plan refinement) stays unbuilt, no sealed set opened.

Plan: `docs/V420_RESEARCH_PLAN.md` (amendment A1). Ledger: `docs/V420_EXPERIMENTS.md` (V420-E0..E2). Decision: `docs/DECISIONS.md` V420-D1.
Evidence: `docs/evidence/v420/v420-measure.json` (gates + all tables), `v420-cuda-smoke.json` (engineering timings only).
Branch `experiment/workstation-v420-recurrent-depth`. Launcher `scripts/v420_run.sh`. All runs CUDA FP32, RTX 2000 Ada, about 1.8 GPU-hours.

## 1. Gates (MEASURED, V4_TRAIN_DEV, headline cell KQRvK M3, 507 positions, seeds 5101/5102/5103)

| gate | rule | measured | verdict |
|---|---|---|---|
| S0 | every run finite, healthy, 2,000 updates | all 16 runs complete, no `run-aborted.json`, health rule passed | PASS |
| S4 | BASE8 (seed 5101) top-1 on all dev within 0.01 of V4 Stage A 0.8403 | 0.8474 (diff +0.0070) | PASS, pipeline validated |
| S1 | CI(RK4 − R1) wholly > 0, every seed > 0, pooled gain >= +0.03 | mean +0.0217, CI [+2.7e-18, +0.0427], per seed 0.0000 / +0.0335 / +0.0316 | **FAIL** (gain < 0.03; CI lower bound is zero; seed 5101 gain exactly 0) |
| S2 | RV top-1 K=8 >= K=4 − 0.01 and K=16 >= K=4 − 0.02 | 0.6292 (K=4), 0.6220 (K=8), 0.6174 (K=16) | passes, but moot after S1 (see 3) |
| S3 | RK4 − WIDE >= −0.02 (reported, not a kill) | mean −0.0007, CI [−0.0210, +0.0197] | OK |

## 2. Accuracy by arm (means over seeds; headline = KQRvK M3 top-1; all = all 4,403 dev positions)

| arm (parameters) | trained at | evaluated at | headline | all dev | dev CE |
|---|---|---|---|---|---|
| R1 (4.7M) | K=1 | K=1 | 0.642 | 0.844 | 1.259 |
| RK2 (4.7M) | K=2 | K=2 | 0.636 | 0.835 | 1.274 |
| RK4 (4.7M) | K=4 | K=4 | 0.663 | 0.843 | 1.250 |
| RV (4.7M) | K=1..4 sampled | K=1 / 2 / 4 / 8 / 16 | 0.633 / 0.633 / 0.629 / 0.622 / 0.617 | 0.831 / 0.831 / 0.830 / 0.829 / 0.826 | 1.30–1.34 |
| WIDE (14.6M, 4 unshared blocks) | K=4 | K=4 | 0.664 | 0.846 | 1.253 |
| BASE8 (27.7M, seed 5101 only) | K=8 | K=8 | 0.649 | 0.847 | 1.251 |

Diagnostics, not gates: fixed-K arms evaluated away from their training depth degrade sharply (R1 at K=16 headline 0.412; RK4 at K=1 0.411;
RK4 at K=16 0.638), so a block trained at one depth is not a general iterate. Only RV, trained across depths, is flat over K.

## 3. Reading, and what is NOT established
- **Established (MEASURED):** on this exactly-labelled task, a 4.7M one-block model scores within 0.02 headline / 0.003 all-dev top-1 of a 14.6M
  and a 27.7M unshared model. The whole range of capacity and depth tested here moves the headline cell by about 0.02, with a CI touching zero.
  Depth by recurrence helps no more than extra unshared depth does, and neither helps much.
- **INFERRED, NOT TESTED:** accuracy on KQRvK M3 is limited by something other than board-transformer depth or capacity, for example that
  mate-in-3 needs search-like lookahead a single-position network does not do, or label ambiguity. This is the same direction V3 pointed
  to; it is not established here.
- **S2 passing is not evidence for recurrence.** RV is flat across K because it learned to be nearly depth-insensitive (K=1 scores the same as K=16), which
  satisfies the "does not collapse" rule without any benefit from depth.
- **Not established:** anything about larger scale, other tasks, adaptive halting (P3 was gated on S1 and S2 and is not authorised), longer training, or
  other recurrence designs. Three seeds, one 507-position cell and 2,000 updates per run limit what can be claimed.
- Training loss gave the same hint during the run: the one-block R1 ended with a last-25 training loss (0.8895 on seed 5101) no worse than BASE8 (0.9046).

## 4. Process notes
- The plan's original S4 was impossible as written (a 1-block arm against V4's 8-block baseline); it was amended before any run with owner approval (A1), and BASE8 then validated the pipeline.
- A test FEN of mine was illegal and one console API differed from my first guess; both were found by tests/compilation before any run. No run was repeated or discarded.
- `v4_tune_v1` and HOLDOUT_C were never loaded. Run directories hold final weights only and are not resumable.

`V420 CLOSED. NO RETRY. HOLDOUT_C AND v4_tune_v1 UNTOUCHED.`
