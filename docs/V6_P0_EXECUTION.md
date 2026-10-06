# Revised V6 P0 execution freeze

Scientific implementation source: `42f47b852451d59645dcc50120e4a426b034f493`.
V5 remains closed as NO_SIGNAL. No V5 source or checkpoint has changed.

## MEASURED prerequisites

Full release workspace: 614 passed, two original ignored; native exit 0.
Scoped formatting, CUDA all-target Clippy and serial pinned CUDA build: native exit 0.
CPU and RTX 2050 CUDA FP32 microbatch-2 qualification: PASS. Both readers passed
exact normal/profile forward, gradients, post-AdamW parameters/moments, independent
clone purity, 50 resident updates, restore and exact continuation, absent baseline
gradients, deep-depth-five R1 dependency and null invariance.
Principal total parameters 7,687,216; one-pass 6,703,152, including frozen root.
CUDA device-wide sampled peak 451 MiB; this is sampled memory, not a continuous peak.
TRAIN/DEV/CONFIRM actual custody passed; pairwise overlaps zero; CONFIRM sealed,
evaluated=false. DEV and CONFIRM were used only for byte custody, never model inference.

## Acquisition measurement, not policy tuning

Frozen 216 TRAIN roots: 183 B0-right, 33 B0-wrong. Narrow policy acquired a correct
branch for 11/33 wrong roots; broader policy 12/33. Both cover correct branches for
183/183 right roots. Full by-cell/policy/stratum counts are in acquisition receipt.
Unqueried replies remain unknown. No policy, objective or equation was retuned.

## Both disposable invocations preregistered together

Exact 96 IDs, strata, 192 persisted packets, raw-plan SHA, source/config and both
qualification hashes are frozen in `plan-binding-42f47b8.json`. Seed 6300, 200
updates, physical batch 2, effective batch 24, warmup 20, peak LR 1e-3, unchanged
AdamW/cosine and eligible-branch objective. Principal R4; full-information one-pass
R1. Shared-shape initial tensors match. Same root episode bags and policy exposure.

Invoke `recur64-v6` with the three bound raw paths and immutable Stage A import,
CUDA, and unique output receipt, then `learn --arm principal` or `--arm one-pass`,
`--plan runs/v6-p0/frozen-plan-42f47b8.json`,
`--plan-binding docs/evidence/v6-p0/plan-binding-42f47b8.json`, both current-source
qualification paths, and separate `runs/v6-p0/seed-6300/principal` / `one-pass` dirs.
Use `--max-minutes 40` to reserve startup/endpoint overhead below the 45-minute
process bound; resume only a clean bounded stop, with exact arguments plus
`--resume`. Track native process elapsed time separately; maximum total two hours
per arm. No endpoint comparison until both arms complete. Runtime/integrity failure
stops remaining work; a performance failure does not authorize tuning or retry.

Counting unit: roots, corrected under BOTH policies, harmed under EITHER;
shuffle specificity requires loss of correctness under BOTH policies. Frozen
thresholds and all stop rules remain those in V6_P0_REVISED_CONTRACT.md.

## NOT RUN at this freeze

Both fitting arms, DEV inference, scientific 800-update campaign, other seeds,
CONFIRM/V4_TUNE/HOLDOUT_C inference, controller, self-play, and V5 rescue.
