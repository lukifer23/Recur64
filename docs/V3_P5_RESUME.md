# V3 P5 LR screen: where we stopped, and how to resume

> **SUPERSEDED (2026-10-01, V3-D20):** the update-150 state described below is voided and quarantined; the screen
> restarts from update 0 with the P5.1-hardened launcher. Kept for the record; do not resume from it.


**Stopped deliberately on 2026-10-01 (workstation move).** No screen result exists. The pre-registered plan
(`docs/V3_P5_PLAN.md`), decisions V3-D16 to V3-D19, the screen contract (`docs/evidence/v3/v3-p5-recipe.json`,
contract digest `105ac3133877f954ed00e6ce9caaadabf6d5da1cf78a7ab99d44a195d03009d2`) and the resolved layout
(micro16 x accum8) are committed and pushed; nothing about them changes on resume.

## State at the stop

- Run 1 of 6 (LR 7.5e-5, seed 5101), run digest `adc844428c65486c0b0a37604ec99a7a8b761c229a01443b3ee7925bdd171b01`,
  was stopped by killing the launcher and then the trainer **immediately after the update-150 checkpoint was written**.
  `runs/v3/p5/v3-p5-run-lr7.5e-5-seed5101/state-1` holds update 150 (`state-0` holds update 100 as a fallback).
  The update-0 TUNE evaluation (`eval-u0000.json`, S_run 3.55327) is saved. Runs 2 to 6 have not started.
- Training was healthy up to the stop (update 130: total loss 3.64, policy 1.52, selector 2.11, grad norm 4.4).
- `runs/` is git-ignored: the checkpoints live only on this machine's disk. Do not delete `runs/v3/p5/`.

## Resume (after the machine is back)

```bash
cd /c/Users/LukeScaggs/Documents/Recur64
git pull --ff-only origin experiment/workstation-v3-active-search   # if working from another checkout
cargo build --release --features cuda -p recur64-cli                # the binary must be the CUDA build
sed -i 's/$//' scripts/v3_p5_run_screen.sh                          # git may have turned LF into CRLF
nohup bash scripts/v3_p5_run_screen.sh > /dev/null 2>&1 &
```

The script sets `CUDA_PATH=$LOCALAPPDATA/Recur64/cuda/12.9.1` and puts its `bin` on `PATH` (without it the device
known-answer guard refuses to run; there is no CPU fallback). It runs the six runs in order: seed 5101 for LR
7.5e-5, 1.5e-4, 3.0e-4, then seed 5102 for the same. A run whose directory already has a state resumes from its
newest valid checkpoint and skips evaluations whose `eval-uNNNN.json` already exists, so relaunching run 1
continues from update 150. A different recipe is refused. If the update-150 checkpoint fails to load, delete only
`state-1` of that run to fall back to update 100; do not otherwise edit run directories.
Progress: `runs/v3/p5/screen.log` and each run's `.err`. Logs are appended on resume.

Caveats to carry into the results: resume exactness (bit-exact) was proven on CPU only; a CUDA resume is not claimed
bit-identical to an uninterrupted run (GPU reductions are not guaranteed deterministic), and run 1 will have
been resumed once. Report that in `V3_P5_RESULTS.md`. Each run is projected at about 0.47 h; all six about 3 h.

## After the six runs

1. Copy `runs/v3/p5/summaries/*.json` to `docs/evidence/v3/`.
2. `recur64 v3-p5 select --summaries docs/evidence/v3 --recipe docs/evidence/v3/v3-p5-recipe.json --selection-out docs/evidence/v3/v3-p5-lr-selection.json --selected-recipe-out docs/evidence/v3/v3-p5-selected-recipe.json` (applied once; it refuses to overwrite).
3. Write `docs/V3_P5_RESULTS.md`, append STATUS / README / DECISIONS / ledger, Commit C, push, stop.
   End the report with `P6 NOT RUN - awaiting owner review/approval.`
