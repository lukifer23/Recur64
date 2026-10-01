#!/bin/bash
# P6: the three ALL-INFO runs, seeds 5101, 5102, 5103, one process each (each < 2 h projected).
# Fail-stop: any non-zero exit halts the launcher; inspect before relaunching (a run directory with a valid
# state resumes; one with only partial artifacts is refused by the CLI).
cd /c/Users/LukeScaggs/Documents/Recur64 || exit 1
export CUDA_PATH="$LOCALAPPDATA/Recur64/cuda/12.9.1"
export PATH="$CUDA_PATH/bin:$PATH"
mkdir -p runs/v3/p6/summaries
for seed in 5101 5102 5103; do
  stem="v3-p6-allinfo-seed${seed}"
  echo "=== ALL-INFO seed $seed -> $stem" >> runs/v3/p6/allinfo.log
  ./target/release/recur64 v3-p6 train --train runs/v25/p25/data/proof-train.json --tune runs/v3/data/proof-v3-tune-v1.json --recipe docs/evidence/v3/v3-p6-recipe.json --seed $seed --device cuda --run-dir runs/v3/p6/$stem --summary runs/v3/p6/summaries/$stem.json >> runs/v3/p6/$stem.out 2>> runs/v3/p6/$stem.err
  rc=$?
  echo "exit $rc for $stem" >> runs/v3/p6/allinfo.log
  if [ "$rc" -ne 0 ]; then
    echo "STOPPED after $stem (exit $rc): inspect before continuing" >> runs/v3/p6/allinfo.log
    exit "$rc"
  fi
done
echo ALL_DONE >> runs/v3/p6/allinfo.log
