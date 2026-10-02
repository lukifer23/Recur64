#!/bin/bash
# V3.5: the three on-policy runs, seeds 5101, 5102, 5103, one process each (each ~30 min).
# Layout micro16 x accum8 (frozen in V35-B). Fail-stop: any non-zero exit halts the launcher;
# inspect before relaunching (a run directory with a valid state resumes; one with only
# partial artifacts is refused by the CLI). HOLDOUT_C is never opened.
cd /c/Users/LukeScaggs/Documents/Recur64 || exit 1
export CUDA_PATH="$LOCALAPPDATA/Recur64/cuda/12.9.1"
export PATH="$CUDA_PATH/bin:$PATH"
mkdir -p runs/v35
init_dir() {
  case "$1" in
    5101) echo runs/v3/p5/v3-p5-run-lr3e-4-seed5101/final ;;
    5102) echo runs/v3/p5/v3-p5-run-lr3e-4-seed5102/final ;;
    5103) echo runs/v3/p6/v3-p5-run-lr3e-4-seed5103/final ;;
  esac
}
for seed in 5101 5102 5103; do
  stem="v35-run-seed${seed}"
  echo "=== V3.5 seed $seed -> $stem" >> runs/v35/run.log
  ./target/release/recur64 v3-p35 train --train runs/v25/p25/data/proof-train.json --train-trace runs/v3/p4/trace-train --tune runs/v3/data/proof-v3-tune-v1.json --tune-trace runs/v3/p4/trace-tune --seed $seed --init-dir "$(init_dir $seed)" --device cuda --layout 16x8 --run-dir runs/v35/$stem >> runs/v35/$stem.out 2>> runs/v35/$stem.err
  rc=$?
  echo "exit $rc for $stem" >> runs/v35/run.log
  if [ "$rc" -ne 0 ]; then
    echo "STOPPED after $stem (exit $rc): inspect before continuing" >> runs/v35/run.log
    exit "$rc"
  fi
done
echo ALL_DONE >> runs/v35/run.log
