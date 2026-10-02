#!/bin/bash
# V4 P1 Stage A: the base tower on V4_TRAIN_FIT, seeds 5101/5102/5103, one process each (each well
# under the 2-hour guard). TRAIN only. Protocol: docs/V4_EXPERIMENTS.md V4-E2. Fail-stop.
cd /c/Users/LukeScaggs/Documents/Recur64 || exit 1
export CUDA_PATH="$LOCALAPPDATA/Recur64/cuda/12.9.1"
export PATH="$CUDA_PATH/bin:$PATH"
mkdir -p runs/v4 docs/evidence/v4
for seed in 5101 5102 5103; do
  echo "=== stage A seed $seed" >> runs/v4/run.log
  ./target/release/recur64 v4 train --train runs/v25/p25/data/proof-train.json --stage a --device cuda --seed $seed --updates 2000 --lr 3e-4 --run-dir runs/v4/a-seed$seed >> runs/v4/a-seed$seed.out 2>> runs/v4/a-seed$seed.err
  rc=$?
  echo "exit $rc for stage A seed $seed" >> runs/v4/run.log
  if [ "$rc" -ne 0 ]; then echo "STOPPED (stage A seed $seed)" >> runs/v4/run.log; exit "$rc"; fi
  ./target/release/recur64 v4 measure --train runs/v25/p25/data/proof-train.json --kind a --device cuda --run-dirs runs/v4/a-seed$seed --output docs/evidence/v4/stage-a-seed$seed.json >> runs/v4/a-seed$seed.out 2>> runs/v4/a-seed$seed.err
  rc=$?
  echo "exit $rc for stage A measure seed $seed" >> runs/v4/run.log
  if [ "$rc" -ne 0 ]; then echo "STOPPED (stage A measure seed $seed)" >> runs/v4/run.log; exit "$rc"; fi
done
echo STAGE_A_DONE >> runs/v4/run.log
