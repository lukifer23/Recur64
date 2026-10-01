#!/bin/bash
# P6 baseline replication (V3-D22): the exact selected P5 recipe at paired seed 5103.
cd /c/Users/LukeScaggs/Documents/Recur64 || exit 1
export CUDA_PATH="$LOCALAPPDATA/Recur64/cuda/12.9.1"
export PATH="$CUDA_PATH/bin:$PATH"
stem="v3-p5-run-lr3e-4-seed5103"
mkdir -p runs/v3/p6
echo "=== 3.0e-4 5103 -> $stem" >> runs/v3/p6/replication.log
./target/release/recur64 v3-p5 train --train runs/v25/p25/data/proof-train.json --train-trace runs/v3/p4/trace-train --tune runs/v3/data/proof-v3-tune-v1.json --tune-trace runs/v3/p4/trace-tune --recipe docs/evidence/v3/v3-p5-recipe.json --lr 3.0e-4 --seed 5103 --device cuda --p6-baseline-replication --selected-recipe docs/evidence/v3/v3-p5-selected-recipe.json --run-dir runs/v3/p6/$stem --summary runs/v3/p6/summaries/$stem.json >> runs/v3/p6/$stem.out 2>> runs/v3/p6/$stem.err
rc=$?
echo "exit $rc for $stem" >> runs/v3/p6/replication.log
exit $rc
