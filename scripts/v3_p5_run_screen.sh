#!/bin/bash
# Six preregistered screening runs, one process each (every run < 2 h projected).
cd /c/Users/LukeScaggs/Documents/Recur64 || exit 1
export CUDA_PATH="$LOCALAPPDATA/Recur64/cuda/12.9.1"
export PATH="$CUDA_PATH/bin:$PATH"
for seed in 5101 5102; do
  for pair in "7.5e-5:7.5e-5" "1.5e-4:1.5e-4" "3.0e-4:3e-4"; do
    lr=${pair%%:*}; tag=${pair##*:}
    stem="v3-p5-run-lr${tag}-seed${seed}"
    echo "=== $lr $seed -> $stem" >> runs/v3/p5/screen.log
    ./target/release/recur64 v3-p5 train       --train runs/v25/p25/data/proof-train.json --train-trace runs/v3/p4/trace-train       --tune runs/v3/data/proof-v3-tune-v1.json --tune-trace runs/v3/p4/trace-tune       --recipe docs/evidence/v3/v3-p5-recipe.json --lr $lr --seed $seed --device cuda       --run-dir runs/v3/p5/$stem --summary runs/v3/p5/summaries/$stem.json       >> runs/v3/p5/$stem.out 2>> runs/v3/p5/$stem.err
    echo "exit $? for $stem" >> runs/v3/p5/screen.log
  done
done
echo ALL_DONE >> runs/v3/p5/screen.log
