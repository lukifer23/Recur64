#!/bin/bash
# V4 P1 Stage C/D: utility head on frozen base+evidence from the Stage B finals; both losses on
# identical data/seeds, frozen selection rule, then the integration smoke report.
# Protocol: docs/V4_EXPERIMENTS.md V4-E2.
cd /c/Users/LukeScaggs/Documents/Recur64 || exit 1
export CUDA_PATH="$LOCALAPPDATA/Recur64/cuda/12.9.1"
export PATH="$CUDA_PATH/bin:$PATH"
BIN=./target/release/recur64
TRAIN=runs/v25/p25/data/proof-train.json
EV=docs/evidence/v4
step() {
  local name="$1"; shift
  echo "=== $name" >> runs/v4/run.log
  "$@" >> "runs/v4/$name.out" 2>> "runs/v4/$name.err"
  local rc=$?
  echo "exit $rc for $name" >> runs/v4/run.log
  if [ "$rc" -ne 0 ]; then echo "STOPPED ($name)" >> runs/v4/run.log; exit "$rc"; fi
}
for loss in ranking regression; do
  for seed in 5101 5102 5103; do
    step "c-$loss-seed$seed" $BIN v4 train --train $TRAIN --stage c --device cuda --seed $seed --updates 300 --lr 3e-4 --loss $loss --init runs/v4/b-seed$seed --run-dir runs/v4/c-$loss-seed$seed
  done
  step "c-measure-$loss" $BIN v4 measure --train $TRAIN --kind c --device cuda --run-dirs runs/v4/c-$loss-seed5101,runs/v4/c-$loss-seed5102,runs/v4/c-$loss-seed5103 --output $EV/stage-c-$loss.json
done
step "c-select-loss" $BIN v4 select-loss --ranking $EV/stage-c-ranking.json --regression $EV/stage-c-regression.json --output $EV/stage-c-loss-selected.json
LOSS=$(grep -o '"selected_loss": "[a-z]*"' $EV/stage-c-loss-selected.json | cut -d'"' -f4)
echo "selected loss $LOSS" >> runs/v4/run.log
step "d-measure" $BIN v4 measure --train $TRAIN --kind d --device cuda --run-dirs runs/v4/c-$LOSS-seed5101,runs/v4/c-$LOSS-seed5102,runs/v4/c-$LOSS-seed5103 --output $EV/stage-d-integration.json
echo STAGE_C_DONE >> runs/v4/run.log
