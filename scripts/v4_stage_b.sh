#!/bin/bash
# V4 P1 Stage B: (1) TRAIN-only LR screen on seed 5101, (2) the three seeds at the screened LR,
# (3) the pre-registered mechanism measurement (questions A, B, C, G) on V4_TRAIN_DEV.
# Halts if A, B or C fail (stop architecture development). Protocol: docs/V4_EXPERIMENTS.md V4-E2.
cd /c/Users/LukeScaggs/Documents/Recur64 || exit 1
export CUDA_PATH="$LOCALAPPDATA/Recur64/cuda/12.9.1"
export PATH="$CUDA_PATH/bin:$PATH"
BIN=./target/release/recur64
TRAIN=runs/v25/p25/data/proof-train.json
EV=docs/evidence/v4
step() { # name, command...
  local name="$1"; shift
  echo "=== $name" >> runs/v4/run.log
  "$@" >> "runs/v4/$name.out" 2>> "runs/v4/$name.err"
  local rc=$?
  echo "exit $rc for $name" >> runs/v4/run.log
  if [ "$rc" -ne 0 ]; then echo "STOPPED ($name)" >> runs/v4/run.log; exit "$rc"; fi
}
LRS="3e-4 1e-3 3e-3"
REPORTS=""; LRLIST=""
for lr in $LRS; do
  step "b-screen-lr$lr" $BIN v4 train --train $TRAIN --stage b --device cuda --seed 5101 --updates 400 --lr $lr --init runs/v4/a-seed5101 --run-dir runs/v4/b-screen-lr$lr
  step "b-screen-measure-lr$lr" $BIN v4 measure --train $TRAIN --kind b --device cuda --run-dirs runs/v4/b-screen-lr$lr --output $EV/stage-b-screen-lr$lr.json
  REPORTS="$REPORTS,$EV/stage-b-screen-lr$lr.json"; LRLIST="$LRLIST,$lr"
done
step "b-select-lr" $BIN v4 select-lr --reports "${REPORTS#,}" --lrs "${LRLIST#,}" --output $EV/stage-b-lr-selected.json
LR=$(grep -o '"selected_lr": [0-9.e-]*' $EV/stage-b-lr-selected.json | cut -d' ' -f2)
echo "selected LR $LR" >> runs/v4/run.log
for seed in 5101 5102 5103; do
  step "b-seed$seed" $BIN v4 train --train $TRAIN --stage b --device cuda --seed $seed --updates 1200 --lr $LR --init runs/v4/a-seed$seed --run-dir runs/v4/b-seed$seed
done
step "b-measure" $BIN v4 measure --train $TRAIN --kind b --device cuda --run-dirs runs/v4/b-seed5101,runs/v4/b-seed5102,runs/v4/b-seed5103 --output $EV/stage-b-mechanism.json
if grep -q '"stop_architecture_development": true' $EV/stage-b-mechanism.json; then
  echo "STAGE_B_VERDICT: A, B or C FAILED - stop architecture development" >> runs/v4/run.log
  exit 3
fi
echo STAGE_B_DONE >> runs/v4/run.log
