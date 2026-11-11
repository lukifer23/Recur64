#!/bin/bash
# V420 P2: train every arm/seed with the frozen recipe, then apply the frozen gates once. Fail-stop:
# any non-zero exit stops the script and leaves run-aborted.json / the log visible. TRAIN data only.
# Order: BASE8 first (pipeline check S4), then each seed's arms. Plan: docs/V420_RESEARCH_PLAN.md.
cd /c/Users/LukeScaggs/Documents/Recur64 || exit 1
export CUDA_PATH="$LOCALAPPDATA/Recur64/cuda/12.9.1"
export PATH="$CUDA_PATH/bin:$PATH"
TRAIN=runs/v25/p25/data/proof-train.json
ROOT=runs/v420
mkdir -p "$ROOT" docs/evidence/v420
run() { # arm seed
  local arm=$1 seed=$2
  echo "=== $arm seed $seed $(date -Is)" >> "$ROOT/run.log"
  ./target/release/recur64 v420 train --train $TRAIN --arm "$arm" --seed "$seed" --device cuda --run-dir "$ROOT/$arm-seed$seed" >> "$ROOT/$arm-seed$seed.out" 2>> "$ROOT/$arm-seed$seed.err"
  local rc=$?
  echo "exit $rc for $arm seed $seed" >> "$ROOT/run.log"
  if [ "$rc" -ne 0 ]; then echo "STOPPED ($arm seed $seed)" >> "$ROOT/run.log"; exit "$rc"; fi
}
run BASE8 5101
for seed in 5101 5102 5103; do
  for arm in R1 RK2 RK4 RV WIDE; do run "$arm" "$seed"; done
done
./target/release/recur64 v420 measure --train $TRAIN --device cuda --runs-root "$ROOT" --output docs/evidence/v420/v420-measure.json >> "$ROOT/measure.out" 2>> "$ROOT/measure.err"
rc=$?
echo "exit $rc for measure" >> "$ROOT/run.log"
echo V420_DONE >> "$ROOT/run.log"
exit "$rc"
