#!/bin/bash
# V500: evaluate every KRvK net as soon as it exists (CPU only; the GPU keeps training the next net).
# VALIDATION (budget 1e5, widths 2,3,4) is for the frozen width rule; TEST (budget 1e6, widths 1,2,3) feeds the gates.
cd /c/Users/LukeScaggs/Documents/Recur64 || exit 1
mkdir -p runs/v500/eval
for tag in full-5102 full-5103 extrap-5101 extrap-5102 extrap-5103; do
  net=runs/v500/v2-$tag/net.json
  until [ -f "$net" ]; do sleep 20; done
  sleep 5
  [ -f runs/v500/eval/v2-$tag-test.json ] && continue
  ./target/release/examples/check_net "$net" runs/v500/v2-$tag/check-net.json >> runs/v500/eval/pipeline.log 2>&1
  ./target/release/examples/eval_prover "$net" KRvK val 20 100000 2,3,4 runs/v500/eval/v2-$tag-val.json 10 >> runs/v500/eval/pipeline.log 2>&1
  ./target/release/examples/eval_prover "$net" KRvK test 40 1000000 1,2,3 runs/v500/eval/v2-$tag-test.json 10 >> runs/v500/eval/pipeline.log 2>&1
  echo "evaluated $tag" >> runs/v500/eval/pipeline.log
done
echo PIPELINE_DONE >> runs/v500/eval/pipeline.log
