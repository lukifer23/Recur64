#!/bin/bash
# V500 ablation: wait for the piece-square-only net, then run the same protocol as the full nets (CPU only).
cd /c/Users/LukeScaggs/Documents/Recur64 || exit 1
until grep -q TRAIN_DONE runs/v500/abl-sq-5101.log 2>/dev/null; do sleep 30; done
N=runs/v500/abl-sq-5101/net.json
B=./target-opt/release/examples
$B/check_net.exe $N runs/v500/abl-sq-5101/check-net.json > runs/v500/eval/abl-sq.log 2>&1
$B/eval_prover.exe $N KRvK val 20 100000 2,3,4 runs/v500/eval/abl-sq-val.json 12 >> runs/v500/eval/abl-sq.log 2>&1
$B/eval_prover.exe $N KRvK test 40 1000000 1,2,3 runs/v500/eval/abl-sq-test.json 12 >> runs/v500/eval/abl-sq.log 2>&1
echo DONE >> runs/v500/eval/abl-sq.log
