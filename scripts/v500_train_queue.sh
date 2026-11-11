#!/bin/bash
# V500: queued training runs (v2 listwise ranking, 20 epochs, h1 256, h2 64). Two queues run concurrently on the one GPU.
cd /c/Users/LukeScaggs/Documents/Recur64 || exit 1
export CUDA_PATH="$LOCALAPPDATA/Recur64/cuda/12.9.1"
export PATH="$CUDA_PATH/bin:$PATH"
queue() {
  for job in "$@"; do
    variant=${job%%:*}; seed=${job##*:}
    d=runs/v500/v2-$variant-$seed
    [ -f $d/net.json ] && continue
    ./target/release/examples/train_net $d $seed $variant 20 > runs/v500/v2-$variant-$seed.log 2>&1
    echo "exit $? for $variant $seed" >> runs/v500/queue.log
  done
}
if [ "$1" = A ]; then queue full:5102 extrap:5101 extrap:5103; else queue full:5103 extrap:5102; fi
echo "QUEUE_$1_DONE" >> runs/v500/queue.log
