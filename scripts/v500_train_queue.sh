#!/bin/bash
# V500: sequential training queue (v2 listwise ranking, 20 epochs, h1 256, h2 64). One job at a time on the one GPU
# (two concurrent jobs were measured to be slower in total). Job spec: variant:seed:materials.
cd /c/Users/LukeScaggs/Documents/Recur64 || exit 1
export CUDA_PATH="$LOCALAPPDATA/Recur64/cuda/12.9.1"
export PATH="$CUDA_PATH/bin:$PATH"
for job in full:5102:both full:5103:both full:5101:kbn extrap:5101:both extrap:5102:both extrap:5103:both; do
  IFS=: read variant seed mats <<< "$job"
  tag=$variant-$seed; [ "$mats" != both ] && tag=$variant-$seed-$mats
  d=runs/v500/v2-$tag
  [ -f $d/net.json ] && continue
  ./target/release/examples/train_net $d $seed $variant 20 256 64 $mats > runs/v500/v2-$tag.log 2>&1
  echo "exit $? for $tag" >> runs/v500/queue.log
done
echo QUEUE_DONE >> runs/v500/queue.log
