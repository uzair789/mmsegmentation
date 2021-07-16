#!/usr/bin/env bash

CONFIG='work_dirs/pointrend_r50_224x224_80k_raleigh/pointrend_r50_512x1024_80k_raleigh.py'
GPUS=2
#CONFIG=$1
CHECKPOINT='work_dirs/pointrend_r50_224x224_80k_raleigh/iter_40000.pth'
#GPUS=$3
PORT=${PORT:-29501}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval mIoU
