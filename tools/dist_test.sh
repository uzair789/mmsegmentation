#!/usr/bin/env bash

WORK_FOLDER='pointrend_r50_224x224_singlescale_80k_raleigh'
WORK_FOLDER='pointrend_r50_512x1024_80k_raleigh'
WORK_DIR='work_dirs/'${WORK_FOLDER}
CONFIG=${WORK_DIR}'/pointrend_r50_512x1024_80k_raleigh.py'
GPUS=2
#CONFIG=$1
CHECKPOINT=${WORK_DIR}'/iter_80000.pth'
#GPUS=$3
PORT=${PORT:-29502}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval mIoU
