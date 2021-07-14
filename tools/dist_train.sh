#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES='0,2'
CONFIG='configs/point_rend/pointrend_r50_512x1024_80k_raleigh.py'
GPUS=2
#CONFIG=$1
#GPUS=$2
#PORT=${PORT:-29500}
PORT=${PORT:-29501}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
