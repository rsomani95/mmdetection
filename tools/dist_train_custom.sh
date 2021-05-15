#!/usr/bin/env bash

CONFIG="../configs/retinanet/retinanet_mnv3_aa_blocks_fpn_1x_coco.py"
# CONFIG="../configs/retinanet/retinanet_r50_fpn_1x_coco_custom.py"
GPUS=3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
