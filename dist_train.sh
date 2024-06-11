#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
WORK_DIR=$3
PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# CUDA_VISIBLE_DEVICES="0,1,2,3" python3 -m torch.distributed.launch  --nproc_per_node=4 --master_port=$((RANDOM + 10000)) \
#     $(dirname "$0")/train.py $CONFIG --work-dir=$WORK_DIR --launcher pytorch ${@:3} 


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch  --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) \
    $(dirname "$0")/train.py $CONFIG --work-dir=$WORK_DIR --launcher pytorch ${@:4}
    
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python3 -m torch.distributed.launch  --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) \
#     $(dirname "$0")/train.py $CONFIG --work-dir=$WORK_DIR --launcher pytorch ${@:4} --resume-from /home/u1120230266/Cascade-CLIP/output/coco-st/latest.pth