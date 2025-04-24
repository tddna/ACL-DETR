#!/usr/bin/env bash

export NCCL_TIMEOUT=3600000
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0

set -x

EXP_DIR=exps/r50_deformable_detr
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
