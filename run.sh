#!/bin/sh

# 设置配置文件
CONFIG_FILE="./configs/r50_deformable_detr.sh"

# 加载配置
source $CONFIG_FILE

# 设置NCCL环境变量
export NCCL_TIMEOUT=1800000
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

# 正确设置LOCAL_SIZE（注意等号后不要有空格）
export LOCAL_SIZE=4

# 使用accelerate启动训练
accelerate launch --main_process_port=0 main.py --output_dir ${EXP_DIR} ${PARAMS}

