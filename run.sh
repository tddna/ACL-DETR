#!/bin/sh
ps aux | grep python | grep -v grep | awk '{print $2}' | xargs -I{} kill -9 {}
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/r50_deformable_detr.sh
# GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 ./configs/r50_deformable_detr.sh
# GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 ./configs/r50_deformable_detr.sh
# GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/r50_deformable_detr.sh
