#!/usr/bin/env bash

# set -x

# EXP_DIR=exps/r50_deformable_detr
# PY_ARGS=${@:1}

# python -u main.py \
#     --output_dir ${EXP_DIR} \
#     ${PY_ARGS}
#!/usr/bin/env bash

# 只设置参数，不执行命令
EXP_DIR=exps/r50_deformable_detr

# 定义额外参数
PARAMS=${@:1}

# 不要在这里执行Python命令
# 移除或注释下面这几行:
# python -u main.py \
#     --output_dir ${EXP_DIR} \
#     ${PY_ARGS}