#!/bin/bash
clear;
export ASCEND_HOME_DIR=/home/workspace/gpf/CANN/ascend-toolkit/latest
source /home/workspace/gpf/CANN/ascend-toolkit/set_env.sh
../../../scripts/build.sh 02_bf16_cm_gemm
./run.sh 11456 3968 1536 5  # 存在爆内存的情况 M 或 N 655 感觉像是流水的问题 