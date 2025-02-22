#!/bin/bash
clear;
export ASCEND_HOME_DIR=/home/workspace/gpf/CANN/ascend-toolkit/latest
source /home/workspace/gpf/CANN/ascend-toolkit/set_env.sh
../../../scripts/build.sh 04_int8_cm_gemm
./run.sh 2000 3000 4000 5  # 存在爆内存的情况 M 或 N 655 K 800 极限 感觉像是流水的问题 