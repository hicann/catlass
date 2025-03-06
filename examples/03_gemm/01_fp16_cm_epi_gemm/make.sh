#!/bin/bash
clear;
export ASCEND_HOME_DIR=/home/workspace/gpf/CANN/ascend-toolkit/latest
source /home/workspace/gpf/CANN/ascend-toolkit/set_env.sh
../../../scripts/build.sh 01_fp16_cm_epi_gemm
./run.sh 1243 8192 2423 3 # 存在爆内存的情况 M 或 N 655 感觉像是流水的问题 