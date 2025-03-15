#!/bin/bash
clear;
export ASCEND_HOME_DIR=/home/workspace/gpf/CANN/ascend-toolkit/latest
source /home/workspace/gpf/CANN/ascend-toolkit/set_env.sh
../../../scripts/build.sh 01_fp16_cm_gemm
rm -rf ./data/input/* rm -rf ./data/output/* 
./run.sh 432 2861 355 3 # 存在爆内存的情况 M 或 N 655 感觉像是流水的问题 
# M : 6392 , N : 5389 , K : 2369
# ./run.sh 6392 5389 2369 3 # 存在爆内存的情况 M 或 N 655 感觉像是流水的问题 