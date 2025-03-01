#!/bin/bash
clear;
export ASCEND_HOME_DIR=/home/workspace/gpf/CANN/ascend-toolkit/latest
source /home/workspace/gpf/CANN/ascend-toolkit/set_env.sh
../../../scripts/build.sh 01_fp16_rm_epi_gemm
./run.sh 9872 8107 845 2 0 # 存在爆内存的情况 M 或 N 655 感觉像是流水的问题 