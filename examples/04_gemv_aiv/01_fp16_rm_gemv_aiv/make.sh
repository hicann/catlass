#!/bin/bash
clear;
export ASCEND_HOME_DIR=/home/workspace/gpf/CANN/ascend-toolkit/latest
source /home/workspace/gpf/CANN/ascend-toolkit/set_env.sh
# export ASCEND_HOME_DIR=/home/workspace/wyf/Ascend/ascend-toolkit/latest
# source /home/workspace/wyf/Ascend/ascend-toolkit/set_env.sh
../../../scripts/build.sh 01_fp16_rm_gemv_aiv
./run.sh 32 512 1.0 1.0 0