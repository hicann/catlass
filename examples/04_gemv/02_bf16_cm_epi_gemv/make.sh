#!/bin/bash
clear;
export ASCEND_HOME_DIR=/home/workspace/gpf/CANN/ascend-toolkit/latest
source /home/workspace/gpf/CANN/ascend-toolkit/set_env.sh
# ../../../scripts/build.sh 01_fp16_rm_epi_gemv
bash ../../../scripts/build.sh 02_bf16_cm_epi_gemv

./run.sh 7680 8320


# M : 7680 , N : 8320 