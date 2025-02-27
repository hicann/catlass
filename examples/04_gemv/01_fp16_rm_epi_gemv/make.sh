#!/bin/bash
clear;
export ASCEND_HOME_DIR=/home/workspace/gpf/CANN/ascend-toolkit/latest
source /home/workspace/gpf/CANN/ascend-toolkit/set_env.sh
# ../../../scripts/build.sh 01_fp16_rm_epi_gemv
bash ../../../scripts/build.sh 01_fp16_rm_epi_gemv

./run.sh 16256 13120


# 2790 720 错最后6个

# 3813 7176 几乎全错
#   M : 4833 , N : 8003 
#   M : 16256 , N : 13120 
