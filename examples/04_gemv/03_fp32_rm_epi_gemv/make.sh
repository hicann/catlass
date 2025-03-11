#!/bin/bash
clear;
export ASCEND_HOME_DIR=/home/workspace/gpf/CANN/ascend-toolkit/latest
source /home/workspace/gpf/CANN/ascend-toolkit/set_env.sh
# ../../../scripts/build.sh 01_fp16_rm_epi_gemv
bash ../../../scripts/build.sh 03_fp32_rm_epi_gemv

./run.sh 32  257
# ./run.sh 4672 1088
# ./run.sh 12480 256


#   M : 16256 , N : 768 
#   M : 4672 , N : 1088 
#   M : 12480 , N : 256 
#   M : 960 , N : 16256 
