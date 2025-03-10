#!/bin/bash
clear;
export ASCEND_HOME_DIR=/home/workspace/gpf/CANN/ascend-toolkit/latest
rm -rf ./sim_result
source /home/workspace/gpf/CANN/ascend-toolkit/set_env.sh
# export LD_LIBRARY_PATH=/home/workspace/gpf/CANN/ascend-toolkit/latest/tools/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH

# ../../../scripts/build.sh 01_fp16_rm_epi_gemv
../../../scripts/build.sh 01_fp16_rm_epi_gemv 
# msprof op simulator --application="../../../build/bin/01_fp16_rm_epi_gemv 640 1024 0 1" --output="./sim_result"

./run.sh 12480 512


#   M : 16256 , N : 768 
#   M : 4672 , N : 1088 
#   M : 12480 , N : 256 
