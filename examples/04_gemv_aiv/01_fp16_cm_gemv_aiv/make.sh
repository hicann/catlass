# #!/bin/bash
clear;
export ASCEND_HOME_DIR=/home/workspace/gpf/CANN/ascend-toolkit/latest
source /home/workspace/gpf/CANN/ascend-toolkit/set_env.sh
# export LD_LIBRARY_PATH=/home/workspace/gpf/CANN/ascend-toolkit/latest/tools/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH
# # export ASCEND_HOME_DIR=/home/workspace/wyf/Ascend/ascend-toolkit/latest
# # source /home/workspace/wyf/Ascend/ascend-toolkit/set_env.sh
../../../scripts/build.sh 01_fp16_cm_gemv_aiv
# msprof op simulator --application="../../../build/bin/01_fp16_cm_gemv_aiv 640 1024 0 1" --output="./sim_result"
./run.sh 1055 1024 0 0
