#!/bin/bash
clear;
export ASCEND_HOME_DIR=/home/workspace/gpf/CANN/ascend-toolkit/latest
source /home/workspace/gpf/CANN/ascend-toolkit/set_env.sh
# export ASCEND_HOME_DIR=/home/workspace/wyf/Ascend/ascend-toolkit/latest
# source /home/workspace/wyf/Ascend/ascend-toolkit/set_env.sh
./run.sh 512 32 0