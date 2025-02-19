#!/bin/bash
clear;
export ASCEND_HOME_DIR=/home/workspace/gpf/CANN/ascend-toolkit/8.0.0.alpha003
source /home/workspace/gpf/CANN/ascend-toolkit/set_env.sh
./run.sh 10403 11322 12403 5  # 存在爆内存的情况 M 或 N 655 感觉像是流水的问题 