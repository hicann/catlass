#!/bin/bash
export ASCEND_HOME_DIR=/home/workspace/gpf/CANN/ascend-toolkit/latest
source /home/workspace/gpf/CANN/ascend-toolkit/set_env.sh
clear;
rm -rf ./data/output.txt
../../../scripts/build.sh 03_fp32_cm_epi_groupgemm
for (( i=1; i<=3; i++ ))
do
    num1=$[ $RANDOM % 20 + 1 ]
    echo $num1
    ./run.sh $num1 1
done