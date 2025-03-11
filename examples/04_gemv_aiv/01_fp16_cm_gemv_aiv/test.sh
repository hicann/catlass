#!/bin/bash
export ASCEND_HOME_DIR=/home/workspace/gpf/CANN/ascend-toolkit/latest
source /home/workspace/gpf/CANN/ascend-toolkit/set_env.sh
clear;
rm -rf ./data/output.txt
touch ./data/output.txt
../../../scripts/build.sh 01_fp16_cm_gemv_aiv
for (( i=1; i<=50; i++ ))
do
    num1=$[ ($RANDOM % 10000 + 1) ]
    num2=$[ ($RANDOM % 10000 + 1) ]
    echo $num1 $num2
    ./run.sh $num1 $num2 0 0
done