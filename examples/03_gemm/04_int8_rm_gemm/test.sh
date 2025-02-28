#!/bin/bash
export ASCEND_HOME_DIR=/home/workspace/gpf/CANN/ascend-toolkit/latest
source /home/workspace/gpf/CANN/ascend-toolkit/set_env.sh
clear;
rm -rf ./data/output.txt
../../../scripts/build.sh 04_int8_rm_gemm
for (( i=1; i<=3; i++ ))
do
    num1=$[ $RANDOM % 1000 + 1 ]
    num2=$[ $RANDOM % 1000 + 1 ]
    num3=$[ $RANDOM % 1000 + 1 ]
    echo $num1 $num2 $num3
    ./run.sh $num1 $num2 $num3 2
done