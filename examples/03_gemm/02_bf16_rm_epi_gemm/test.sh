#!/bin/bash
export ASCEND_HOME_DIR=/home/workspace/gpf/CANN/ascend-toolkit/latest
source /home/workspace/gpf/CANN/ascend-toolkit/set_env.sh
clear;
rm -rf ./data/output.txt
../../../scripts/build.sh 02_bf16_rm_epi_gemm
for (( i=1; i<=10; i++ ))
do
    num1=$[ $RANDOM % 10000 + 1 ]
    num2=$[ $RANDOM % 10000 + 1 ]
    num3=$[ $RANDOM % 10000 + 1 ]
    echo $num1 $num2 $num3
    ./run.sh $num1 $num2 $num3 2
done