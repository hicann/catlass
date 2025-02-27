#!/bin/bash
export ASCEND_HOME_DIR=/home/workspace/gpf/CANN/ascend-toolkit/latest
source /home/workspace/gpf/CANN/ascend-toolkit/set_env.sh
clear;
rm -rf ./data/output.txt
# 编译文件
bash ../../../scripts/build.sh 03_fp32_rm_epi_gemv

for (( i=1; i<=25; i++ ))
do
    # num1=$[ $RANDOM % 10000 + 1 ]
    # num2=$[ $RANDOM % 10000 + 1 ]
    # num1=$[ ($RANDOM % 256 + 1) * 32]
    # num2=$[ ($RANDOM % 256 + 1) * 32]
    num1=$[ ($RANDOM % 256 + 1) * 64]
    num2=$[ ($RANDOM % 256 + 1) * 64]
    echo $num1 $num2 
    ./run.sh $num1 $num2
done