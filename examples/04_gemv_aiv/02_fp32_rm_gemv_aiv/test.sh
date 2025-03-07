#!/bin/bash
export ASCEND_HOME_DIR=/home/workspace/gpf/CANN/ascend-toolkit/latest
source /home/workspace/gpf/CANN/ascend-toolkit/set_env.sh
clear;
rm -rf ./data/output.txt
touch ./data/output.txt
../../../scripts/build.sh 02_fp32_rm_gemv_aiv
for (( i=1; i<=5; i++ ))
do
    num1=$[ ($RANDOM % 256 + 1) * 64]
    num2=$[ ($RANDOM % 256 + 1) * 64]
    # 生成 0 到 1 之间的随机浮点数
    num3=$(echo "scale=4; $RANDOM / 32767" | bc)
    num4=$(echo "scale=4; $RANDOM / 32767" | bc)
    echo $num1 $num2 $num3 $num4
    ./run.sh $num1 $num2 $num3 $num4 $5
done