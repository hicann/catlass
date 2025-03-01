#!/bin/bash
# 不需要TIK打印出内存信息
rm -rf ./prof
mkdir -p ./prof ./data ./data/input ./data/output
export PRINT_TIK_MEM_ACCESS=FALSE
OP_NAME=GEMM
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
); cd $CURRENT_DIR
# A 矩阵的形状 M * K
# B 矩阵的形状 K * N
M=${1}
N=${2}
K=${3}
deviceId=${4}
mode=${5}
set -e
CANN_DIR=${ASCEND_HOME_PATH}
# 生成测试数据
python3 ./scripts/gen_data.py $M $N $K

../../../build/bin/01_fp16_cm_epi_gemm $M $N $K $deviceId $mode
# 验证数据
python3 ./scripts/verify_data.py $M $N $K
# ../../../scripts/build.sh 01_fp16_cm_epi_gemm 
# msprof op  --application="../../../build/bin/01_fp16_cm_epi_gemm $M $N $K $deviceId"  --output=./prof
# python3 ./scripts/verify_data.py $M $N $K


rm -rf ./data/input ./data/output
# 性能测试 注意路径问题
# cd ./examples/03_gemm/01_fp16_cm_epi_gemm/
# msprof op simulator --output=../prof ./main
# msprof --application="./main $transA $transB $M $N $K $alpha $beta $deviceId" --output=../prof #--aic-metrics=PipeUtilization