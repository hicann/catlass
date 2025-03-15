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
groupCnt=${1}
set -e
CANN_DIR=${ASCEND_HOME_PATH}
# 生成测试数据  需要修改
python3 ./scripts/gen_data_group.py $groupCnt
python3 ./scripts/gen_data.py $groupCnt
# ../../../scripts/build.sh 01_fp16_rm_epi_groupgemm
# msprof op --output=./prof ../../../build/bin/01_fp16_rm_epi_groupgemm $M $N $K $deviceId
../../../build/bin/01_fp16_rm_epi_groupgemm $groupCnt
# 验证数据
python3 ./scripts/verify_data.py $groupCnt
rm -rf ./data/input ./data/output
# 性能测试 注意路径问题
# cd ./examples/03_gemm/01_fp16_rm_epi_groupgemm/
# msprof op simulator --output=../prof ./main
# msprof --application="./main $transA $transB $M $N $K $alpha $beta $deviceId" --output=../prof #--aic-metrics=PipeUtilization