#!/bin/bash
# 不需要TIK打印出内存信息
# clear
# export ASCEND_HOME_DIR=/home/workspace/gpf/CANN/ascend-toolkit/8.0.0.alpha003
# source /home/workspace/gpf/CANN/ascend-toolkit/set_env.sh
export PRINT_TIK_MEM_ACCESS=FALSE

export ASCEND_SLOG_PRINT_TO_STDOUT=0
# 编译文件
bash ../../../scripts/build.sh 01_fp16_cm_epi_gemv

# 生成测试数据
# python3 ./scripts/gen_data.py $1 $2

# msprof op --output=./prof ../../../build/bin/01_fp16_rm_gemm $M $N $K $deviceId
../../../build/bin/01_fp16_cm_epi_gemv $1 $2 $3 $4
# 验证数据
# python3 ./scripts/verify_data.py $1 $2

rm data/input/*
rm data/output/*

# 性能测试 注意路径问题
# cd ./examples/03_gemm/01_fp16_rm_gemm/
# msprof op simulator --output=../prof ./main
# msprof --application="./main $transA $transB $M $N $alpha $beta $deviceId" --output=../prof 
msprof op --output=./prof --launch-count=5 ../../../build/bin/01_fp16_cm_epi_gemv $1 $2 $3 $4

#--aic-metrics=PipeUtilization