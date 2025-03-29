#!/bin/bash
export ASCEND_HOME_DIR=/home/workspace/gpf/CANN/ascend-toolkit/latest
source /home/workspace/gpf/CANN/ascend-toolkit/set_env.sh
# export ASCEND_SLOG_PRINT_TO_STDOUT=0




../../scripts/build.sh 18_gemv_aic
for (( i=1; i<=10; i++ ))
do
    num1=$[ $RANDOM % 1000 + 1 ]
    num2=$[ $RANDOM % 1000 + 1 ]
    echo $num1 $num2 
    ../../build/bin/18_gemv_aic $num1 $num2
done