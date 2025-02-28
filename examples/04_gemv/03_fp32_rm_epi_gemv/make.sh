#!/bin/bash
clear;
export ASCEND_HOME_DIR=/home/workspace/gpf/CANN/ascend-toolkit/latest
source /home/workspace/gpf/CANN/ascend-toolkit/set_env.sh
# ../../../scripts/build.sh 01_fp16_rm_epi_gemv
bash ../../../scripts/build.sh 03_fp32_rm_epi_gemv

./run.sh 4833 8015




# 4833 8004 报错，最后一个数完全算错 aclError:507015
# 4833 8003 报错，最后一个数完全算错 aclError:507015
# 4833 8005 报错，最后一个数完全算错 aclError:507015
# 4833 8006 报错，最后一个数完全算错 aclError:507015
# 4833 8007 报错，最后一个数完全算错 aclError:507015

# 4833 8002 正确