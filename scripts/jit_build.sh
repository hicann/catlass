#!/bin/bash

# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# The kernel compile options in this jit-build script will be integrated in ../examples/CMakeLists.txt in the future.

# default input
LAUNCH_SRC_FILE="_gen_launch.cpp"
OUTPUT_LIB_FILE="_gen_module.so"

if [ $# -ge 1 ] ; then
    LAUNCH_SRC_FILE=$1
fi
if [ $# -ge 2 ]; then
    OUTPUT_LIB_FILE=$2
fi

LAUNCH_OBJ_FILE="${LAUNCH_SRC_FILE%.cpp}.o"
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")

cd "$(dirname "$0")"

# customize your own compile options here
# e.g. add an additional -I parameter to include custom header file directory
bisheng -O2 -std=c++17 -xcce --cce-aicore-arch=dav-c220 \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=true \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    -DL2_CACHE_HINT \
    -I$ASCEND_HOME_PATH/compiler/tikcpp \
    -I$ASCEND_HOME_PATH/compiler/tikcpp/tikcfw \
    -I$ASCEND_HOME_PATH/compiler/tikcpp/tikcfw/impl \
    -I$ASCEND_HOME_PATH/compiler/tikcpp/tikcfw/interface \
    -I$ASCEND_HOME_PATH/include \
    -I$ASCEND_HOME_PATH/include/experiment/runtime \
    -I$ASCEND_HOME_PATH/include/experiment/msprof \
    -I$PYTHON_INCLUDE \
    -I../include \
    -I../examples/common \
    -Wno-macro-redefined -Wno-ignored-attributes \
    -fPIC -c $LAUNCH_SRC_FILE -o $LAUNCH_OBJ_FILE

COMPILE_RET=$? 
if [ $COMPILE_RET -ne 0 ] ; then
    exit $COMPILE_RET
fi

# customize your own linking options here
# e.g. add an additional -l parameter to link to a custom shared library
bisheng --cce-fatobj-link -fPIC --cce-aicore-arch=dav-c220 -shared -o $OUTPUT_LIB_FILE $LAUNCH_OBJ_FILE \
    -L$ASCEND_HOME_PATH/lib64 \
    -lruntime -lstdc++ -lascendcl -lm -ltiling_api -lplatform -lc_sec -ldl -lnnopbase

exit $?