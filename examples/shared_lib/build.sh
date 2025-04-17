#!/bin/bash

# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

for i in "$@"
do
    case $i in
        --shared_lib_src_dir=*)
            SHARED_LIB_SRC_DIR="${i#*=}"
            shift # past argument=value
        ;;
        --output_path=*)
            OUTPUT_PATH="${i#*=}"
            shift # past argument=value
        ;;
        --act_src_dir=*)
            ACT_SRC_DIR="${i#*=}"
    esac
done

mkdir -p $OUTPUT_PATH

function build_lib() {
    TYPE=$1
    if [[ $TYPE == "static" ]]; then
        OUTPUT_NAME=libact_kernel.a
        COMPILE_OPTION="--cce-build-static-lib"
    elif [[ $TYPE == "shared" ]]; then
        OUTPUT_NAME=libact_kernel.so
        COMPILE_OPTION="--shared"
    else
        echo "Unknown type: $TYPE"
        exit 1
    fi
    echo -e "Building CXX shared library $OUTPUT_NAME"
    bisheng -O2 -fPIC -std=c++17 -xcce --cce-aicore-arch=dav-c220 \
    -I$ASCEND_HOME_PATH/compiler/tikcpp \
    -I$ASCEND_HOME_PATH/compiler/tikcpp/tikcfw \
    -I$ASCEND_HOME_PATH/compiler/tikcpp/tikcfw/impl \
    -I$ASCEND_HOME_PATH/compiler/tikcpp/tikcfw/interface \
    -I$ASCEND_HOME_PATH/include \
    -I$ASCEND_HOME_PATH/include/experiment/runtime \
    -I$ASCEND_HOME_PATH/include/experiment/msprof \
    -I$SHARED_LIB_SRC_DIR \
    -I$SHARED_LIB_SRC_DIR/impl \
    -I$ACT_SRC_DIR/include \
    -DL2_CACHE_HINT \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=true \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    -Wno-macro-redefined -Wno-ignored-attributes \
    -L$ASCEND_HOME_PATH/lib64 \
    -lruntime \
    $SHARED_LIB_SRC_DIR/act_kernel.cpp $COMPILE_OPTION -o $OUTPUT_PATH/$OUTPUT_NAME
    echo -e "Built $OUTPUT_NAME"
}

cp $SHARED_LIB_SRC_DIR/act_kernel.h $OUTPUT_PATH/
build_lib shared & build_lib static
