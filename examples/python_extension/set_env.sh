#!/bin/bash

# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

PYTHON_PKG_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
TORCH_LIB_PATH=$PYTHON_PKG_PATH/torch/lib
TORCH_NPU_LIB_PATH=$PYTHON_PKG_PATH/torch_npu/lib
SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
KERNEL_PATH=$(dirname "$SCRIPT_PATH")/../shared_lib
KERNEL_REAL_PATH=$(dirname "$SCRIPT_PATH")/../../output/shared_lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TORCH_LIB_PATH:$TORCH_NPU_LIB_PATH:$KERNEL_PATH:$KERNEL_REAL_PATH