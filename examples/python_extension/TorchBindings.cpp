/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>

#include "AscendCTKernelWrapper.h"

using namespace AscendCTKernelWrapper;

at::Tensor RunBasicMatmulTorch(const at::Tensor &mat1, const at::Tensor &mat2, const std::string &outDType)
{
    return RunBasicMatmul(mat1.to(GetAtDevice()), mat2.to(GetAtDevice()), outDType);
}

TORCH_LIBRARY(AscendCTTorch, m) { m.def("basic_matmul(Tensor mat1, Tensor mat2, str c) -> Tensor"); }

TORCH_LIBRARY_IMPL(AscendCTTorch, CPU, m) { m.impl("basic_matmul", &RunBasicMatmulTorch); }