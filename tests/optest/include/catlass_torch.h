/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */
 
#ifndef OPTEST_CATLASS_TORCH_H
#define OPTEST_CATLASS_TORCH_H

#include <torch/extension.h>

at::Tensor basic_matmul(
    const at::Tensor& mat1, const at::Tensor& mat2, const std::string& outDType, const bool transA, const bool transB,
    const bool formatA, const bool formatB);

#endif