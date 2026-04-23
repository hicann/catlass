/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_TORCH_RUN_NPU_FUNC_H
#define CATLASS_TORCH_RUN_NPU_FUNC_H

#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>

#define RUN_NPU_FUNC(func, ...)                                                                                    \
    do {                                                                                                           \
        if ((func) == nullptr) {                                                                                   \
            throw std::runtime_error(                                                                              \
                std::string("Function pointer is null at ") + __FILE__ + ":" + std::to_string(__LINE__) + " in " + \
                #func);                                                                                            \
        }                                                                                                          \
    } while (false) at_npu::native::OpCommand::RunOpApiV2(#func, [=]() -> aclError { return func(__VA_ARGS__); });

#endif