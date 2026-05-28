/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @brief 自包含的 Kernel Runner 头文件。
 *
 * 提供 RunKernel<Kernel> —— 一站式 host 函数。
 *
 * Workspace 由外部（torch 层）统一管理，在加载 .so 后 / 首次 kernel 调用前调用：
 *   CatlassSetWorkspaceAlloc(myAlloc);
 *
 * 如果未设置，RunKernel 回退到 aclrtMalloc（裸 NPU 内存）。
 */

#ifndef OPTEST_KERNELS_COMMON_KERNEL_RUNNER_H
#define OPTEST_KERNELS_COMMON_KERNEL_RUNNER_H

#include <cstdint>
#include <type_traits>

#include <acl/acl.h>

#include "catlass/catlass.hpp"
#include "catlass/status.hpp"

#include "common/workspace_alloc.h"

#ifndef KERNEL_NAME
#define KERNEL_NAME undefined
#endif

#ifndef KERNEL_TYPE
#define KERNEL_TYPE
#endif

namespace Catlass {

// ── 设备端 kernel 入口 ──

template <class Kernel>
CATLASS_GLOBAL KERNEL_TYPE void KERNEL_NAME(typename Kernel::Params params)
{
    Kernel kernel;
    kernel(params);
}

// ── 一站式 host 启动 ──

template <class Kernel>
inline void RunKernel(typename Kernel::Arguments args, aclrtStream stream, uint32_t coreNum)
{
    if (!Kernel::CanImplement(args)) {
        return;
    }
    size_t wsSize = Kernel::GetWorkspaceSize(args);
    uint8_t* ws = nullptr;
    if (wsSize > 0) {
        if (g_catlassWorkspaceAlloc) {
            ws = g_catlassWorkspaceAlloc(wsSize);
        } else {
            aclrtMalloc(reinterpret_cast<void**>(&ws), wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        }
    }
    auto params = Kernel::ToUnderlyingArguments(args, ws);
    KERNEL_NAME<Kernel><<<coreNum, nullptr, stream>>>(params);
}

} // namespace Catlass

#endif // OPTEST_KERNELS_COMMON_KERNEL_RUNNER_H
