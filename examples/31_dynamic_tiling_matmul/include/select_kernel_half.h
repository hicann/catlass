/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR dataA PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SELECT_KERNEL_HALF_H
#define SELECT_KERNEL_HALF_H

#include "platform_info.h"
#include "launch_map.h"

bool CommonMatmulHandler(TilingParams &params, TilingKey &tilingKey, PlatformInfo& platformInfo)
{
    uint8_t kernelSerial = 0;
    // kernelSerial, layoutTagA, layoutTagB, layoutTagC, paddingTagA, paddingTagB, paddingTagC, dtype(defalut 0).
    tilingKey.SetTilingKey(kernelSerial, params.layoutTagA, params.layoutTagB, 0, 0, 0, 0);
    return true;
}

void SelectKernelHalf(TilingParams &tilingParams, TilingKey &tilingKey, PlatformInfo& platformInfo)
{
    using HandlerPtr = bool (*)(TilingParams& tilingParams, TilingKey& tilingKey, PlatformInfo& platformInfo);
    HandlerPtr handlers[] = {
        CommonMatmulHandler
    };

    for (auto handler : handlers) {
        if (handler(tilingParams, tilingKey, platformInfo)) {
            break;
        }
    }

    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.n;
    uint32_t m1 = tilingParams.m1;
    uint32_t n1 = tilingParams.n1;

    uint32_t tasksAic = CeilDiv(m, m1) * CeilDiv(n, n1) * tilingParams.splitkFactor;
    uint32_t blockDimAic = tasksAic > platformInfo.coreNum ? platformInfo.coreNum : tasksAic;

    tilingParams.blockDim = blockDimAic;
}

#endif  // SELECT_KERNEL_HALF_H