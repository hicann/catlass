/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTEST_X_ATTENTION_DEVICE_HPP
#define OPTEST_X_ATTENTION_DEVICE_HPP

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "x_attention_helper.hpp"

using namespace AscendC;

template <typename InputType, bool SharedPaged, bool UnsharedPaged>
CATLASS_GLOBAL void XAttention(
    uint64_t hardwareSyncAddr, GM_ADDR query, GM_ADDR sharedKey, GM_ADDR sharedValue, GM_ADDR unsharedKey,
    GM_ADDR unsharedValue, GM_ADDR unsharedBlockTable, GM_ADDR sharedKvLens, GM_ADDR decodeStep,
    GM_ADDR sharedBlockTable, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::SetSyncBaseAddr(hardwareSyncAddr);

    __gm__ XAttentionTilingData* tilingData = reinterpret_cast<__gm__ XAttentionTilingData*>(tiling);
    GM_ADDR s = workspace;
    GM_ADDR p = s + tilingData->mm1OutSize;
    GM_ADDR oTemp = p + tilingData->smOnlineOutSize;
    GM_ADDR oUpdate = oTemp + tilingData->mm2OutSize;
    GM_ADDR sharedWorkspace = oUpdate + tilingData->updateSize;
    GM_ADDR unsharedWorkspace = sharedWorkspace + tilingData->sharedWorkspaceSize;

    XAttnKernelParams params{
        query, sharedKey, sharedValue, unsharedKey, unsharedValue, sharedBlockTable, unsharedBlockTable,
        sharedKvLens, decodeStep, s, p, oTemp, oUpdate, sharedWorkspace, unsharedWorkspace, output, tiling};

    int64_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
    if (coreIdx < tilingData->sharedCoreNum) {
        CallSharedInferKernelShort<InputType, SharedPaged>(params, tilingData);
    } else {
        CallUnsharedInferKernel<InputType, UnsharedPaged>(params, tilingData);
    }
    AscendC::SyncAll<false>();
    CallCombineScale<InputType>(params, tilingData);
}

#endif // OPTEST_X_ATTENTION_DEVICE_HPP
