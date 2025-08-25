/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include "kernel_operator.h"

constexpr uint16_t TILE_LENGTH = 512;

class L2CacheClean {
public:
    __aicore__ inline L2CacheClean() {}
    __aicore__ inline void Init(__gm__ void* x, uint64_t blockLen)
    {
        xGm.SetGlobalBuffer((__gm__ int8_t *)x + blockLen * AscendC::GetBlockIdx(), blockLen);
        pipe.InitBuffer(inQueueX, 1, TILE_LENGTH * sizeof(int8_t));
        kernelBlockLen = blockLen;
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = kernelBlockLen / TILE_LENGTH;
        int32_t tail = kernelBlockLen - loopCount * TILE_LENGTH;
        AscendC::LocalTensor<int8_t> xLocal = inQueueX.AllocTensor<int8_t>();
        for (int32_t i = 0; i < loopCount; i++) {
            AscendC::DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH);
        }
        AscendC::DataCopy(xLocal, xGm[loopCount * TILE_LENGTH], tail);
        inQueueX.FreeTensor(xLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::A1, 1> inQueueX;
    AscendC::GlobalTensor<int8_t> xGm;
    uint64_t kernelBlockLen;
};

extern "C" __global__ __aicore__ void ClearL2Cache(__gm__ void* x, __gm__ void* tilingSize)
{
    if ASCEND_IS_AIV {
        return;
    }
    L2CacheClean op;
    uint64_t len = *(__gm__ uint64_t *)tilingSize;
    op.Init(x, len);
    op.Process();
}

namespace Catlass {
void DoClearL2Cache(uint32_t blockDim, void* l2ctrl, void* stream, void* buffer, void* tilingSize)
{
    ClearL2Cache<<<blockDim, l2ctrl, stream>>>(buffer, tilingSize);
}
} // namespace Catlass