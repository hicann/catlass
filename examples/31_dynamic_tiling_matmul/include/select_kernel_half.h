#ifndef SELECT_KERNEL_HALF_H
#define SELECT_KERNEL_HALF_H

#include "launch_map.h"

bool CommonMatmulHandler(TilingParams& params, TilingKey& tilingKey) {
    uint8_t kernelSerial = 0;
    tilingKey.setTilingKey(kernelSerial, params.layoutTagA, params.layoutTagB, 0, 0, 0);
    return true;
}

void SelectKernelHalf(TilingParams &tilingParams, TilingKey &tilingKey)
{
    using HandlerPtr Handlers[] =
    { CommonMatmulHalfHandler }

    for (auto handler : Handlers)
    {
        if (handler(tilingParams, tilingKey)) {
            break;
        }
    }

    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.n;
    uint32_t m1 = tilingParams.m1 * 16;
    uint32_t n1 = tilingParams.n1 * 16;

    uint32_t tasksAic = CeilDiv(m, m1) * CeilDiv(n, n1) * tilingParams.splitkFactor;
    uint32_t blockDimAic = taskAic > 20 ? 20 : tasksAic;

    tilingParams.blockDim = blockDimAic;
}

#endif  // SELECT_KERNEL_HALF_H