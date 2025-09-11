#ifndef SELECT_KERNEL_HALF_H
#define SELECT_KERNEL_HALF_H

#include "platform_info.h"
#include "launch_map.h"

bool CommonMatmulHandler(TilingParams &params, TilingKey &tilingKey, PlatformInfo& platformInfo)
{
    uint8_t kernelSerial = 0;
    tilingKey.SetTilingKey(kernelSerial, params.layoutTagA, params.layoutTagB, 0, 0, 0);
    return true;
}

void SelectKernelHalf(TilingParams &tilingParams, TilingKey &tilingKey, PlatformInfo& platformInfo)
{
    using HandlerPtr = bool (*)(TilingParams& tilingParams, TilingKey& tilingKey, PlatformInfo& platformInfo);
    HandlerPtr handlers[] = {CommonMatmulHandler};

    for (auto handler : handlers) {
        if (handler(tilingParams, tilingKey, platformInfo)) {
            break;
        }
    }

    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.n;
    uint32_t m1 = tilingParams.m1 * 16;
    uint32_t n1 = tilingParams.n1 * 16;

    uint32_t tasksAic = CeilDivHost(m, m1) * CeilDivHost(n, n1) * tilingParams.splitkFactor;
    uint32_t blockDimAic = tasksAic > platformInfo.coreNum ? platformInfo.coreNum : tasksAic;

    tilingParams.blockDim = blockDimAic;
}

#endif  // SELECT_KERNEL_HALF_H