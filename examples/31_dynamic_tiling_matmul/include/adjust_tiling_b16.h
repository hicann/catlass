#ifndef ADJUST_TILING_B16_H
#define ADJUST_TILING_B16_H

#include "utils.h"
#include "tiling_params.h"
#include "platform_info.h"

using fp16_t = __fp16;

void AdjustTilingB16Layout00(TilingParams &tilingParams, PlatformInfo& platformInfo)
{
    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.m;
    uint32_t k = tilingParams.m;
    uint32_t m1 = 128, n1 = 256, k1 = 256;

    if (n >= 256) {
        // n0 = 256 delivers optimal bandwidth performance.
        uint32_t maxBlocks = RoundUpHost(CeilDivHost(m, m1) * CeilDivHost(n, n1), platformInfo.coreNum);
        BalanceWorkload(m, n, m1, n1, 32);
        uint32_t blocks = CeilDivHost(m, 64) * CeilDivHost(n, 512);
        if (blocks < maxBlocks - platformInfo.coreNum && k <= 128) {
            m1 = 64;
            n1 = 512;
        }
    } else {
        m1 = 128;
        n1 = RoundUpHost(n, 16);
        BalanceWorkload(m, n, m1, n1, 32);
        uint32_t maxBlocks = RoundUpHost(CeilDivHost(m, m1) * CeilDivHost(n, n1), platformInfo.coreNum);
        uint32_t m1t = m1;
        while (JudgeSpace<fp16_t>(m1t + 16, n1, k1, platformInfo)) {
            m1t += 16;
            uint32_t blocks = CeilDivHost(m, m1t) * CeilDivHost(n, n1);
            if (blocks <= maxBlocks - platformInfo.coreNum) {
                m1 = m1t;
            }
        }
    }
    if (k > 65536 || n > 65536) {
        m1 = 128;
        n1 = 256;
    }
    k1 = GetMaxK1<fp16_t>(m1, n1);
    SetTile(tilingParams, m1, n1, k1);
}

void AdjustTilingB16Layout01(TilingParams &tilingParams, PlatformInfo& platformInfo)
{
    uint32_t m1 = 128, n1 = 256, k1 = 256;
    SetTile(tilingParams, m1, n1, k1);
}

void AdjustTilingB16Layout10(TilingParams &tilingParams, PlatformInfo& platformInfo)
{
    uint32_t m1 = 128, n1 = 256, k1 = 256;
    SetTile(tilingParams, m1, n1, k1);
}

void AdjustTilingB16Layout11(TilingParams &tilingParams, PlatformInfo& platformInfo)
{
    uint32_t m1 = 256, n1 = 128, k1 = 256;
    SetTile(tilingParams, m1, n1, k1);
}

using FuncType = void (*)(TilingParams &tilingParams, PlatformInfo& platformInfo);
std::array<std::array<FuncType, 2>, 2> AdjustTilingB16 = {
    {{{AdjustTilingB16Layout00, AdjustTilingB16Layout01}}, {{AdjustTilingB16Layout10, AdjustTilingB16Layout11}}}};
#endif  // ADJUST_TILING_B16_H