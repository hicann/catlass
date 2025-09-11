#ifndef ADJUST_TILING_B16_H
#define ADJUST_TILING_B16_H

#include "utils.h"
#include "tiling_params.h"
#include "platform_info.h"
#include "fp16_t.h"
using fp16_t = op::fp16_t;

void AdjustTilingB16Layout00(TilingParams &tilingParams, PlatformInfo& platformInfo)
{
    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.m;
    uint32_t k = tilingParams.m;
    uint32_t m1 = 128, n1 = 256, k1 = 256;

    if (n >= 256) {
        // n0 = 256 delivers optimal bandwidth performance.
        uint32_t maxBlocks = RoundUpHost(CeilDivHost(m, m1) * CeilDivHost(n, n1), platformInfo.coreNum);
        BalanceWorkload(m, n, m1, n1, 32, platformInfo);
        uint32_t blocks = CeilDivHost(m, 64) * CeilDivHost(n, 512);
        if (blocks < maxBlocks - platformInfo.coreNum && k <= 128) {
            m1 = 64;
            n1 = 512;
        }
    } else {
        m1 = 128;
        n1 = RoundUpHost(n, 16);
        BalanceWorkload(m, n, m1, n1, 32, platformInfo);
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
    if (k >= 65536 || n > 65536) {
        m1 = 128;
        n1 = 256;
    }
    k1 = GetMaxK1<fp16_t>(m1, n1, platformInfo);
    SetTile(tilingParams, m1, n1, k1);
}

void AdjustTilingB16Layout01(TilingParams &tilingParams, PlatformInfo& platformInfo)
{
    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.m;
    uint32_t k = tilingParams.m;
    uint32_t m1 = 128, n1 = 256, k1 = 256;
    // When LayoutA is RowMajor and LayoutB is ColumnMajor, bandwidth issues can be completely disregarded,
    // simply choose the tiling configureation with the most balanced workload
    double ratio = (double)(m * k + k * n) / (m * n);
    if (m > n && (ratio > 0.1 || n < 256)) {
        m1 = 256;
        n1 = 128;
        BalanceWorkload(m, n, m1, n1, 64, platformInfo);
        BalanceWorkload(n, m, n1, m1, 64, platformInfo); 
    } else {
        BalanceWorkload(n, m, n1, m1, 64, platformInfo);
        BalanceWorkload(m, n, m1, n1, 64, platformInfo); 
    }
    uint32_t maxBlocks = RoundUpHost(CeilDivHost(m, m1) * CeilDivHost(n, n1), platformInfo.coreNum);
    if (m < n) {
        uint32_t n1t = n1;
        while (JudgeSpace<fp16_t>(m1, n1t + 16, k1, platformInfo)) {
            n1t += 16;
            uint32_t blocks = CeilDivHost(m, m1) * CeilDivHost(n, n1t);
            if (blocks <= maxBlocks - platformInfo.coreNum) {
                n1 = n1t;
            }
        }
    } else {
        uint32_t m1t = m1;
        while (JudgeSpace<fp16_t>(m1t + 16, n1, k1, platformInfo)) {
            m1t += 16;
            uint32_t blocks = CeilDivHost(m, m1t) * CeilDivHost(n, n1);
            if (blocks <= maxBlocks - platformInfo.coreNum) {
                m1 = m1t;
            }
        }
    }
    if (k > 65536) {
        if (m < n || (ratio < 0.1 && n >= 256)) {
            m1 = 128;
            n1 = 256;
        } else {
            m1 = 256;
            n1 = 128;
        }
    }
    k1 = GetMaxK1<fp16_t>(m1, n1, platformInfo);
    SetTile(tilingParams, m1, n1, k1);
}

void AdjustTilingB16Layout10(TilingParams &tilingParams, PlatformInfo& platformInfo)
{
    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.m;
    uint32_t k = tilingParams.m;
    uint32_t m1 = 128, n1 = 256, k1 = 256;
    double ratio = (double)(m * k + k * n) / (m * n);
    if (m > n && (ratio > 0.1 || n < 256)) {
        m1 = 256;
        n1 = 128;
        k1 = 256;
    }
    if (m < m1) {
        m1 = RoundUpHost(m, 16);
    }
    if (n < n1) {
        n1 = RoundUpHost(n, 16);
    }

    uint32_t blocks = CeilDivHost(m, m1) * CeilDivHost(n, n1);
    if (blocks <= platformInfo.coreNum / 4) {
        if (n1 > 16) {
            n1 /= 2;
        }
        if (m1 > 16) {
            m1 /= 2;
        }
    } else if (blocks <= platformInfo.coreNum / 2) {
        if (m1 > n1) {
            m1 /= 2;
        } else if (n1 > 16) {
            n1 /= 2;
        }
    }
    if (n >= 65536 || m >= 65536) {
        if (m < n || (ratio < 0.1 && n >= 256)) {
            m1 = 128;
            n1 = 256;
        } else {
            m1 = 256;
            n1 = 128;
        }
    }
    k1 = GetMaxK1<fp16_t>(m1, n1, platformInfo);
    SetTile(tilingParams, m1, n1, k1); 
}

void AdjustTilingB16Layout11(TilingParams &tilingParams, PlatformInfo& platformInfo)
{
    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.m;
    uint32_t k = tilingParams.m;
    uint32_t m1 = 256, n1 = 128, k1 = 256;

    if (m >= 256) {
        // n0 = 256 delivers optimal bandwidth performance.
        uint32_t maxBlocks = RoundUpHost(CeilDivHost(m, m1) * CeilDivHost(n, n1), platformInfo.coreNum);
        BalanceWorkload(n, m, n1, m1, 32, platformInfo);
        uint32_t blocks = CeilDivHost(n, 64) * CeilDivHost(m, 512);
        if (blocks < maxBlocks - platformInfo.coreNum && k <= 128) {
            n1 = 64;
            m1 = 512;
        }
    } else {
        n1 = 128;
        m1 = RoundUpHost(m, 16);
        BalanceWorkload(n, m, n1, m1, 32, platformInfo);
        uint32_t maxBlocks = RoundUpHost(CeilDivHost(m, m1) * CeilDivHost(n, n1), platformInfo.coreNum);
        uint32_t n1t = n1;
        while (JudgeSpace<fp16_t>(n1t + 16, m1, k1, platformInfo)) {
            n1t += 16;
            uint32_t blocks = CeilDivHost(n, n1t) * CeilDivHost(m, m1);
            if (blocks <= maxBlocks - platformInfo.coreNum) {
                n1 = n1t;
            }
        }
    }
    if (k >= 65536 || m > 65536) {
        m1 = 256;
        n1 = 128;
    }
    // fixpipe bound
    double ratio = (double)(m * k + k * n) / (m * n);
    if (ratio < 0.1 && n >= 256) {
        m1 = 128;
        n1 = 256;
    }
    k1 = GetMaxK1<fp16_t>(m1, n1, platformInfo);
    SetTile(tilingParams, m1, n1, k1);
}

using FuncType = void (*)(TilingParams &tilingParams, PlatformInfo& platformInfo);
std::array<std::array<FuncType, 2>, 2> AdjustTilingB16 = {
    {{{AdjustTilingB16Layout00, AdjustTilingB16Layout01}}, {{AdjustTilingB16Layout10, AdjustTilingB16Layout11}}}};
#endif  // ADJUST_TILING_B16_H