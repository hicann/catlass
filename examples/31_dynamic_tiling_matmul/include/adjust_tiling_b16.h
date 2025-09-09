#ifndef ADJUST_TILING_B16_H
#define ADJUST_TILING_B16_H

using fp16_t = __fp16;

void AdjustTilingB16Layout00(TilingParams &tilingParams)
{
    uint32_t m = tilingParams.m;
    uint32_t n = tilingParams.m;
    uint32_t k = tilingParams.m;
    uint32_t m1 = 128, n1 = 256, k1 = 256;

    if (n >= 256) {
        // n0 = 256 delivers optimal bandwidth performance.
        uint32_t maxBlocks = RoundUp(CeilDiv(m, m1) * CeilDiv(n, n1), CORE_NUM);
        BalanceWorkload(m, n, m1, n1, 32);
        uint32_t blocks = CeilDiv(m, 64) * CeilDiv(n, 512);
        if (blocks < maxBlocks - CORE_NUM && k <= 128) {
            m1 = 64;
            n1 = 512;
        }
    } else {
        m1 = 128;
        n1 = RoundUp(n, 16);
        BalanceWorkload(m, n, m1, n1, 32);
        uint32_t maxBlocks = RoundUp(CeilDiv(m, m1) * CeilDiv(n, n1), CORE_NUM);
        uint32_t m1t = m1;
        while (JudgeSpace<fp16_t>(m1t + 16, n1, k1)) {
            m1t += 16;
            uint32_t blocks = CeilDiv(m, m1t) * CeilDiv(n, n1);
            if (blocks <= maxBlocks - CORE_NUM) {
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

void AdjustTilingB16Layout01(TilingParams &tilingParams)
{
    uint32_t m1 = 128, n1 = 256, k1 = 256;
    SetTile(tilingParams, m1, n1, k1);
}

void AdjustTilingB16Layout10(TilingParams &tilingParams)
{
    uint32_t m1 = 128, n1 = 256, k1 = 256;
    SetTile(tilingParams, m1, n1, k1);
}

void AdjustTilingB16Layout11(TilingParams &tilingParams)
{
    uint32_t m1 = 256, n1 = 128, k1 = 256;
    SetTile(tilingParams, m1, n1, k1);
}

using FuncType = void (*)(TilingParams &tilingParams);
std::array<std::array<FuncType, 2>, 2> AdjustTilingB16 = {
    {{{AdjustTilingB16Layout00, AdjustTilingB16Layout01}}, {{AdjustTilingB16Layout10, AdjustTilingB16Layout11}}}};
#endif  // ADJUST_TILING_B16_H