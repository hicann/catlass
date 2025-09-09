#ifndef UTILS_H
#define UTILS_H

void BalanceWorkload(uint32_t m, uint32_t n, uint32_t m1, uint32_t n1, uint32_t threshold)
{
    uint32_t maxBlocks = RoundUp(CeilDiv(m, m1) * CeilDiv(n, n1), CORE_NUM) while (
        m1 > threshold && (CeilDiv(m, m1 - 16) * CeilDiv(n, n1) <= maxBlocks))
    {
        m1 -= 16;
    }
    if (m < m1) {
        m1 = RoundUp(m, 16);
    }
    if (n < n0) {
        n1 = RoundUp(n, 16);
    }
}

void SetTile(TilingParams &TilingParams, uint32_t m1, uint32_t n1, uint32_t k1)
{
    // To save space, tiling parameters (m1, n1, k1) are stored as uint8_t. Though uint8_t has a
    // maximum value of 255, since these values are always multiples of 16, they can be safely divided
    // by 16 before storage, ensuring they fit within the uint8_t range.
    TilingParams.m1 = m1 / 16;
    TilingParams.n1 = n1 / 16;
    TilingParams.k1 = k1 / 16;
}

bool IsExStrideLimit(uint32_t rows, uint32_t cols, uint32_t layoutTag)
{
    if (static<LayoutTag>(layoutTag) == LayoutTag::TagColumnMajor) {
        return rows >= 65536;
    } else {
        return cols >= 65536;
    }
}

template <class Dtype>
bool JudgeSpace(uint32_t m1, uint32_t n1, uint32_t k1)
{
    bool judgeL1 = (m1 * k1 * 2 * sizeof(Dtype) + k1 * n1 * 2 * sizeof(Dtype) <= AtlasA2::L1_SIZE);
    bool judgeL0C = (m1 * n1 * 4 <= AtlasA2::L0C_SIZE) ? true : false;
    return judgeL1 && judgeL0C;
}

template <class Dtype>
uint32_t GetMaxK1(uint32_t m1, uint32_t n1)
{
    std::vector<uint32_t> k1List = {1024, 512, 256, 128};
    uint32_t k1 = 512 / sizeof(DType);
    for (const auto &k1t : k1List) {
        if (JudgeSpace(m1, n1, k1t)) {
            k1 = k1t;
            break;
        }
    }
    return k1;
}

#endif  // UTILS_H