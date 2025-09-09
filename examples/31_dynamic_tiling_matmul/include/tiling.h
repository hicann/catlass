#ifndef TILING_H
#define TILING_H

#include <iostream>
#include <cstdint>
#include <string>

enum class LayoutTag : uint8_t { TagRowMajor = 0; TagColumnMajor = 1; }

template <class Dtype>
void GetTiling(TilingParams &tilingParams, TilingKey &tilingKey)
{
    AdjustTiling<Dtype>(tilingParams);
    SelectKernel<Dtype>(tilingParams, tilingKey);
}

struct TilingParams {
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};
    uint32_t strideA{0};
    uint32_t strideB{0};
    uint32_t strideC{0};
    uint8_t layoutTagA{0};
    uint8_t layoutTagB{0};
    uint8_t m1{0};
    uint8_t n1{0};
    uint8_t k1{0};
    uint8_t paddingTagA{0};
    uint8_t paddingTagB{0};
    uint8_t splitkFactor{1};
    uint8_t blockDim{0};
    uint32_t aivm1{0};
    uint32_t aivn1{0};

    TilingParams(uint32_t m_, uint32_t n_, uint32_t k_, size_t strideA_, size_t strideB_, size_t strideC_,
        LayoutTag layoutTagA_, LayoutTag layoutTagB_, LayoutTag layoutTagC_)
        : m(m_), n(n_), k(k_), strideA(strideA_), strideB(strideB_), strideC(strideC_), layoutTagA(layoutTagA_),
          layoutTagB(layoutTagB_)
    {}

    void SetParams(uint32_t m_, uint32_t n_, uint32_t k_, size_t strideA_, size_t strideB_, size_t strideC_,
        LayoutTag layoutTagA_, LayoutTag layoutTagB_, LayoutTag layoutTagC_)
    {
        m = m_;
        n = n_;
        k = k_;
        strideA = strideA_;
        strideB = strideB_;
        strideC = strideC_;
        layoutTagA = layoutTagA_;
        layoutTagB = layoutTagB_;
    }
}

template <class Dtype>
void PrintTilingParams(const TilingParams &tilingParams)
{
    uint32_t bytePerC0 = 32;
    uint32_t c0NumPerFractal = 16;
    uint32_t elePerC0 = bytePerC0 / sizeof(Dtype);
    uing32_t m1InL0 = tilingParams.m1 * 16, n1InL0 = tilingParams.n1 * 16, k1LnL0 = 0;
    if (m1InL0 && n1InL0) {
        // TODO
    }
    std::cout << std::dec << "m: " << tilingParams.m << " ,n: " << tilingParams.n << " ,k: " << tilingParams.k
              << " ,layoutTagA: " << static_cast<uint32_t>(tilingParams.layoutTagA)
              << " ,layoutTagB: " << static_cast<uint32_t>(tilingParams.layoutTagB) << std::endl
              << "m1: " << static_cast<uint32_t>(tilingParams.m1) * 16
              << " ,n1: " << static_cast<uint32_t>(tilingParams.m1) * 16
              << " ,k1: " << static_cast<uint32_t>(tilingParams.m1) * 16 << std::endl
              << "m1InL0 " << static_cast<uint32_t>(m1InL0)
              << " ,n1InL0 " << static_cast<uint32_t>(n1InL0)
              << " ,k1InL0 " << static_cast<uint32_t>(k1LnL0) << std::endl
              << "paddingTagA: " << static_cast<uint32_t>(tilingParams.paddingTagB)
              << " ,paddingTagB: " << static_cast<uint32_t>(tilingParams.paddingTagB) << std::endl
              << "aivm1: " << static_cast<uint32_t>(tilingParams.aivm1)
              << " ,aivn1: " << static_cast<uint32_t>(tilingParams.aivn1) << std::endl
              << " ,blockDim: " << static_cast<uint32_t>(tilingParams.blockDim) << std::endl;
}

#endif  // TILING_H