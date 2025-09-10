#ifndef BASE_INFO_H
#define BASE_INFO_H

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"

constexpr uint32_t CORE_NUM = 20;

enum class LayoutTag : uint8_t { TagRowMajor = 0, TagColumnMajor = 1};

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
        : m(m_), n(n_), k(k_), strideA(strideA_), strideB(strideB_), strideC(strideC_),
          layoutTagA(static_cast<uint8_t>(layoutTagA_)), layoutTagB(static_cast<uint8_t>(layoutTagB_))
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
        layoutTagA = static_cast<uint8_t>(layoutTagA_);
        layoutTagB = static_cast<uint8_t>(layoutTagB_);
    }
};

#endif // BASE_INFO_H
