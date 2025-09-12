/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR dataA PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_INFO_H
#define BASE_INFO_H

enum class LayoutTag : uint8_t { TagRowMajor = 0, TagColumnMajor = 1};

struct TilingParams {
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};
    uint64_t strideA{0};
    uint64_t strideB{0};
    uint64_t strideC{0};
    uint16_t m1{0};
    uint16_t n1{0};
    uint16_t k1{0};
    uint16_t splitkFactor{1};
    // The following parameters are only used in tiling and are not read by the kernel.
    uint8_t layoutTagA;
    uint8_t layoutTagB;
    uint8_t layoutTagC;
    uint8_t paddingTagA;
    uint8_t paddingTagB;
    uint8_t paddingTagC;
    uint8_t blockDim{0};

    TilingParams() {}

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
