/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_ATLAS300I_TILE_CAST_HPP
#define CATLASS_GEMM_TILE_ATLAS300I_TILE_CAST_HPP

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/layout/layout.hpp"
namespace Catlass::Gemm::Tile {
template <class ArchTag, class ElementDst, class ElementSrc>
struct TileCast {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<int8_t>::value;

    CATLASS_DEVICE
    TileCast()
    {
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementDst> const &dstTensor,
        AscendC::LocalTensor<ElementSrc> const &srcTensor,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc
    )
    {
        constexpr uint32_t maxRepeatNum = 255;
        constexpr uint32_t maxMask = 64;
        uint32_t repeatCount = (layoutSrc.orgShape(0) * layoutSrc.orgShape(1) / maxMask) / maxRepeatNum;
        uint32_t repeatRemainder = (layoutSrc.orgShape(0) * layoutSrc.orgShape(1) / maxMask) % maxRepeatNum;
        for (uint32_t i = 0; i < repeatCount; ++i) {
            AscendC::Cast<ElementDst, ElementSrc, true>(
                dstTensor[i * maxRepeatNum * maxMask], srcTensor[i * maxRepeatNum * maxMask],
                AscendC::RoundMode::CAST_NONE, maxMask, maxRepeatNum,
                AscendC::UnaryRepeatParams(
                    1, 1, sizeof(ElementDst) * maxMask / BYTE_PER_BLK, sizeof(ElementSrc) * maxMask / BYTE_PER_BLK
                )
            );
        }
        AscendC::Cast<ElementDst, ElementSrc, true>(
            dstTensor[repeatCount * maxRepeatNum * maxMask], srcTensor[repeatCount * maxRepeatNum * maxMask],
            AscendC::RoundMode::CAST_NONE, maxMask, repeatRemainder,
            AscendC::UnaryRepeatParams(
                1, 1, sizeof(ElementDst) * maxMask / BYTE_PER_BLK, sizeof(ElementSrc) * maxMask / BYTE_PER_BLK
            )
        );
    }
};
} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_CAST_HPP