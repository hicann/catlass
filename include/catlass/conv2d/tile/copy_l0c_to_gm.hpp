/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV2D_TILE_COPY_L0C_TO_GM_HPP
#define CATLASS_CONV2D_TILE_COPY_L0C_TO_GM_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/conv2d/conv2d_type.hpp"

namespace Catlass::Conv2d::Tile {

enum class ScaleGranularity {
    UNDEFINED = -1,
    NO_QUANT = 0,
    PER_TENSOR,
    PER_CHANNEL,
    PER_GROUP
};

template <
    class ArchTag,
    class ElementSrc,
    class ElementDst,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT
>
struct CopyL0CToGmQuantMode {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};

// CopyL0CToGm cast fp32 to fp16
template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    float, half,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::F322F16;
};

template <
    class ArchTag,
    class ElementAccumulator,
    class GmType,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool ReluEnable = false
>
struct CopyL0CToGm {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};

template <
    class ElementAccumulator_,
    class ElementDst_,
    bool ReluEnable_
>
struct CopyL0CToGm<Catlass::Arch::AtlasA2,
                   ElementAccumulator_,
                   Conv2d::Conv2dType<ElementDst_, layout::Output>,
                   ScaleGranularity::NO_QUANT,
                   ReluEnable_>
{
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = Catlass::layout::zN;
    using LayoutDst = Catlass::layout::Output;
    static constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;
    static constexpr uint16_t C0 = BYTE_PER_C0 / sizeof(ElementDst);

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementDst> const &dst,
        AscendC::LocalTensor<ElementSrc> const &src,
        LayoutDst const &dstLayout, uint8_t unitFlag = 0) // (Batch, Cout1, Ho, Wo, C0)
    {
        uint32_t cout1Actual = dstLayout.shape(1);
        uint32_t coutRound = cout1Actual * C0;
        uint32_t hoActual = dstLayout.shape(2);
        uint32_t woActual = dstLayout.shape(3);
        uint32_t howoActual = hoActual * woActual;
        uint32_t howoRound = RoundUp<C0>(howoActual);
        uint32_t strideHo = dstLayout.stride(2); // Wo * C0
        uint32_t strideHoWo = dstLayout.stride(1); // Ho * Wo * C0 
        uint32_t HoWo = strideHoWo / C0;

        for (int hoIdx = 0; hoIdx < hoActual; hoIdx++) {
            size_t gmOffset = hoIdx * strideHo;
            size_t l0Offset = hoIdx * woActual * C0;
            AscendC::FixpipeParamsV220 fixPipeParams(
                coutRound, // nSize
                woActual, // mSize
                howoRound, // srcStride
                HoWo, // dstStride
                reluEn
            );
            fixPipeParams.quantPre = quantPre;
            AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_NZ>(
                dst[gmOffset],
                src[l0Offset],
                fixPipeParams
            );
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace Catlass::Conv2d::Tile

#endif // CATLASS_CONV2D_TILE_COPY_L0C_TO_GM_HPP

