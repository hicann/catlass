/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_GEMV_TILE_COPY_L0C_TO_GM_HPP
#define ACOT_GEMV_TILE_COPY_L0C_TO_GM_HPP

#include "acot/matmul/matmul_type.hpp"

namespace acot::gemv::tile
{

    enum class ScaleGranularity
    {
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
        ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT>
    struct CopyL0CToGmQuantMode
    {
        static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
    };

    // CopyL0CToGm cast fp32 to fp16
    template <>
    struct CopyL0CToGmQuantMode<
        acot::arch::AtlasA2,
        float, half,
        ScaleGranularity::NO_QUANT>
    {
        static constexpr auto VALUE = QuantMode_t::F322F16;
    };

    // CopyL0CToGm cast fp32 to bf16
    template <>
    struct CopyL0CToGmQuantMode<
        acot::arch::AtlasA2,
        float, bfloat16_t,
        ScaleGranularity::NO_QUANT>
    {
        static constexpr auto VALUE = QuantMode_t::F322BF16;
    };

    // CopyL0CToGm output fp32
    template <>
    struct CopyL0CToGmQuantMode<
        acot::arch::AtlasA2,
        float, float,
        ScaleGranularity::NO_QUANT>
    {
        static constexpr auto VALUE = QuantMode_t::NoQuant;
    };

    // CopyL0CToGm output int32
    template <>
    struct CopyL0CToGmQuantMode<
        acot::arch::AtlasA2,
        int32_t, int32_t,
        ScaleGranularity::NO_QUANT>
    {
        static constexpr auto VALUE = QuantMode_t::NoQuant;
    };

    // CopyL0CToGm cast int32_t to fp16
    template <>
    struct CopyL0CToGmQuantMode<
        acot::arch::AtlasA2,
        int32_t, half,
        ScaleGranularity::PER_TENSOR>
    {
        static constexpr auto VALUE = QuantMode_t::DEQF16;
    };

    template <>
    struct CopyL0CToGmQuantMode<
        acot::arch::AtlasA2,
        int32_t, half,
        ScaleGranularity::PER_CHANNEL>
    {
        static constexpr auto VALUE = QuantMode_t::VDEQF16;
    };

    template <
        class ArchTag,
        class ElementAccumulator,
        class GmType,
        ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
        bool ReluEnable = false>
    struct CopyL0CToGm
    {
        static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
    };

    template <
        class ElementAccumulator_,
        class ElementDst_,
        bool ReluEnable_>
    struct CopyL0CToGm<acot::arch::AtlasA2,
                       ElementAccumulator_,
                       matmul::MatmulType<ElementDst_, layout::RowMajor>,
                       ScaleGranularity::NO_QUANT,
                       ReluEnable_>
    {
        using ArchTag = acot::arch::AtlasA2;
        using ElementDst = ElementDst_;
        using ElementSrc = ElementAccumulator_;
        using LayoutSrc = acot::layout::zN;
        using LayoutDst = acot::layout::RowMajor;
        static constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst,
                                                              ScaleGranularity::NO_QUANT>::VALUE;
        static constexpr auto reluEn = ReluEnable_;

        static constexpr uint32_t ELE_NUM_PER_C0 =  BYTE_PER_C0 / sizeof(ElementDst);

        ACOT_DEVICE
        void operator()(AscendC::GlobalTensor<ElementDst> const &dst, AscendC::LocalTensor<ElementSrc> const &src,
                        LayoutDst const &dstLayout, LayoutSrc const &srcLayout, uint8_t unitFlag = 0)
        {
            AscendC::FixpipeParamsV220 intriParams;

            // Fixpipe layout information
            intriParams.nSize = dstLayout.shape(1);     //N方向的size大小
            intriParams.mSize = dstLayout.shape(0);     //M方向的size大小
            intriParams.srcStride = srcLayout.stride(3) / srcLayout.stride(0); // 相邻连续数据片段间隔
            intriParams.dstStride = dstLayout.stride(0);

            // Fixpipe auxiliary arguments
            intriParams.quantPre = quantPre;
            intriParams.reluEn = reluEn;
            intriParams.unitFlag = unitFlag;

            // Call AscendC Fixpipe
            AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(dst, src, intriParams);

            // uint32_t MAlignment = srcLayout.shape(0);
            // uint32_t NAlignment = ELE_NUM_PER_C0;
            // if constexpr (std::is_same<ElementSrc, float>::value && std::is_same<ElementDst, float>::value){
            //     // 说明原来的一定是float数据类型
            //     NAlignment = srcLayout.shape(0);
            // }
            // uint32_t MActual = dstLayout.shape(0);
            // uint32_t NActual = dstLayout.shape(1);
            // uint32_t MRound = RoundUp(MActual, MAlignment);
            // uint32_t NRound = RoundUp(NActual, NAlignment);
            // uint32_t strideC = dstLayout.stride(0);
            // AscendC::DataCopyCO12DstParams params;
            // // 一定是Nz2Nd的，而且L0C一定是行优先的
            // params.nSize = NActual; // 参数写反了  解决了问题
            // params.mSize = MActual;
            // params.dstStride = strideC;
            // params.srcStride = MRound;
            // params.quantPre = quantPre;
            // params.reluPre = 0;
            // params.channelSplit = false;
            // params.unitFlag = unitFlag;
            // params.nz2ndEn = true;
            // AscendC::DataCopy(dst, src, params);



        }
    };

    template <
        class ElementAccumulator_,
        class ElementDst_,
        bool ReluEnable_>
    struct CopyL0CToGm<acot::arch::AtlasA2,
                       ElementAccumulator_,
                       matmul::MatmulType<ElementDst_, layout::zN>,
                       ScaleGranularity::NO_QUANT,
                       ReluEnable_>
    {
        using ArchTag = acot::arch::AtlasA2;
        using ElementDst = ElementDst_;
        using ElementSrc = ElementAccumulator_;
        using LayoutSrc = acot::layout::zN;
        using LayoutDst = acot::layout::zN;
        static constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst,
                                                              ScaleGranularity::NO_QUANT>::VALUE;
        static constexpr auto reluEn = ReluEnable_;

        ACOT_DEVICE
        void operator()(AscendC::GlobalTensor<ElementDst> const &dst, AscendC::LocalTensor<ElementSrc> const &src,
                        LayoutDst const &dstLayout, LayoutSrc const &srcLayout, uint8_t unitFlag = 0)
        {
            AscendC::FixpipeParamsV220 intriParams;

            // Fixpipe layout information
            intriParams.nSize = dstLayout.shape(2) * dstLayout.shape(3);
            intriParams.mSize = dstLayout.shape(0) * dstLayout.shape(1);
            intriParams.srcStride = srcLayout.stride(3) / srcLayout.shape(2);
            intriParams.dstStride = dstLayout.stride(3) / (BYTE_PER_C0 / sizeof(ElementDst));

            // Fixpipe auxiliary arguments
            intriParams.quantPre = quantPre;
            intriParams.reluEn = reluEn;
            intriParams.unitFlag = unitFlag;

            // Call AscendC Fixpipe
            AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_NZ>(dst, src, intriParams);
        }
    };

} // namespace acot::gemv::tile

#endif // ACOT_GEMV_TILE_COPY_L0C_TO_GM_HPP
