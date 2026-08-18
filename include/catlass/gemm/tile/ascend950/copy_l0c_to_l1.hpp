/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_ASCEND950_COPY_L0C_TO_L1_HPP
#define CATLASS_GEMM_TILE_ASCEND950_COPY_L0C_TO_L1_HPP

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/tile/ascend950/copy_l0c_to_dst.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Tile {

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToL1Tla<
    Catlass::Arch::Ascend950, TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::A1>,
    ScaleGranularity::NO_QUANT, ReluEnable_, std::enable_if_t<tla::detail::iszN<ElementDst_, LayoutDst_>::value>> {
    using ArchTag = Catlass::Arch::Ascend950;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementDst); // L1

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dstTensor, TensorSrc const& srcTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value &&
                TensorSrc::position == AscendC::TPosition::CO1 && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be L1 and zN");

        AscendC::DataCopyCO12DstParams intriParams;

        intriParams.nSize = RoundUp<ELE_NUM_PER_C0>(tla::get<1>(dstTensor.originShape()));
        intriParams.mSize = tla::get<0>(dstTensor.originShape());
        intriParams.dstStride = tla::get<1, 1>(dstTensor.stride()) / ELE_NUM_PER_C0;
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.quantPre = quantPre;
        intriParams.nz2ndEn = false;
        intriParams.reluPre = reluEn;
        intriParams.unitFlag = unitFlag;

        if constexpr (std::is_same_v<ElementSrc, float> && std::is_same_v<ElementDst, float>) {
            intriParams.channelSplit = true;
        }

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_ASCEND950_COPY_L0C_TO_L1_HPP
