/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_ATLAS300I_TILE_DEQUANT_HPP
#define CATLASS_GEMM_TILE_ATLAS300I_TILE_DEQUANT_HPP

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/layout/layout.hpp"
namespace Catlass::Gemm::Tile {
template <class ArchTag, class DstType, class Src1Type, class Src2Type>
struct TileDequant {};

///////////////////////////////////////////////////////
template <class ArchTag>
struct TileDequant<
    ArchTag,
    Gemm::GemmType<float, layout::zN, AscendC::TPosition::VECCALC>,           // dst
    Gemm::GemmType<float, layout::zN, AscendC::TPosition::VECCALC>,           // src1
    Gemm::GemmType<float, layout::VectorLayout, AscendC::TPosition::VECCALC>> // src2
{
    using LayoutDst = layout::zN;
    using LayoutSrc1 = layout::zN;
    using LayoutSrc2 = layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<int8_t>::value;

    CATLASS_DEVICE
    TileDequant()
    {
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<float> const &dstTensor,
        AscendC::LocalTensor<float> const &src1Tensor,
        AscendC::LocalTensor<float> const &src2Tensor,
        LayoutDst const &layoutDst,
        LayoutSrc1 const &layoutSrc1,
        LayoutSrc2 const &LayoutSrc2
    )
    {
        constexpr uint32_t maxRepeatNum = 255;
        uint32_t count = layoutSrc1.orgShape(1) / C0_NUM_PER_FRACTAL;
        uint32_t mRound = layoutSrc1.orgShape(0);
        uint32_t mRepeatCount = mRound / maxRepeatNum;
        uint32_t mRepeatRemainder = mRound % maxRepeatNum;
        for (uint32_t i = 0; i < count; ++i) {
            if (mRepeatCount > 0) {
                AscendC::Mul<float>(
                    dstTensor[i * mRound * C0_NUM_PER_FRACTAL], src1Tensor[i * mRound * C0_NUM_PER_FRACTAL],
                    src2Tensor[i * C0_NUM_PER_FRACTAL], (uint64_t)C0_NUM_PER_FRACTAL, maxRepeatNum,
                    AscendC::BinaryRepeatParams(1, 1, 1, 2, 2, 0)
                );
            }
            AscendC::Mul<float>(
                dstTensor[i * mRound * C0_NUM_PER_FRACTAL + mRepeatCount * maxRepeatNum * C0_NUM_PER_FRACTAL],
                src1Tensor[i * mRound * C0_NUM_PER_FRACTAL + mRepeatCount * maxRepeatNum * C0_NUM_PER_FRACTAL],
                src2Tensor[i * C0_NUM_PER_FRACTAL], (uint64_t)C0_NUM_PER_FRACTAL, mRepeatRemainder,
                AscendC::BinaryRepeatParams(1, 1, 1, 2, 2, 0)
            );
        }
    }
};
///////////////////////////////////////////////////////
} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_TILE_MULS_HPP