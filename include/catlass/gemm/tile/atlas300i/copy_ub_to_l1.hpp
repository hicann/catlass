/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_ATLAS300I_COPY_UB_TO_L1_HPP
#define CATLASS_GEMM_TILE_ATLAS300I_COPY_UB_TO_L1_HPP

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

namespace Catlass::Gemm::Tile {

template <class ArchTag, class UbType, class L1Type = void>
struct CopyUBToL1 {};
///////////////////////////
// for 310PW8A8
///////////////////////////
template <class ArchTag>
struct CopyUBToL1<
    ArchTag,
    Gemm::GemmType<int8_t, layout::zZ, AscendC::TPosition::VECCALC>,
    Gemm::GemmType<int8_t, layout::zZ, AscendC::TPosition::A1>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::zZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<int8_t>::value;
    // Mehtods
    CATLASS_DEVICE
    CopyUBToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<int8_t> const &dstTensor,
        AscendC::LocalTensor<int8_t> const &srcTensor,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc
    )
    {
        AscendC::DataCopy(dstTensor, srcTensor, layoutSrc.orgShape(0) * layoutSrc.orgShape(1));
    }
};

template <class ArchTag>
struct CopyUBToL1<
    ArchTag,
    Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::VECCALC>,
    Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::B1>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<int8_t>::value;
    // Mehtods
    CATLASS_DEVICE
    CopyUBToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<int8_t> const &dstTensor,
        AscendC::LocalTensor<int8_t> const &srcTensor,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc
    )
    {
        AscendC::DataCopy(dstTensor, srcTensor, layoutSrc.orgShape(0) * layoutSrc.orgShape(1));
    }
};

} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_COPY_UB_TO_L1_HPP
