/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_ATLAS300I_COPY_L0C_TO_UB_HPP
#define CATLASS_GEMM_TILE_ATLAS300I_COPY_L0C_TO_UB_HPP

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

namespace Catlass::Gemm::Tile {

template <class Arch, class ElementAccumulator, class ElementDst>
struct CopyL0CToUB {};

template <class ElementAccumulator_, class ElementDst_>
struct CopyL0CToUB<Catlass::Arch::Atlas300I, ElementAccumulator_, ElementDst_> {
    using ArchTag = Catlass::Arch::Atlas300I;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = layout::zN;
    using LayoutDst = layout::zN;

    CATLASS_DEVICE
    CopyL0CToUB() = default;

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementDst> const &dst,
        AscendC::LocalTensor<ElementSrc> const &src,
        LayoutDst const &dstLayout,
        LayoutSrc const &srcLayout
    )
    {
        static constexpr size_t DATABLOCKSIZE = 512 * sizeof(ElementSrc) / 2;
        AscendC::DataCopyParams dataCopyParams;
        AscendC::DataCopyEnhancedParams enhanceParams;
        enhanceParams.blockMode = AscendC::BlockMode::BLOCK_MODE_MATRIX;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = dstLayout.orgShape(1) * dstLayout.orgShape(0) * sizeof(ElementSrc) / DATABLOCKSIZE;
        AscendC::DataCopy(dst, src, dataCopyParams, enhanceParams);
    }
};

} // namespace Catlass::Gemm::Tile

#endif // namespace Catlass::Gemm::Tile