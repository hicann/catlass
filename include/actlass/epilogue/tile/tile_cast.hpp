/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDCT_EPILOGUE_TILE_TILE_CAST_HPP
#define ASCENDCT_EPILOGUE_TILE_TILE_CAST_HPP

#include "AscendCT/AscendCT.hpp"

namespace AscendCT::epilogue::tile {

template <
    /// Tag indicating architecture
    class ArchTag_,
    /// Compute data type
    class DstType_,
    class SrcType_,
    /// Length of the compute buffer
    class TileShape_
>
struct TileCast {
    using ArchTag = ArchTag_;
    using ElementDst = typename DstType_::Element;
    using ElementSrc = typename SrcType_::Element;
    using TileShape = TileShape_;

    ASCENDCT_DEVICE
    TileCast() {}

    ASCENDCT_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementDst> const &ubOut,
        AscendC::LocalTensor<ElementSrc> const &ubIn
    )
    {
        AscendC::Cast(ubOut, ubIn, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
    }
};

} // namespace AscendCT::epilogue::tile

#endif
