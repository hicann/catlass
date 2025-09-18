/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV2D_BLOCK_BLOCK_MMAD_HPP
#define CATLASS_CONV2D_BLOCK_BLOCK_MMAD_HPP

#include "catlass/catlass.hpp"
#include "catlass/conv2d/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"

namespace Catlass::Conv2d::Block {

template <
    class DispatchPolicy,
    class FmapL1TileShape,
    class FilterL1TileShape,
    class L0TileShape,
    class FmapType,
    class FilterType,
    class OutputType,
    class BiasType = void,
    class TileCopy = Conv2d::Tile::TileCopy<typename DispatchPolicy::ArchTag, FmapType, FilterType, OutputType, BiasType>,
    class TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, FmapType, FilterType, BiasType>
>
struct BlockMmad {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

/// new add for the reason that i am using the dispatchpolicy which is same as the policy of the optimized_matmul
// so i add a new one class to avoid the conflict
template <
    class DispatchPolicy,
    class FmapL1TileShape,
    class FilterL1TileShape,
    class L0TileShape,
    class FmapType,
    class FilterType,
    class OutputType,
    class BiasType = void,
    class TileCopy = Conv2d::Tile::TileCopy<typename DispatchPolicy::ArchTag, FmapType, FilterType, OutputType, BiasType>,  // change the name
    class TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, FmapType, FilterType, BiasType>
>
struct BlockConv2d {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

} // namespace Catlass::Conv2d::Block

#include "catlass/conv2d/block/block_mmad_pingpong.hpp"

#endif // CATLASS_CONV2D_BLOCK_BLOCK_MMAD_HPP
