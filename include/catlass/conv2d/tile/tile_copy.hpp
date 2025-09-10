/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV2D_TILE_TILE_COPY_HPP
#define CATLASS_CONV2D_TILE_TILE_COPY_HPP

#include <type_traits>
#include "catlass/catlass.hpp"
#include "catlass/detail/tag_to_layout.hpp"
#include "catlass/conv2d/tile/copy_gm_to_l1.hpp"
#include "catlass/conv2d/tile/copy_l0c_to_gm.hpp"
#include "catlass/conv2d/tile/copy_l1_to_l0.hpp"
#include "catlass/conv2d/helper.hpp"

namespace Catlass::Conv2d::Tile {

template <
    /// Tag indicating architecture
    class ArchTag,
    /// Conv2dType for Fmap operand
    class FmapType,
    /// Conv2dType type for Filter operand
    class FilterType,
    /// Conv2dType type for Output operand
    class OutputType,
    /// Conv2dType type for Bias operand
    class BiasType = void
>
struct TileCopy {
    using ElementFmap = typename FmapType::Element;
    using ElementFilter = typename FilterType::Element;
    using ElementAccumulator =
        typename Conv2d::helper::ElementAccumulatorSelector<ElementFmap, ElementFilter>::ElementAccumulator;

    using CopyGmToL1A = Conv2d::Tile::CopyGmToL1<ArchTag, FmapType>;
    using CopyGmToL1B = Conv2d::Tile::CopyGmToL1<ArchTag, FilterType>;
    using CopyL1ToL0A = Conv2d::Tile::CopyL1ToL0A<
        ArchTag, typename helper::L1ATypeSelector<FmapType>::L1AType>;
    using CopyL1ToL0B = Conv2d::Tile::CopyL1ToL0B<
        ArchTag, typename helper::L1BTypeSelector<FilterType>::L1BType>;
    using CopyL0CToGm = Conv2d::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, OutputType>;
};

} // namespace Catlass::Conv2d::Tile

#endif // CATLASS_CONV2D_TILE_TILE_COPY_HPP
