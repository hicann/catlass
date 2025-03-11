/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_MATMUL_BLOCK_BLOCK_MMAD_HPP
#define ACOT_MATMUL_BLOCK_BLOCK_MMAD_HPP

#include "acot/acot.hpp"
#include "acot/matmul/tile/tile_copy.hpp"
#include "acot/matmul/tile/tile_mmad.hpp"

namespace acot::matmul::block {

template <
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class AType,
    class BType,
    class CType,
    class BiasType = void,
    class TileCopy = matmul::tile::TileCopy<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>,
    class TileMmad = matmul::tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockMmad {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class TensorA,
    class TensorB,
    class TensorC,
    class TensorBias = void,
    class TileCopy = matmul::tile::PackedTileCopyV2<typename DispatchPolicy::ArchTag, TensorA, layout::RowMajor,
        TensorB, layout::RowMajor, TensorC, layout::RowMajor, TensorBias, layout::RowMajor>,
    class TileMmad = matmul::tile::TileMmadV2<typename DispatchPolicy::ArchTag, typename TileCopy::TensorL0A,
        typename TileCopy::TensorL0B, typename TileCopy::TensorL0C>
>
struct BlockMmadV2 {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadV2 is not implemented for this DispatchPolicy");
};

} // namespace acot::matmul::block

#include "acot/matmul/block/block_mmad_pingpong.hpp"
#include "acot/matmul/block/block_mmad_fa_qk.hpp"
#include "acot/matmul/block/block_mmad_fa_pv.hpp"
#include "acot/matmul/block/block_mmad_preload.hpp"
#include "acot/matmul/block/block_mmad_preload_async.hpp"
#include "acot/matmul/block/block_mmad_pingpong_v2.hpp"
#include "acot/matmul/block/block_mmad_preload_v2.hpp"

#endif // ACOT_MATMUL_BLOCK_BLOCK_MMAD_HPP
