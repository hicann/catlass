/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDCT_MATMUL_BLOCK_BLOCK_MMAD_HPP
#define ASCENDCT_MATMUL_BLOCK_BLOCK_MMAD_HPP

#include "AscendCT/AscendCT.hpp"
#include "AscendCT/gemm/tile/tile_copy.hpp"
#include "AscendCT/gemm/tile/tile_mmad.hpp"

namespace AscendCT::gemm::block {

template <
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class AType,
    class BType,
    class CType,
    class BiasType = void,
    class TileCopy = gemm::tile::TileCopy<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>,
    class TileMmad = gemm::tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
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
    class TileCopy = gemm::tile::PackedTileCopyTla<typename DispatchPolicy::ArchTag, TensorA, layout::RowMajor,
        TensorB, layout::RowMajor, TensorC, layout::RowMajor, TensorBias, layout::RowMajor>,
    class TileMmad = gemm::tile::TileMmadTla<typename DispatchPolicy::ArchTag, typename TileCopy::TensorL0A,
        typename TileCopy::TensorL0B, typename TileCopy::TensorL0C>
>
struct BlockMmadTla {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadTla is not implemented for this DispatchPolicy");
};

} // namespace AscendCT::gemm::block

#include "AscendCT/gemm/block/block_mmad_pingpong.hpp"
#include "AscendCT/gemm/block/block_mmad_fa_qk.hpp"
#include "AscendCT/gemm/block/block_mmad_fa_pv.hpp"
#include "AscendCT/gemm/block/block_mmad_mla_qk.hpp"
#include "AscendCT/gemm/block/block_mmad_mla_pv.hpp"
#include "AscendCT/gemm/block/block_mmad_preload.hpp"
#include "AscendCT/gemm/block/block_mmad_preload_async.hpp"
#include "AscendCT/gemm/block/block_mmad_pingpong_tla.hpp"
#include "AscendCT/gemm/block/block_mmad_preload_tla.hpp"
#include "AscendCT/gemm/block/block_mmad_preload_async_with_callback.hpp"

#endif // ASCENDCT_MATMUL_BLOCK_BLOCK_MMAD_HPP
