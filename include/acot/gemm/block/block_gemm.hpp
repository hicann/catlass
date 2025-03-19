/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_GEMM_BLOCK_BLOCK_GEMM_HPP
#define ACOT_GEMM_BLOCK_BLOCK_GEMM_HPP

#include "acot/acot.hpp"
#include "acot/gemm/helper.hpp"
#include "acot/gemm/tile/tile_copy.hpp"
#include "acot/gemm/tile/tile_mmad.hpp"
#include "acot/matmul_coord.hpp"

namespace acot::gemm::block{
template<
    class DispatchPolicy_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_ = void, 
    class TileCopy_ = acot::gemm::tile::TileCopy<typename DispatchPolicy_::ArchTag, AType_, BType_, CType_, BiasType_>,
    class TileMmad_ = acot::gemm::tile::TileMmad<typename DispatchPolicy_::ArchTag, AType_, BType_, CType_, BiasType_>
>
struct BlockGemm{
    static_assert(DEPENDENT_FALSE<DispatchPolicy_>, "BlockMmad is not implemented for this DispatchPolicy");
};

}

#include "acot/gemm/block/block_gemm_preload.hpp"

#endif // ACOT_GEMM_BLOCK_BLOCK_GEMM_HPP