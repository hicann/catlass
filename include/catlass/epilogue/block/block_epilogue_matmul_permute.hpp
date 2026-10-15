/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_MATMUL_PERMUTE_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_MATMUL_PERMUTE_HPP

#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/matmul_permute_policy.hpp"
#include "catlass/epilogue/tile/tile_matmul_permute.hpp"

namespace Catlass::Epilogue::Block {

template <
    uint32_t S1_, uint32_t S2_, bool SplitM_, bool UseVectorPack_, uint32_t UB_STAGES_, class ElementOut_,
    class ElementIn_>
class BlockEpilogue<
    EpilogueAscend950MatmulPermute<S1_, S2_, SplitM_, UseVectorPack_, UB_STAGES_>, ElementOut_, ElementIn_> {
public:
    using DispatchPolicy = EpilogueAscend950MatmulPermute<S1_, S2_, SplitM_, UseVectorPack_, UB_STAGES_>;
    using TileEpilogue =
        Tile::TileMatmulPermute<S1_, S2_, SplitM_, UseVectorPack_, UB_STAGES_, ElementOut_, ElementIn_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOut = ElementOut_;
    using ElementIn = ElementIn_;
    static constexpr uint32_t S1 = DispatchPolicy::S1;
    static constexpr uint32_t S2 = DispatchPolicy::S2;
    static constexpr bool SPLIT_M = DispatchPolicy::SPLIT_M;
    static constexpr bool USE_VECTOR_PACK = DispatchPolicy::USE_VECTOR_PACK;
    static constexpr uint32_t UB_STAGES = DispatchPolicy::UB_STAGES;
    static constexpr bool USES_WORKSPACE = true;
    static_assert(UB_STAGES > 0 && UB_STAGES <= 2, "MatmulPermute supports one or two UB stages.");

    CATLASS_DEVICE
    BlockEpilogue() = default;

    template <class TensorGmOut, class TensorUbIn, class TensorScratch>
    CATLASS_DEVICE void operator()(
        TensorGmOut& tensorGmOut, TensorUbIn& tensorUbIn, TensorScratch& tensorScratch, GM_ADDR workspace)
    {
        TileEpilogue tileEpilogue;
        tileEpilogue(tensorGmOut, tensorUbIn, tensorScratch, workspace);
    }
};

} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_MATMUL_PERMUTE_HPP
