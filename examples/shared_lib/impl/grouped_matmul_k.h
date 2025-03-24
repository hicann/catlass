/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHARED_LIB_IMPL_GROUPED_MATMUL_K_H
#define SHARED_LIB_IMPL_GROUPED_MATMUL_K_H

#include "AscendCT/AscendCT.hpp"
#include "AscendCT/arch/arch.hpp"
#include "AscendCT/layout/layout.hpp"
#include "AscendCT/gemm/block/block_mmad.hpp"
#include "AscendCT/gemm/block/block_swizzle.hpp"
#include "AscendCT/gemm/dispatch_policy.hpp"
#include "AscendCT/gemm/kernel/grouped_matmul_k.hpp"
#include "AscendCT/gemm/GemmType"

template <class LayoutA, class LayoutB, class LayoutC>
ASCENDCT_GLOBAL void grouped_matmul_k(GemmCoord problemShape, uint32_t problemCount, GM_ADDR gmGroupList, GM_ADDR gmA,
                                     LayoutA layoutA, GM_ADDR gmB, LayoutB layoutB, GM_ADDR gmC, LayoutC layoutC)
{
    constexpr uint32_t preloadStages = 1;
    constexpr uint32_t l1Stages = 2;
    constexpr uint32_t l0AStages = 4;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;
    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;

    using ArchTag = arch::AtlasA2;
    using DispatchPolicy = gemm::MmadAtlasA2PreloadAsync<preloadStages, l1Stages, l0AStages, l0BStages, l0CStages,
                                                           enableUnitFlag, enableShuffleK>;
    using L1TileShape = GemmShape<128, 256, 256>;
    using L0TileShape = GemmShape<128, 256, 64>;

    using AType = gemm::GemmType<half, LayoutA>;
    using BType = gemm::GemmType<half, LayoutB>;
    using CType = gemm::GemmType<half, LayoutC>;

    using BlockMmad = gemm::block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockEpilogue = void;
    using BlockScheduler = typename gemm::block::GemmIdentityBlockSwizzle<3, 1>;

    // kernel level
    using MatmulKernel = gemm::kernel::GroupedMatmulK<BlockMmad, BlockEpilogue, BlockScheduler, int32_t>;

    typename MatmulKernel::Params params{problemShape, problemCount, gmGroupList, gmA,    layoutA,
                                         gmB,          layoutB,      gmC,         layoutC};

    // call a kernel
    MatmulKernel matmul;
    matmul(params);
}

#endif // SHARED_LIB_IMPL_GROUPED_MATMUL_K_H