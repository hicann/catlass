/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHARED_LIB_IMPL_OPTIMIZED_MATMUL_H
#define SHARED_LIB_IMPL_OPTIMIZED_MATMUL_H

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/optimized_matmul.hpp"
#include "catlass/gemm/gemm_type.hpp"

using namespace Catlass;

template <class Layout> size_t GetWorkspaceLen(Layout layout, size_t blockRows, size_t blockCols)
{
    return RoundUp(static_cast<size_t>(layout.shape(0)), blockRows) *
           RoundUp(static_cast<size_t>(layout.shape(1)), blockCols);
}

bool IsNeedPadding(layout::RowMajor layout, uint32_t align)
{
    // If the stride is greater than 65536, padding is required to reduce the stride.
    if (layout.stride(0) < 65536) {
        return layout.stride(0) % align != 0;
    } else {
        return true;
    }
}

bool IsNeedPadding(layout::ColumnMajor layout, uint32_t align)
{
    // If the stride is greater than 65536, padding is required to reduce the stride.
    if (layout.stride(1) < 65536) {
        return layout.stride(1) % align != 0;
    } else {
        return true;
    }
}

template <class LayoutA, class LayoutB, class LayoutC, class LayoutWA, class LayoutWB, class BlockMmad>
CATLASS_DEVICE void LaunchMatmulDynamicSwizzle(GemmCoord problemShape, GM_ADDR gmA, LayoutA layoutA, GM_ADDR gmB,
                                               LayoutB layoutB, GM_ADDR gmC, LayoutC layoutC, GM_ADDR gmWA,
                                               LayoutWA layoutWA, GM_ADDR gmWB, LayoutWB layoutWB)
{
    if (problemShape.m() > problemShape.n()) {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        using BlockEpilogue = void;
        // kernel level
        using MatmulKernel = Gemm::Kernel::OptimizedMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;
        typename MatmulKernel::Params params{problemShape, gmA,  layoutA,  gmB,  layoutB, gmC,
                                             layoutC,      gmWA, layoutWA, gmWB, layoutWB};
        // call a kernel
        MatmulKernel matmul;
        matmul(params);
    } else {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
        using BlockEpilogue = void;
        // kernel level
        using MatmulKernel = Gemm::Kernel::OptimizedMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;
        typename MatmulKernel::Params params{problemShape, gmA,  layoutA,  gmB,  layoutB, gmC,
                                             layoutC,      gmWA, layoutWA, gmWB, layoutWB};

        // call a kernel
        MatmulKernel matmul;
        matmul(params);
    }
}

template <class LayoutA, class LayoutB, class LayoutC>
CATLASS_GLOBAL void optimized_matmul(uint64_t fftsAddr, GemmCoord problemShape, GM_ADDR gmA, LayoutA layoutA,
                                     GM_ADDR gmB, LayoutB layoutB, GM_ADDR gmC, LayoutC layoutC, GM_ADDR gmWA,
                                     GM_ADDR gmWB)
{
    using ArchTag = Arch::AtlasA2;
    AscendC::SetSyncBaseAddr(fftsAddr);

    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    using DispatchPolicy = Gemm::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;

    // if LayoutA and LayoutB is both ColumnMajor,
    // L1TileShape using GemmShape<256, 128, 256> can achieve better performance.
    using L1TileShape =
        std::conditional_t<std::is_same_v<LayoutA, layout::ColumnMajor> && std::is_same_v<LayoutB, layout::ColumnMajor>,
                           GemmShape<256, 128, 256>, GemmShape<128, 256, 256>>;
    using L0TileShape =
        std::conditional_t<std::is_same_v<LayoutA, layout::ColumnMajor> && std::is_same_v<LayoutB, layout::ColumnMajor>,
                           GemmShape<256, 128, 64>, GemmShape<128, 256, 64>>;
    ;
    if (gmA == gmWA && gmB == gmWB) {
        // no need to padding A and B.
        using LayoutWA = LayoutA;
        using LayoutWB = LayoutB;
        using AType = Gemm::GemmType<half, LayoutWA>;
        using BType = Gemm::GemmType<half, LayoutWB>;
        using CType = Gemm::GemmType<half, LayoutC>;
        using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        LayoutWA layoutWA = LayoutWA(layoutA.shape(0), layoutA.shape(1));
        LayoutWB layoutWB = LayoutWB(layoutB.shape(0), layoutB.shape(1));
        LaunchMatmulDynamicSwizzle<LayoutA, LayoutB, LayoutC, LayoutWA, LayoutWB, BlockMmad>(
            problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, gmWA, layoutWA, gmWB, layoutWB);
    } else if (gmA == gmWA && gmB != gmWB) {
        // no need to padding A, but B needs padding.
        using LayoutWA = LayoutA;
        using LayoutWB = std::conditional_t<std::is_same_v<LayoutB, layout::RowMajor>, layout::PaddingRowMajor,
                                            layout::PaddingColumnMajor>;
        using AType = Gemm::GemmType<half, LayoutWA>;
        using BType = Gemm::GemmType<half, LayoutWB>;
        using CType = Gemm::GemmType<half, LayoutC>;
        using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        LayoutWA layoutWA = LayoutWA(layoutA.shape(0), layoutA.shape(1));
        LayoutWB layoutWB = LayoutWB(layoutB.shape(0), layoutB.shape(1), L1TileShape::K, L1TileShape::N);
        LaunchMatmulDynamicSwizzle<LayoutA, LayoutB, LayoutC, LayoutWA, LayoutWB, BlockMmad>(
            problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, gmWA, layoutWA, gmWB, layoutWB);
    } else if (gmA != gmWA && gmB == gmWB) {
        // no need to padding B, but A needs padding.
        using LayoutWA = std::conditional_t<std::is_same_v<LayoutA, layout::RowMajor>, layout::PaddingRowMajor,
                                            layout::PaddingColumnMajor>;
        using LayoutWB = LayoutB;
        using AType = Gemm::GemmType<half, LayoutWA>;
        using BType = Gemm::GemmType<half, LayoutWB>;
        using CType = Gemm::GemmType<half, LayoutC>;
        using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        LayoutWA layoutWA = LayoutWA(layoutA.shape(0), layoutA.shape(1), L1TileShape::M, L1TileShape::K);
        LayoutWB layoutWB = LayoutWB(layoutB.shape(0), layoutB.shape(1));
        LaunchMatmulDynamicSwizzle<LayoutA, LayoutB, LayoutC, LayoutWA, LayoutWB, BlockMmad>(
            problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, gmWA, layoutWA, gmWB, layoutWB);
    } else {
        // Both A and B need padding.
        using LayoutWA = std::conditional_t<std::is_same_v<LayoutA, layout::RowMajor>, layout::PaddingRowMajor,
                                            layout::PaddingColumnMajor>;
        using LayoutWB = std::conditional_t<std::is_same_v<LayoutB, layout::RowMajor>, layout::PaddingRowMajor,
                                            layout::PaddingColumnMajor>;
        using AType = Gemm::GemmType<half, LayoutWA>;
        using BType = Gemm::GemmType<half, LayoutWB>;
        using CType = Gemm::GemmType<half, LayoutC>;
        using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        LayoutWA layoutWA = LayoutWA(layoutA.shape(0), layoutA.shape(1), L1TileShape::M, L1TileShape::K);
        LayoutWB layoutWB = LayoutWB(layoutB.shape(0), layoutB.shape(1), L1TileShape::K, L1TileShape::N);
        LaunchMatmulDynamicSwizzle<LayoutA, LayoutB, LayoutC, LayoutWA, LayoutWB, BlockMmad>(
            problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, gmWA, layoutWA, gmWB, layoutWB);
    }
}

#endif // SHARED_LIB_IMPL_OPTIMIZED_MATMUL_H