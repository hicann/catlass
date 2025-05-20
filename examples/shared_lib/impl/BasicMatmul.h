/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHARED_LIB_IMPL_BASIC_MATMUL_H
#define SHARED_LIB_IMPL_BASIC_MATMUL_H

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/basic_matmul.hpp"
#include "catlass/gemm/gemm_type.hpp"

using namespace Catlass;

template <class LayoutA, class LayoutB, class LayoutC, typename IN_TYPE, typename OUT_TYPE>
CATLASS_DEVICE void basic_matmul_kernel(GemmCoord problemShape, GM_ADDR gmA, LayoutA layoutA, GM_ADDR gmB,
                                        LayoutB layoutB, GM_ADDR gmC, LayoutC layoutC)
{
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;
    using L1TileShape = GemmShape<128, 256, 256>;
    using L0TileShape = GemmShape<128, 256, 64>;

    using AType = Gemm::GemmType<IN_TYPE, LayoutA>;
    using BType = Gemm::GemmType<IN_TYPE, LayoutB>;
    using CType = Gemm::GemmType<OUT_TYPE, LayoutC>;

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockEpilogue = void;

    if (problemShape.m() > problemShape.n()) {
        // Swizzle offset is 3 and direction is 0.
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

        // kernel level
        using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;

        typename MatmulKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC};

        // call a kernel
        MatmulKernel matmul;
        matmul(params);
    } else {
        // Swizzle offset is 3 and direction is 1.
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

        // kernel level
        using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;

        typename MatmulKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC};

        // call a kernel
        MatmulKernel matmul;
        matmul(params);
    }
}

template <class LayoutA, class LayoutB, class LayoutC, aclDataType IN_TYPE, aclDataType OUT_TYPE>
CATLASS_GLOBAL void basic_matmul(GemmCoord problemShape, GM_ADDR gmA, LayoutA layoutA, GM_ADDR gmB, LayoutB layoutB,
                                 GM_ADDR gmC, LayoutC layoutC)
{
    if constexpr (IN_TYPE == ACL_FLOAT16 && OUT_TYPE == ACL_FLOAT16) {
        basic_matmul_kernel<LayoutA, LayoutB, LayoutC, half, half>(problemShape, gmA, layoutA, gmB, layoutB, gmC,
                                                                   layoutC);
    }

    if constexpr (IN_TYPE == ACL_BF16 && OUT_TYPE == ACL_BF16) {
        basic_matmul_kernel<LayoutA, LayoutB, LayoutC, bfloat16_t, bfloat16_t>(problemShape, gmA, layoutA, gmB, layoutB,
                                                                               gmC, layoutC);
    }
}
#endif // SHARED_LIB_IMPL_BASIC_MATMUL_H