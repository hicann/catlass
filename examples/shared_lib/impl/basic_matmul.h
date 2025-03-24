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

#include "AscendCT/AscendCT.hpp"
#include "AscendCT/arch/arch.hpp"
#include "AscendCT/layout/layout.hpp"
#include "AscendCT/gemm/block/block_mmad.hpp"
#include "AscendCT/gemm/block/block_swizzle.hpp"
#include "AscendCT/gemm/dispatch_policy.hpp"
#include "AscendCT/gemm/kernel/basic_matmul.hpp"
#include "AscendCT/gemm/gemm_type.hpp"

using namespace AscendCT;

template <class LayoutA, class LayoutB, class LayoutC, typename IN_TYPE, typename OUT_TYPE>
ASCENDCT_DEVICE void basic_matmul_kernel(GemmCoord problemShape, GM_ADDR gmA, LayoutA layoutA, GM_ADDR gmB,
                                        LayoutB layoutB, GM_ADDR gmC, LayoutC layoutC)
{
    using ArchTag = arch::AtlasA2;
    using DispatchPolicy = gemm::MmadAtlasA2Pingpong<true>;
    using L1TileShape = GemmShape<128, 256, 256>;
    using L0TileShape = GemmShape<128, 256, 64>;

    using AType = gemm::GemmType<IN_TYPE, LayoutA>;
    using BType = gemm::GemmType<IN_TYPE, LayoutB>;
    using CType = gemm::GemmType<OUT_TYPE, LayoutC>;

    using BlockMmad = gemm::block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockEpilogue = void;

    if (problemShape.m() > problemShape.n()) {
        // Swizzle offset is 3 and direction is 0.
        using BlockScheduler = typename gemm::block::GemmIdentityBlockSwizzle<3, 0>;

        // kernel level
        using MatmulKernel = gemm::kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;

        typename MatmulKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC};

        // call a kernel
        MatmulKernel matmul;
        matmul(params);
    } else {
        // Swizzle offset is 3 and direction is 1.
        using BlockScheduler = typename gemm::block::GemmIdentityBlockSwizzle<3, 1>;

        // kernel level
        using MatmulKernel = gemm::kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;

        typename MatmulKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC};

        // call a kernel
        MatmulKernel matmul;
        matmul(params);
    }
}

template <class LayoutA, class LayoutB, class LayoutC, aclDataType IN_TYPE, aclDataType OUT_TYPE>
ASCENDCT_GLOBAL void basic_matmul(GemmCoord problemShape, GM_ADDR gmA, LayoutA layoutA, GM_ADDR gmB, LayoutB layoutB,
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