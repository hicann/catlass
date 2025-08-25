/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#ifndef SHARED_LIB_IMPL_CONV_BIAS_H
#define SHARED_LIB_IMPL_CONV_BIAS_H

// for supporting older gcc, to find the reason
#include <iostream>

#include <acl/acl.h>

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/conv/block/block_conv.hpp"
#include "catlass/conv/block/block_swizzle.hpp"
#include "catlass/conv/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/conv/kernel/conv3d_bias.hpp"
#include "catlass/layout/layout.hpp"

namespace Catlass {

template<class LayoutFmap, class LayoutFilter, class LayoutOut, class FmapDtype, class BiasDType, class OutDType>
CATLASS_GLOBAL void conv_bias(Conv3dParams problemShape, GM_ADDR gmFmap, GM_ADDR gmFilter, GM_ADDR gmBias, GM_ADDR gmOut)
{
    constexpr uint32_t l1AStages = 1;
    constexpr uint32_t l1BStages = 1;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;
    constexpr bool enableUnitFlag = true;
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Conv::ConvAtlasA2Pingpong<
        l1AStages, l1BStages,
        l0AStages, l0BStages,
        l0CStages, enableUnitFlag
    >;
    using CoreTileShape = ConvCoreShape<2, 2, 2, 3>;
    using FmapL1TileShape = ConvFmapL1Shape<16, 1, 1>;
    using FilterL1TileShape = ConvFilterL1Shape<1, 1, 16>;
    using L0TileShape = ConvL0Shape<16, 16, 16>;

    using FmapType = Gemm::GemmType<FmapDtype, LayoutFmap>;
    using FilterType = Gemm::GemmType<FmapDtype, LayoutFilter>;
    using BiasType = Gemm::GemmType<BiasDType, layout::VectorLayout>;
    using OutType = Gemm::GemmType<OutDType, LayoutOut>;

    using BlockConv = Conv::Block::BlockConv<DispatchPolicy, CoreTileShape, FmapL1TileShape, FilterL1TileShape, L0TileShape, FmapType, FilterType, OutType, BiasType>;
    using BlockEpilogue = void;

    LayoutFmap layoutFmap= LayoutFmap::MakeLayout(problemShape.batch(), problemShape.di(), problemShape.cin1(), problemShape.hi(), problemShape.wi(), problemShape.cin0());
    LayoutFilter layoutFilter = LayoutFilter::MakeLayout(problemShape.kdc1khkw(), problemShape.n1(), problemShape.n0(), problemShape.cin0());
    LayoutOut layoutOut= LayoutFmap::MakeLayout(problemShape.batch(), problemShape.dout(), problemShape.cout1(), problemShape.ho(), problemShape.wo(), problemShape.cout0());

    // Swizzle offset is 3 and direction is 0.
    using BlockScheduler = typename Conv::Block::Conv3dIdentityBlockSwizzle<3, 0>;

    // kernel level
    using ConvKernel = Conv::Kernel::ConvBias<BlockConv, BlockEpilogue, BlockScheduler>;

    typename ConvKernel::Params params{problemShape, gmFmap, layoutFmap, gmFilter, layoutFilter, gmOut, layoutOut, gmBias};

    // call a kernel
    ConvKernel conv;
    conv(params);
}
} // namespace Catlass
#endif // SHARED_LIB_IMPL_CONV_BIAS_H