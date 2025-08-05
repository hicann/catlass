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
#include "catlass/conv/block/block_conv3d_pingpong_bias.hpp"
#include "catlass/conv/block/block_swizzle.hpp"
#include "catlass/conv/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/conv/kernel/conv3d_bias.hpp"
#include "catlass/layout/layout.hpp"

namespace Catlass {

template<class LayoutFmap, class LayoutFilter, class LayoutOut, class LayoutBias, class FmapDtype, class BiasDType, class OutDType>
CATLASS_GLOBAL void conv_bias(Conv3dParams<FmapDtype> problemShape, GM_ADDR gmFmap, GM_ADDR gmFilter, GM_ADDR gmBias, GM_ADDR gmOut)
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
        enableUnitFlag
    >;
    using CoreTileShape = ConvCoreShape<singleCoreNo, singleCoreDo, singleCoreCo1, singleCoreHoWo>;
    using FmapL1TileShape = ConvFmapL1Shape<mAL1, Kd, Ci1>;
    using FilterL1TileShape = ConvFilterL1Shape<Kd, Ci1, nBL1>;
    using L0TileShape = ConvL0Shape<mL0, kL0, nL0>;

    using FmapType = Conv::ConvType<FmapDtype, LayoutFmap>;
    using FilterType = Conv::ConvType<FmapDtype, LayoutFilter>;
    using BiasType = Conv::ConvType<BiasDType, LayoutBias>;
    using OutType = Conv::ConvType<FmapDtype, LayoutOut>;

    using BlockConv = Conv::Block::BlockConv<DispatchPolicy, CoreTileShape, FmapL1TileShape, FilterL1TileShape, L0TileShape, FmapType, FilterType, OutType, BiasType>;
    using BlockEpilogue = void;

    LayoutFmap layoutFmap{problemShape.n(), problemShape.di(), problemShape.cin1(), problemShape.hi(), problemShape.wi(), problemShape.cin0()};
    LayoutFilter layoutFilter{problemShape.kdc1khkw(), problemShape.n1(), problemShape.n0(), problemShape.cin0()};
    LayoutOut layoutOut{problemShape.n(), problemShape.do(), problemShape.cout1(), problemShape.ho(), problemShape.wo(), problemShape.cout0()};
    LayoutBias layoutBias{problemShape.cout()};

    // Swizzle offset is 3 and direction is 0.
    using BlockScheduler = typename conv::Block::Conv3dIdentityBlockSwizzle<3, 0>;

    // kernel level
    using ConvKernel = Conv::Kernel::BasicConv<BlockConv, BlockEpilogue, BlockScheduler>;

    typename ConvKernel::Params params{problemShape, gmFmap, LayoutFmap, gmFilter, LayoutFilter, gmOut, layoutOut, gmBias, layoutBias};

    // call a kernel
    ConvKernel conv;
    conv(params);
}
} // namespace Catlass
#endif // SHARED_LIB_IMPL_CONV_BIAS_H