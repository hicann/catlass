/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include "catlass/gemm/kernel/basic_matmul.hpp"

#include <acl/acl.h>

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

#include "catlass_kernel.h"
// #include "common.hpp"
#include "common/kernel_chooser.hpp"
#include "common/visit_dispatcher.hpp"

namespace CatlassKernel {
using namespace Catlass;
using namespace detail;

// 模板元函数：根据输入数据类型选择 TileShape
// 使用通用的 Fp32TileShapeAdapter 统一处理 fp32 类型的 K 维度减半
template <typename InDType>
struct TileShapeSelector {
    using L1TileShape = typename Fp32TileShapeAdapter<InDType, GemmShape<128, 256, 256>>::type;
    using L0TileShape = typename Fp32TileShapeAdapter<InDType, GemmShape<128, 256, 64>>::type;
};

template <typename LayoutA, typename LayoutB, typename LayoutC, typename InDType, typename OutDType>
CATLASS_GLOBAL __attribute__((aic)) void BasicMatmulKernel(
    GemmCoord problemShape, GM_ADDR ptrA, LayoutA layoutA, GM_ADDR ptrB, LayoutB layoutB, GM_ADDR ptrC, LayoutC layoutC)
{
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;

    // 根据输入类型自动选择 TileShape
    using L1TileShape = typename TileShapeSelector<InDType>::L1TileShape;
    using L0TileShape = typename TileShapeSelector<InDType>::L0TileShape;

    using AType = Gemm::GemmType<InDType, LayoutA>;
    using BType = Gemm::GemmType<InDType, LayoutB>;
    using CType = Gemm::GemmType<OutDType, LayoutC>;

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockEpilogue = void;

    // Swizzle offset is 3 and direction is 0.
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

    // kernel level
    using MatmulKernel = typename Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;
    typename MatmulKernel::Params params{problemShape, ptrA, layoutA, ptrB, layoutB, ptrC, layoutC};

    MatmulKernel matmul;
    matmul(params);
}

template <typename LayoutA, typename LayoutB, typename LayoutC, typename InDType, typename OutDType>
void BasicMatmulImpl(const uint32_t blockNum, aclrtStream stream, const KernelInfo& kernelInfo)
{
    GemmCoord problemShape{kernelInfo.m, kernelInfo.n, kernelInfo.k};
    uint8_t* deviceA = kernelInfo.inputAddr.at(0);
    uint8_t* deviceB = kernelInfo.inputAddr.at(1);
    uint8_t* deviceC = kernelInfo.outputAddr.at(0);
    LayoutA layoutA = LayoutA::template MakeLayout<InDType>(problemShape.m(), problemShape.k());
    LayoutB layoutB = LayoutB::template MakeLayout<InDType>(problemShape.k(), problemShape.n());
    LayoutC layoutC = LayoutC::template MakeLayout<OutDType>(problemShape.m(), problemShape.n());

    BasicMatmulKernel<LayoutA, LayoutB, LayoutC, InDType, OutDType>
        <<<blockNum, nullptr, stream>>>(problemShape, deviceA, layoutA, deviceB, layoutB, deviceC, layoutC);
    aclrtSynchronizeStream(stream);
}

// 辅助函数：根据数据类型、转置标志和格式标志选择调用对应的实现
// 使用 KernelChooser 自动递归展开 transA、transB、formatA 和 formatB 布尔参数
template <typename InDType, typename OutDType, bool TransA, bool TransB, bool FormatA, bool FormatB>
struct BasicMatmulKernelCaller {
    static void call(const uint32_t blockNum, aclrtStream stream, const KernelInfo& kernelInfo)
    {
        // 根据 formatA 和 formatB 选择布局
        // 如果 formatA 为 true，使用 zN/nZ 格式；否则使用常规 Row/Col 格式
        // 如果 formatA 为 true 且 transA 为 true，使用 nZ；否则使用 zN
        using LayoutA = typename FormatLayoutSelector<FormatA, TransA>::type;
        // 如果 formatB 为 true，使用 zN/nZ 格式；否则使用常规 Row/Col 格式
        // 如果 formatB 为 true 且 transB 为 true，使用 nZ；否则使用 zN
        using LayoutB = typename FormatLayoutSelector<FormatB, TransB>::type;
        using LayoutC = layout::RowMajor;

        BasicMatmulImpl<LayoutA, LayoutB, LayoutC, InDType, OutDType>(blockNum, stream, kernelInfo);
    }
};

// KernelChooser 特化：终止条件，当所有布尔参数都已展开时调用最终的实现
// 必须在 detail 命名空间中定义，因为 KernelChooser 在该命名空间中
namespace detail {
template <typename InDType, typename OutDType, bool TransA, bool TransB, bool FormatA, bool FormatB>
struct KernelChooser<std::tuple<InDType, OutDType>, BoolVariable<TransA, TransB, FormatA, FormatB>> {
    static void call(const uint32_t blockNum, aclrtStream stream, const KernelInfo& kernelInfo)
    {
        BasicMatmulKernelCaller<InDType, OutDType, TransA, TransB, FormatA, FormatB>::call(
            blockNum, stream, kernelInfo);
    }
};
} // namespace detail

// BasicMatmul 支持的类型组合：
//   INT8 -> INT32
//   FLOAT16/BF16/FLOAT -> FLOAT16/BF16/FLOAT（所有浮点类型之间的组合）
namespace detail {
DEFINE_TYPE_COMBINATION_SUPPORTED(Int8Tag, Int32Tag);
DEFINE_TYPE_COMBINATION_SUPPORTED(HalfTag, HalfTag);
DEFINE_TYPE_COMBINATION_SUPPORTED(HalfTag, BFloat16Tag);
DEFINE_TYPE_COMBINATION_SUPPORTED(HalfTag, FloatTag);
DEFINE_TYPE_COMBINATION_SUPPORTED(BFloat16Tag, HalfTag);
DEFINE_TYPE_COMBINATION_SUPPORTED(BFloat16Tag, BFloat16Tag);
DEFINE_TYPE_COMBINATION_SUPPORTED(BFloat16Tag, FloatTag);
DEFINE_TYPE_COMBINATION_SUPPORTED(FloatTag, HalfTag);
DEFINE_TYPE_COMBINATION_SUPPORTED(FloatTag, BFloat16Tag);
DEFINE_TYPE_COMBINATION_SUPPORTED(FloatTag, FloatTag);
} // namespace detail

void BasicMatmul(const uint32_t blockNum, aclrtStream stream, const KernelInfo& kernelInfo)
{
    // 使用 visit 模式进行类型分发
    visitDataTypePair(kernelInfo.inputDataType, kernelInfo.outputDataType, [&](auto inTag, auto outTag) {
        using InType = typename TagToType<decltype(inTag)>::type;
        using OutType = typename TagToType<decltype(outTag)>::type;

        // 使用 KernelChooser 自动递归展开 transA、transB、formatA 和 formatB 布尔参数
        detail::KernelChooser<std::tuple<InType, OutType>, detail::BoolVariable<>>::call(
            kernelInfo.transA, kernelInfo.transB, kernelInfo.formatA, kernelInfo.formatB, blockNum, stream, kernelInfo);
    });
}
} // namespace CatlassKernel