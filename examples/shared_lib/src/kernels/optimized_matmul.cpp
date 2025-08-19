/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <acl/acl.h>
#include <runtime/rt_ffts.h>

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/optimized_matmul.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "catlass/gemm/device/device_gemm.hpp"

#include "catlass_kernel.h"
#include "common.hpp"

namespace Catlass {
template <
    /// Tag indicating architecture
    class ArchTag,
    /// GemmType for A matrix operand
    class AType,
    /// GemmType type for B matrix operand
    class BType,
    /// GemmType type for C matrix operand
    class CType,
    /// GemmType type for Bias operand
    class BiasType = void>
struct TileCopyOpt : public Catlass::Gemm::Tile::TileCopy<ArchTag, AType, BType, CType, BiasType> {
    using Base = Catlass::Gemm::Tile::TileCopy<ArchTag, AType, BType, CType, BiasType>;
    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    using ElementAccumulator = typename Base::ElementAccumulator;

    // When matrix A is row-major, if the number of rows in matrix A is less than 16,
    // using the CopyGmToL1IntervalDataCopy method can improve the transfer efficiency.
    // The situation is similar for matrix B. If the above conditions are met,
    // please uncomment the following and comment out the original matrix A transfer method

    // using CopyGmToL1A = Gemm::Tile::CopyGmToL1IntervalDataCopy<ArchTag, AType>;

    using CopyGmToL1A = typename Base::CopyGmToL1A;
    using CopyGmToL1B = typename Base::CopyGmToL1B;

    using CopyL1ToL0A = typename Base::CopyL1ToL0A;
    using CopyL1ToL0B = typename Base::CopyL1ToL0B;

    using CopyL0CToGm = typename Base::CopyL0CToGm;
    using BiasTypeSelector = typename Base::BiasTypeSelector;
    using CopyGmToL1Bias = typename Base::CopyGmToL1Bias;
    using CopyL1ToBT = typename Base::CopyL1ToBT;
};
} // namespace Catlass

namespace CatlassKernel {
using namespace Catlass;
template <class LayoutA, class LayoutB, class LayoutC, class InDType, class OutDType>
void OptimizedMatmulImpl(uint32_t blockNum, aclrtStream stream, KernelInfo kernelInfo)
{
    // Prepare FFTS address
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtCheck(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));
    GemmCoord problemShape{kernelInfo.m, kernelInfo.n, kernelInfo.k};
    LayoutA layoutA{kernelInfo.m, kernelInfo.k};
    LayoutB layoutB{kernelInfo.k, kernelInfo.n};
    LayoutC layoutC{kernelInfo.m, kernelInfo.n};
    uint8_t *deviceA = kernelInfo.inputAddr.at(0);
    uint8_t *deviceB = kernelInfo.inputAddr.at(1);
    uint8_t *deviceC = kernelInfo.outputAddr.at(0);
    using ArchTag = Arch::AtlasA2;
    constexpr uint32_t alignByByte = 512;
    constexpr uint32_t alignByElement = alignByByte / sizeof(InDType);
    constexpr bool ENABLE_UNIT_FLAG = true;
    constexpr bool ENABLE_SHUFFLE_K = true;
    using ElementA = InDType;
    using ElementB = InDType;
    using ElementC = OutDType;
    using ElementWorkspace = float;
    using LayoutPaddingA = std::conditional_t<std::is_same_v<LayoutA, layout::RowMajor>, layout::PaddingRowMajor,
                                              layout::PaddingColumnMajor>;
    using LayoutPaddingB = std::conditional_t<std::is_same_v<LayoutB, layout::RowMajor>, layout::PaddingRowMajor,
                                              layout::PaddingColumnMajor>;
    using AType = Gemm::GemmType<ElementA, LayoutA>;
    using BType = Gemm::GemmType<ElementB, LayoutB>;
    using CType = Gemm::GemmType<ElementC, LayoutC>;
    using ATypePadding = Gemm::GemmType<ElementA, LayoutPaddingA>;
    using BTypePadding = Gemm::GemmType<ElementB, LayoutPaddingB>;
    using DispatchPolicy = Gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG, ENABLE_SHUFFLE_K>;

    // if LayoutA and LayoutB is both ColumnMajor,
    // L1TileShape using GemmShape<256, 128, 256> can achieve better performance.
    using L1TileShape =
        std::conditional_t<std::is_same_v<LayoutA, layout::ColumnMajor> && std::is_same_v<LayoutB, layout::ColumnMajor>,
                           GemmShape<256, 128, 256>, GemmShape<128, 256, 256>>;
    using L0TileShape =
        std::conditional_t<std::is_same_v<LayoutA, layout::ColumnMajor> && std::is_same_v<LayoutB, layout::ColumnMajor>,
                           GemmShape<256, 128, 64>, GemmShape<128, 256, 64>>;
    using BlockScheduler30 = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    using BlockScheduler31 = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
    using BlockEpilogue = void;
    using TileCopyDummy = TileCopyOpt<ArchTag, AType, BType, CType>;
    using BlockMmadDummy =
        Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, void, TileCopyDummy>;
    bool isNeedPaddingA = Catlass::Gemm::Kernel::OptimizedMatmul<void, void, BlockMmadDummy, void, void>::IfNeedPadding(
        layoutA, alignByElement);
    bool isNeedPaddingB = Catlass::Gemm::Kernel::OptimizedMatmul<void, void, BlockMmadDummy, void, void>::IfNeedPadding(
        layoutB, alignByElement);
    static const uint32_t COMPUTE_LENGTH_A = 96 * 1024 / sizeof(ElementA);
    using GlobalPaddingA =
        Gemm::Kernel::PaddingMatrixBlockND<ArchTag, ElementA, LayoutA, LayoutPaddingA, COMPUTE_LENGTH_A>;
    static const uint32_t COMPUTE_LENGTH_B = 96 * 1024 / sizeof(ElementB);
    using GlobalPaddingB =
        Gemm::Kernel::PaddingMatrixBlockND<ArchTag, ElementB, LayoutB, LayoutPaddingB, COMPUTE_LENGTH_B>;

    if (!isNeedPaddingA && !isNeedPaddingB) {
        using TileCopy = TileCopyOpt<ArchTag, AType, BType, CType>;
        using BlockMmadOpt =
            Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, void, TileCopy>;
        using MatmulKernel =
            typename Gemm::Kernel::OptimizedMatmul<void, void, BlockMmadOpt, BlockEpilogue, BlockScheduler30>;
        typename MatmulKernel::Arguments arguments{
            problemShape, alignByElement, sizeof(ElementWorkspace), layoutA, layoutB, deviceA, deviceB, deviceC};
        using MatmulAdapter = typename Gemm::Device::DeviceGemm<MatmulKernel>;
        MatmulAdapter matmulOp;
        RunAdapter(matmulOp, arguments, stream, blockNum, fftsAddr);
    } else if (!isNeedPaddingA && isNeedPaddingB) {
        LayoutPaddingB layoutWB = LayoutPaddingB(layoutB.shape(0), layoutB.shape(1), L1TileShape::K, L1TileShape::N);
        using TileCopy = TileCopyOpt<ArchTag, AType, BTypePadding, CType>;
        using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BTypePadding,
                                                    CType, void, TileCopy>;
        using MatmulKernel =
            typename Gemm::Kernel::OptimizedMatmul<void, GlobalPaddingB, BlockMmadOpt, BlockEpilogue, BlockScheduler30>;
        typename MatmulKernel::Arguments arguments{
            problemShape, alignByElement, sizeof(ElementWorkspace), layoutA, layoutWB, deviceA, deviceB, deviceC};
        using MatmulAdapter = typename Gemm::Device::DeviceGemm<MatmulKernel>;
        MatmulAdapter matmulOp;
        RunAdapter(matmulOp, arguments, stream, blockNum, fftsAddr);
    } else if (isNeedPaddingA && !isNeedPaddingB) {
        LayoutPaddingA layoutWA = LayoutPaddingA(layoutA.shape(0), layoutA.shape(1), L1TileShape::M, L1TileShape::K);
        using TileCopy = TileCopyOpt<ArchTag, ATypePadding, BType, CType>;
        using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, ATypePadding, BType,
                                                    CType, void, TileCopy>;
        using MatmulKernel =
            typename Gemm::Kernel::OptimizedMatmul<GlobalPaddingA, void, BlockMmadOpt, BlockEpilogue, BlockScheduler30>;
        typename MatmulKernel::Arguments arguments{
            problemShape, alignByElement, sizeof(ElementWorkspace), layoutWA, layoutB, deviceA, deviceB, deviceC};
        using MatmulAdapter = typename Gemm::Device::DeviceGemm<MatmulKernel>;
        MatmulAdapter matmulOp;
        RunAdapter(matmulOp, arguments, stream, blockNum, fftsAddr);
    } else if (isNeedPaddingA && isNeedPaddingB) {
        LayoutPaddingA layoutWA = LayoutPaddingA(layoutA.shape(0), layoutA.shape(1), L1TileShape::M, L1TileShape::K);
        LayoutPaddingB layoutWB = LayoutPaddingB(layoutB.shape(0), layoutB.shape(1), L1TileShape::K, L1TileShape::N);
        using TileCopy = TileCopyOpt<ArchTag, ATypePadding, BTypePadding, CType>;
        using BlockMmadOpt = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, ATypePadding,
                                                    BTypePadding, CType, void, TileCopy>;
        using MatmulKernel = typename Gemm::Kernel::OptimizedMatmul<GlobalPaddingA, GlobalPaddingB, BlockMmadOpt,
                                                                    BlockEpilogue, BlockScheduler30>;
        typename MatmulKernel::Arguments arguments{
            problemShape, alignByElement, sizeof(ElementWorkspace), layoutWA, layoutWB, deviceA, deviceB, deviceC};
        using MatmulAdapter = typename Gemm::Device::DeviceGemm<MatmulKernel>;
        MatmulAdapter matmulOp;
        RunAdapter(matmulOp, arguments, stream, blockNum, fftsAddr);
    }
}
void OptimizedMatmul(uint32_t blockNum, aclrtStream stream, KernelInfo kernelInfo)
{
    uint32_t m = kernelInfo.m;
    uint32_t n = kernelInfo.n;
    uint32_t k = kernelInfo.k;

    GemmCoord problemShape{kernelInfo.m, kernelInfo.n, kernelInfo.k};

    if (!kernelInfo.transA && !kernelInfo.transB && kernelInfo.inputDataType == ACL_FLOAT16 &&
        kernelInfo.outputDataType == ACL_FLOAT16) {
        OptimizedMatmulImpl<layout::RowMajor, layout::RowMajor, layout::RowMajor, half, half>(blockNum, stream,
                                                                                              kernelInfo);
    }
    // If more conditions are needed, add branches manually.
}
} // namespace CatlassKernel