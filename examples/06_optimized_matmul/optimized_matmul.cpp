/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0, 
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <iostream>
#include <vector>

#include "helper.hpp"
#include "golden.hpp"
#include "fp16_t.h"

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/optimized_matmul.hpp"
#include "catlass/gemm/gemm_type.hpp"

#include "catlass/status.hpp"
#include "catlass/gemm/device/device_gemm.hpp"

using namespace Catlass;
using fp16_t = op::fp16_t;

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
    class BiasType = void
>
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

constexpr uint32_t alignByByte = 512;
constexpr uint32_t alignByElement = alignByByte / sizeof(fp16_t);

using ArchTag = Arch::AtlasA2;
constexpr bool ENABLE_UNIT_FLAG = true;
constexpr bool ENABLE_SHUFFLE_K = true;
using ElementA = half;
using ElementB = half;
using ElementC = half;
using LayoutA = layout::RowMajor;
using LayoutB = layout::ColumnMajor;
using LayoutC = layout::RowMajor;

using AType = Gemm::GemmType<ElementA, LayoutA>;
using BType = Gemm::GemmType<ElementB, LayoutB>;
using CType = Gemm::GemmType<ElementC, LayoutC>;
using DispatchPolicy = Gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG, ENABLE_SHUFFLE_K>;

// if LayoutA and LayoutB is both ColumnMajor,
// L1TileShape using GemmShape<256, 128, 256> can achieve better performance.
using L1TileShape = std::conditional_t<std::is_same_v<LayoutA, layout::ColumnMajor> &&
    std::is_same_v<LayoutB, layout::ColumnMajor>, GemmShape<256, 128, 256>, GemmShape<128, 256, 256>>;
using L0TileShape = std::conditional_t<std::is_same_v<LayoutA, layout::ColumnMajor> &&
    std::is_same_v<LayoutB, layout::ColumnMajor>, GemmShape<256, 128, 64>, GemmShape<128, 256, 64>>;
using BlockScheduler30 = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
using BlockScheduler31 = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
using BlockEpilogue = void;

struct Options {
    const std::string HELPER = "06_optimizd_matmul m n k [device_id]";

    GemmCoord problemShape{128, 128, 128};
    int32_t deviceId{0};

    Options() = default;

    int Parse(int argc, const char **argv)
    {
        enum ArgsIndex {
            M_INDEX = 1,
            N_INDEX,
            K_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > ARGS_MAX || argc <= K_INDEX) {
            std::cerr << HELPER << std::endl;
            return -1;
        }

        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);
        problemShape.k() = std::atoi(argv[K_INDEX]);
        if (argc == ARGS_MAX) {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        return 0;
    }
};

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

bool IsNeedPadding(layout::zN layout, uint32_t align)
{
    return false;
}

bool IsNeedPadding(layout::nZ layout, uint32_t align)
{
    return false;
}

template <class Adapter>
void RunAdapter(Adapter matmul_op, typename Adapter::Arguments args, aclrtStream stream,
    uint32_t aicCoreNum, uint64_t fftsAddr)
{
    size_t sizeWorkspace = matmul_op.GetWorkspaceSize(args);
    uint8_t *deviceWorkspace = nullptr;
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    matmul_op.Initialize(args, deviceWorkspace);
    matmul_op(stream, aicCoreNum, fftsAddr);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }
}

void Run(Options const &options)
{
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    LayoutA layoutA = LayoutA::template MakeLayout<ElementA>(m, k);
    LayoutB layoutB = LayoutB::template MakeLayout<ElementB>(k, n);
    LayoutC layoutC = LayoutC::template MakeLayout<ElementC>(m, n);
    size_t lenA = layoutA.Capacity();
    size_t lenB = layoutB.Capacity();
    size_t lenC = layoutC.Capacity();

    size_t sizeA = lenA * sizeof(fp16_t);
    size_t sizeB = lenB * sizeof(fp16_t);
    size_t sizeC = lenC * sizeof(fp16_t);

    bool isNeedPaddingA = IsNeedPadding(layoutA, alignByElement);
    bool isNeedPaddingB = IsNeedPadding(layoutB, alignByElement);

    // PaddingTag can be NO_PADDING, PADDING_BLOCK_ND, or PADDING_ND.
    using PaddingTag = Catlass::Gemm::Kernel::PaddingTag;
    // Layout zN or layout nZ does not require padding operation.
    constexpr PaddingTag paddingTagA = (std::is_same_v<LayoutA, layout::zN> || std::is_same_v<LayoutA, layout::nZ>) ?
        PaddingTag::NO_PADDING : PaddingTag::PADDING_BLOCK_ND;
    constexpr PaddingTag paddingTagB = (std::is_same_v<LayoutB, layout::zN> || std::is_same_v<LayoutB, layout::nZ>) ?
        PaddingTag::NO_PADDING : PaddingTag::PADDING_BLOCK_ND;
    static const uint32_t COMPUTE_LENGTH_A = 96 * 1024 / sizeof(ElementA);
    using PaddingBuilderA = Catlass::Gemm::Kernel::PaddingBuilder<
        ArchTag, ElementA, LayoutA, COMPUTE_LENGTH_A, paddingTagA>;
    using GlobalPaddingA = PaddingBuilderA::Padding;
    static const uint32_t COMPUTE_LENGTH_B = 96 * 1024 / sizeof(ElementB);
    using PaddingBuilderB = Catlass::Gemm::Kernel::PaddingBuilder<
        ArchTag, ElementB, LayoutB, COMPUTE_LENGTH_B, paddingTagB>;
    using GlobalPaddingB = PaddingBuilderB::Padding;

    std::vector<fp16_t> hostA(lenA);
    std::vector<fp16_t> hostB(lenB);
    golden::FillRandomData<fp16_t>(hostA, -5.0f, 5.0f);
    golden::FillRandomData<fp16_t>(hostB, -5.0f, 5.0f);

    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    // Prepare FFTS address
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    if (m > n) {
        if (isNeedPaddingA && isNeedPaddingB) {
            using LayoutMmadA = typename PaddingBuilderA::LayoutAfterPadding;
            using LayoutMmadB = typename PaddingBuilderB::LayoutAfterPadding;
            using ATypeMmad = Gemm::GemmType<ElementA, LayoutMmadA>;
            using BTypeMmad = Gemm::GemmType<ElementB, LayoutMmadB>;
            using TileCopy = TileCopyOpt<ArchTag, ATypeMmad, BTypeMmad, CType>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<
                DispatchPolicy, L1TileShape, L0TileShape, ATypeMmad, BTypeMmad, CType, void, TileCopy>;
            using MatmulKernel = Gemm::Kernel::OptimizedMatmul<
                GlobalPaddingA, GlobalPaddingB, BlockMmadOpt, BlockEpilogue, BlockScheduler30>;
            MatmulKernel::Arguments arguments{options.problemShape, deviceA, deviceB, deviceC};
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
            MatmulAdapter matmul_op;
            RunAdapter(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
        } else if (isNeedPaddingA) {
            using LayoutMmadA = typename PaddingBuilderA::LayoutAfterPadding;
            using ATypeMmad = Gemm::GemmType<ElementA, LayoutMmadA>;
            using TileCopy = TileCopyOpt<ArchTag, ATypeMmad, BType, CType>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<
                DispatchPolicy, L1TileShape, L0TileShape, ATypeMmad, BType, CType, void, TileCopy>;
            using MatmulKernel = Gemm::Kernel::OptimizedMatmul<
                GlobalPaddingA, void, BlockMmadOpt, BlockEpilogue, BlockScheduler30>;
            MatmulKernel::Arguments arguments{options.problemShape, deviceA, deviceB, deviceC};
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
            MatmulAdapter matmul_op;
            RunAdapter(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
        } else if (isNeedPaddingB) {
            using LayoutMmadB = typename PaddingBuilderB::LayoutAfterPadding;
            using BTypeMmad = Gemm::GemmType<ElementB, LayoutMmadB>;
            using TileCopy = TileCopyOpt<ArchTag, AType, BTypeMmad, CType>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<
                DispatchPolicy, L1TileShape, L0TileShape, AType, BTypeMmad, CType, void, TileCopy>;
            using MatmulKernel = Gemm::Kernel::OptimizedMatmul<
                void, GlobalPaddingB, BlockMmadOpt, BlockEpilogue, BlockScheduler30>;
            MatmulKernel::Arguments arguments{options.problemShape, deviceA, deviceB, deviceC};
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
            MatmulAdapter matmul_op;
            RunAdapter(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
        } else {
            using TileCopy = TileCopyOpt<ArchTag, AType, BType, CType>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<
                DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, void, TileCopy>;
            using MatmulKernel = Gemm::Kernel::OptimizedMatmul<
                void, void, BlockMmadOpt, BlockEpilogue, BlockScheduler30>;
            MatmulKernel::Arguments arguments{options.problemShape, deviceA, deviceB, deviceC};
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
            MatmulAdapter matmul_op;
            RunAdapter(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
        }
    } else {
        if (isNeedPaddingA && isNeedPaddingB) {
            using LayoutMmadA = typename PaddingBuilderA::LayoutAfterPadding;
            using LayoutMmadB = typename PaddingBuilderB::LayoutAfterPadding;
            using ATypeMmad = Gemm::GemmType<ElementA, LayoutMmadA>;
            using BTypeMmad = Gemm::GemmType<ElementB, LayoutMmadB>;
            using TileCopy = TileCopyOpt<ArchTag, ATypeMmad, BTypeMmad, CType>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<
                DispatchPolicy, L1TileShape, L0TileShape, ATypeMmad, BTypeMmad, CType, void, TileCopy>;
            using MatmulKernel = Gemm::Kernel::OptimizedMatmul<
                GlobalPaddingA, GlobalPaddingB, BlockMmadOpt, BlockEpilogue, BlockScheduler31>;
            MatmulKernel::Arguments arguments{options.problemShape, deviceA, deviceB, deviceC};
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
            MatmulAdapter matmul_op;
            RunAdapter(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
        } else if (isNeedPaddingA) {
            using LayoutMmadA = typename PaddingBuilderA::LayoutAfterPadding;
            using ATypeMmad = Gemm::GemmType<ElementA, LayoutMmadA>;
            using TileCopy = TileCopyOpt<ArchTag, ATypeMmad, BType, CType>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<
                DispatchPolicy, L1TileShape, L0TileShape, ATypeMmad, BType, CType, void, TileCopy>;
            using MatmulKernel = Gemm::Kernel::OptimizedMatmul<
                GlobalPaddingA, void, BlockMmadOpt, BlockEpilogue, BlockScheduler31>;
            MatmulKernel::Arguments arguments{options.problemShape, deviceA, deviceB, deviceC};
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
            MatmulAdapter matmul_op;
            RunAdapter(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
        } else if (isNeedPaddingB) {
            using LayoutMmadB = typename PaddingBuilderB::LayoutAfterPadding;
            using BTypeMmad = Gemm::GemmType<ElementB, LayoutMmadB>;
            using TileCopy = TileCopyOpt<ArchTag, AType, BTypeMmad, CType>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<
                DispatchPolicy, L1TileShape, L0TileShape, AType, BTypeMmad, CType, void, TileCopy>;
            using MatmulKernel = Gemm::Kernel::OptimizedMatmul<
                void, GlobalPaddingB, BlockMmadOpt, BlockEpilogue, BlockScheduler31>;
            MatmulKernel::Arguments arguments{options.problemShape, deviceA, deviceB, deviceC};
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
            MatmulAdapter matmul_op;
            RunAdapter(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
        } else {
            using TileCopy = TileCopyOpt<ArchTag, AType, BType, CType>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<
                DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, void, TileCopy>;
            using MatmulKernel = Gemm::Kernel::OptimizedMatmul<
                void, void, BlockMmadOpt, BlockEpilogue, BlockScheduler31>;
            MatmulKernel::Arguments arguments{ options.problemShape, deviceA, deviceB, deviceC};
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
            MatmulAdapter matmul_op;
            RunAdapter(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
        }
    }

    std::vector<fp16_t> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenC);
    golden::ComputeMatmul(options.problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);

    std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char **argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}
