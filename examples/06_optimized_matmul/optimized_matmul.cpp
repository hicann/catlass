/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <vector>

#include "helper.hpp"
#include "golden.hpp"
#include "fp16_t.h"

#include "act/act.hpp"
#include "act/arch/arch.hpp"
#include "act/layout/layout.hpp"
#include "act/gemm/block/block_mmad.hpp"
#include "act/gemm/block/block_swizzle.hpp"
#include "act/gemm/dispatch_policy.hpp"
#include "act/gemm/kernel/optimized_matmul.hpp"
#include "act/gemm/gemm_type.hpp"

#include "act/status.hpp"
#include "act/gemm/device/device_gemm.hpp"

using namespace Act;
using fp16_t = op::fp16_t;

using ArchTag = Arch::AtlasA2;

constexpr bool ENABLE_UNIT_FLAG = true;
constexpr bool ENABLE_SHUFFLE_K = true;
using ElementA = half;
using ElementB = half;
using ElementC = half;
using ElementWorkspace = float;
using LayoutA = layout::RowMajor;
using LayoutB = layout::ColumnMajor;
using LayoutC = layout::RowMajor;
using LayoutPaddingA = std::conditional_t<std::is_same_v<LayoutA, layout::RowMajor>,
            layout::PaddingRowMajor, layout::PaddingColumnMajor>;
using LayoutPaddingB = std::conditional_t<std::is_same_v<LayoutB, layout::RowMajor>,
            layout::PaddingRowMajor, layout::PaddingColumnMajor>;
using AType = Gemm::GemmType<ElementA, LayoutA>;
using BType = Gemm::GemmType<ElementB, LayoutB>;
using CType = Gemm::GemmType<ElementC, LayoutC>;
using ATypePadding = Gemm::GemmType<ElementA, LayoutPaddingA>;
using BTypePadding = Gemm::GemmType<ElementB, LayoutPaddingB>;
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
    if (layout.stride(0) < STRIDE_LIMIT) {
        return layout.stride(0) % align != 0;
    } else {
        return true;
    }
}

bool IsNeedPadding(layout::ColumnMajor layout, uint32_t align)
{
    // If the stride is greater than 65536, padding is required to reduce the stride.
    if (layout.stride(1) < STRIDE_LIMIT) {
        return layout.stride(1) % align != 0;
    } else {
        return true;
    }
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

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(m) * n;

    size_t sizeA = lenA * sizeof(fp16_t);
    size_t sizeB = lenB * sizeof(fp16_t);
    size_t sizeC = lenC * sizeof(fp16_t);

    const uint32_t align = 256;
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};
    bool isNeedPaddingA = IsNeedPadding(layoutA, align);
    bool isNeedPaddingB = IsNeedPadding(layoutB, align);

    // if LayoutA and LayoutB is both ColumnMajor,
    // L1TileShape using GemmShape<256, 128, 256> can achieve better performance.
    using L1TileShape = std::conditional_t<std::is_same_v<LayoutA, layout::ColumnMajor> && 
        std::is_same_v<LayoutB, layout::ColumnMajor>, GemmShape<256, 128, 256>, GemmShape<128, 256, 256>>;

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
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    if (m > n) {
        if (isNeedPaddingA && isNeedPaddingB) {
            using BlockMmadOpt = Gemm::Block::BlockMmad<
                DispatchPolicy, L1TileShape, L0TileShape, ATypePadding, BTypePadding, CType>;
            using MatmulKernel = Gemm::Kernel::OptimizedMatmul<BlockMmadOpt, BlockEpilogue, BlockScheduler30>;
            LayoutPaddingA layoutWA = LayoutPaddingA(
                layoutA.shape(0), layoutA.shape(1), L1TileShape::M, L1TileShape::K);
            LayoutPaddingB layoutWB = LayoutPaddingB(
                layoutB.shape(0), layoutB.shape(1), L1TileShape::K, L1TileShape::N);
            MatmulKernel::Arguments arguments{
                options.problemShape, align, sizeof(ElementWorkspace),
                layoutWA, layoutWB, deviceA, deviceB, deviceC};
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
            MatmulAdapter matmul_op;
            RunAdapter(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
        } else if (isNeedPaddingA) {
            using BlockMmadOpt = Gemm::Block::BlockMmad<
                DispatchPolicy, L1TileShape, L0TileShape, ATypePadding, BType, CType>;
            using MatmulKernel = Gemm::Kernel::OptimizedMatmul<BlockMmadOpt, BlockEpilogue, BlockScheduler30>;
            LayoutPaddingA layoutWA = LayoutPaddingA(
                layoutA.shape(0), layoutA.shape(1), L1TileShape::M, L1TileShape::K);
            MatmulKernel::Arguments arguments{
                options.problemShape, align, sizeof(ElementWorkspace),
                layoutWA, layoutB, deviceA, deviceB, deviceC};
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
            MatmulAdapter matmul_op;
            RunAdapter(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
        } else if (isNeedPaddingB) {
            using BlockMmadOpt = Gemm::Block::BlockMmad<
                DispatchPolicy, L1TileShape, L0TileShape, AType, BTypePadding, CType>;
            using MatmulKernel = Gemm::Kernel::OptimizedMatmul<BlockMmadOpt, BlockEpilogue, BlockScheduler30>;
            LayoutPaddingB layoutWB = LayoutPaddingB(
                layoutB.shape(0), layoutB.shape(1), L1TileShape::K, L1TileShape::N);
            MatmulKernel::Arguments arguments{
                options.problemShape, align, sizeof(ElementWorkspace),
                layoutA, layoutWB, deviceA, deviceB, deviceC};
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
            MatmulAdapter matmul_op;
            RunAdapter(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
        } else {
            using BlockMmadOpt = Gemm::Block::BlockMmad<
                DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
            using MatmulKernel = Gemm::Kernel::OptimizedMatmul<BlockMmadOpt, BlockEpilogue, BlockScheduler30>;
            MatmulKernel::Arguments arguments{
                options.problemShape, align, sizeof(ElementWorkspace),
                layoutA, layoutB, deviceA, deviceB, deviceC};
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
            MatmulAdapter matmul_op;
            RunAdapter(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
        }
    } else {
        if (isNeedPaddingA && isNeedPaddingB) {
            using BlockMmadOpt = Gemm::Block::BlockMmad<
                DispatchPolicy, L1TileShape, L0TileShape, ATypePadding, BTypePadding, CType>;
            using MatmulKernel = Gemm::Kernel::OptimizedMatmul<BlockMmadOpt, BlockEpilogue, BlockScheduler31>;
            LayoutPaddingA layoutWA = LayoutPaddingA(
                layoutA.shape(0), layoutA.shape(1), L1TileShape::M, L1TileShape::K);
            LayoutPaddingB layoutWB = LayoutPaddingB(
                layoutB.shape(0), layoutB.shape(1), L1TileShape::K, L1TileShape::N);
            MatmulKernel::Arguments arguments{
                options.problemShape, align, sizeof(ElementWorkspace),
                layoutWA, layoutWB, deviceA, deviceB, deviceC};
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
            MatmulAdapter matmul_op;
            RunAdapter(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
        } else if (isNeedPaddingA) {
            using BlockMmadOpt = Gemm::Block::BlockMmad<
                DispatchPolicy, L1TileShape, L0TileShape, ATypePadding, BType, CType>;
            using MatmulKernel = Gemm::Kernel::OptimizedMatmul<BlockMmadOpt, BlockEpilogue, BlockScheduler31>;
            LayoutPaddingA layoutWA = LayoutPaddingA(
                layoutA.shape(0), layoutA.shape(1), L1TileShape::M, L1TileShape::K);
            MatmulKernel::Arguments arguments{
                options.problemShape, align, sizeof(ElementWorkspace),
                layoutWA, layoutB, deviceA, deviceB, deviceC};
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
            MatmulAdapter matmul_op;
            RunAdapter(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
        } else if (isNeedPaddingB) {
            using BlockMmadOpt = Gemm::Block::BlockMmad<
                DispatchPolicy, L1TileShape, L0TileShape, AType, BTypePadding, CType>;
            using MatmulKernel = Gemm::Kernel::OptimizedMatmul<BlockMmadOpt, BlockEpilogue, BlockScheduler31>;
            LayoutPaddingB layoutWB = LayoutPaddingB(
                layoutB.shape(0), layoutB.shape(1), L1TileShape::K, L1TileShape::N);
            MatmulKernel::Arguments arguments{
                options.problemShape, align, sizeof(ElementWorkspace),
                layoutA, layoutWB, deviceA, deviceB, deviceC};
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
            MatmulAdapter matmul_op;
            RunAdapter(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
        } else {
            using BlockMmadOpt = Gemm::Block::BlockMmad<
                DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
            using MatmulKernel = Gemm::Kernel::OptimizedMatmul<BlockMmadOpt, BlockEpilogue, BlockScheduler31>;
            MatmulKernel::Arguments arguments{
                options.problemShape, align, sizeof(ElementWorkspace),
                layoutA, layoutB, deviceA, deviceB, deviceC};
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
            MatmulAdapter matmul_op;
            RunAdapter(matmul_op, arguments, stream, aicCoreNum, fftsAddr);
        }
    }

    std::vector<fp16_t> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<ElementWorkspace> hostGolden(lenC);
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
