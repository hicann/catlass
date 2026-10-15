/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue_matmul_permute.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_mmad_pingpong_mutex_tla.hpp"
#include "catlass/gemm/block/block_scheduler_aswt.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/matmul_mix_fixpipe_opti_matmul_permute.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"

#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;
using namespace tla;

using Options = GemmOptions;

namespace {

constexpr uint32_t S1 = 8;
constexpr uint32_t S2 = 4;

enum class MatmulPermuteKernelPreset
{
    Basic,
    VectorPack,
    AswtWindow
};

MatmulPermuteKernelPreset SelectMatmulPermuteKernelPreset(uint32_t m, uint32_t n)
{
    // Keep the host policy small and deterministic.  The two optimized
    // routes are selected only for shapes validated on Ascend950; all other
    // shapes use the generic residue-safe epilogue.
    if (m >= 1024 && n >= 1024) {
        return MatmulPermuteKernelPreset::VectorPack;
    }
    if (m >= 1024 && n >= 512) {
        return MatmulPermuteKernelPreset::AswtWindow;
    }
    return MatmulPermuteKernelPreset::Basic;
}

static_assert(S1 > 0 && S2 > 0, "MatmulPermute requires positive S1/S2.");

using ElementA = half;
using ElementB = half;
using ElementC = half;
using ElementBias = void;

using LayoutTagA = layout::RowMajor;
using LayoutTagB = layout::RowMajor;
using LayoutTagC = layout::RowMajor;

using ArchTag = Arch::Ascend950;
constexpr bool enableUnitFlag = true;
constexpr bool useHF32 = false;
using DispatchPolicy = Gemm::MmadPingpong<ArchTag, enableUnitFlag, useHF32>;

using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
constexpr bool splitM = std::is_same_v<ElementC, ElementAccumulator>;
constexpr auto copyMode = splitM ? Gemm::Tile::CopyL0CToUBMode::SPLIT_M : Gemm::Tile::CopyL0CToUBMode::NO_SPLIT;

using TileCopy = Gemm::Tile::PackedTileCopyTlaToUB<
    ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, void, copyMode>;
using TileMmad = Gemm::Tile::TileMmadTla<DispatchPolicy::ArchTag, ElementA, TileCopy::LayoutTagL1A>;

template <bool UseVectorPack, class Scheduler>
struct MatmulPermuteKernelSelector {
    using L1TileShape = Shape<Int<128>, Int<256>, Int<128>>;
    using L0TileShape = Shape<Int<128>, Int<256>, Int<64>>;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, ElementBias, TileCopy, TileMmad>;
    using BlockEpilogue = Epilogue::Block::BlockEpilogue<
        Epilogue::EpilogueAscend950MatmulPermute<S1, S2, splitM, UseVectorPack>, ElementC, ElementC>;
    using BlockScheduler = Scheduler;
    using Kernel = Gemm::Kernel::KernelMatmulMixFixpipeOptiMatmulPermute<BlockMmad, BlockEpilogue, BlockScheduler>;
    using Adapter = Gemm::Device::DeviceGemm<Kernel>;
};

using BasicL1TileShape = Shape<Int<64>, Int<256>, Int<128>>;
using BasicL0TileShape = Shape<Int<64>, Int<256>, Int<64>>;
using BasicScheduler = Gemm::Block::BlockSchedulerAswt<BasicL1TileShape, BasicL0TileShape>;
using VectorScheduler = Gemm::Block::BlockSchedulerAswt<
    typename MatmulPermuteKernelSelector<true, BasicScheduler>::L1TileShape,
    typename MatmulPermuteKernelSelector<true, BasicScheduler>::L0TileShape>;
using WindowScheduler = Gemm::Block::BlockSchedulerAswt<
    typename MatmulPermuteKernelSelector<false, BasicScheduler>::L1TileShape,
    typename MatmulPermuteKernelSelector<false, BasicScheduler>::L0TileShape>;
struct BasicMatmulPermuteSelector {
    using L1TileShape = BasicL1TileShape;
    using L0TileShape = BasicL0TileShape;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, ElementBias, TileCopy, TileMmad>;
    using BlockEpilogue =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAscend950MatmulPermute<S1, S2, splitM, false>, ElementC, ElementC>;
    using BlockScheduler = BasicScheduler;
    using Kernel = Gemm::Kernel::KernelMatmulMixFixpipeOptiMatmulPermute<BlockMmad, BlockEpilogue, BlockScheduler>;
    using Adapter = Gemm::Device::DeviceGemm<Kernel>;
};
using VectorKernelSelector = MatmulPermuteKernelSelector<true, VectorScheduler>;
using WindowKernelSelector = MatmulPermuteKernelSelector<false, WindowScheduler>;

std::vector<float> ComputeMatmulPermute0213(
    const GemmCoord& shape, const std::vector<fp16_t>& a, const layout::RowMajor& layoutA, const std::vector<fp16_t>& b,
    const layout::RowMajor& layoutB)
{
    const uint32_t m = shape.m();
    const uint32_t n = shape.n();
    const uint32_t nOuter = n / S2;
    std::vector<float> matmul(m * n);
    layout::RowMajor layoutC = layout::RowMajor::MakeLayout<float>(m, n);
    golden::ComputeMatmul(shape, a, layoutA, b, layoutB, matmul, layoutC);

    std::vector<float> out(m * n);
    for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t col = 0; col < n; ++col) {
            uint32_t dst = ((((row / S1) * S2 + (col / nOuter)) * S1 + (row % S1)) * nOuter + (col % nOuter));
            out[dst] = matmul[row * n + col];
        }
    }
    return out;
}

template <class KernelSelector>
static void Run(const Options& options)
{
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();
    if (m % S1 != 0 || n % S2 != 0) {
        std::cerr << "MatmulPermute requires M % " << S1 << " == 0, N % " << S2 << " == 0 on Ascend950."
                  << std::endl;
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
        return;
    }

    LayoutTagA tagA = LayoutTagA::MakeLayout<ElementA>(m, k);
    LayoutTagB tagB = LayoutTagB::MakeLayout<ElementB>(k, n);
    LayoutTagC tagC = LayoutTagC::MakeLayout<ElementC>(m, n);

    size_t lenA = tagA.Capacity();
    size_t lenB = tagB.Capacity();
    size_t lenC = tagC.Capacity();
    size_t sizeA = lenA * sizeof(ElementA);
    size_t sizeB = lenB * sizeof(ElementB);
    size_t sizeC = lenC * sizeof(ElementC);
    std::vector<fp16_t> hostA(lenA);
    std::vector<fp16_t> hostB(lenB);
    golden::FillRandomData<fp16_t>(hostA, -5.0f, 5.0f);
    golden::FillRandomData<fp16_t>(hostB, -5.0f, 5.0f);

    uint8_t* deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* deviceWorkspace{nullptr};
    size_t sizeWorkspace{0};
    const auto aclReleaseEarly = [&]() {
        if (deviceWorkspace != nullptr) {
            ACL_CHECK(aclrtFree(deviceWorkspace));
            deviceWorkspace = nullptr;
        }
        ACL_CHECK(aclrtFree(deviceA));
        ACL_CHECK(aclrtFree(deviceB));
        ACL_CHECK(aclrtFree(deviceC));
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
    };

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
    (void)layoutA;
    (void)layoutB;

    using L1TileShape = typename KernelSelector::L1TileShape;
    using MatmulKernel = typename KernelSelector::Kernel;
    using MatmulAdapter = typename KernelSelector::Adapter;
    uint32_t taskNum = CeilDiv(m, tla::get<0>(L1TileShape{})) * CeilDiv(n, tla::get<1>(L1TileShape{}));
    uint32_t aicCoreUsed = std::min<uint32_t>(aicCoreNum, taskNum);
    typename MatmulKernel::Arguments arguments{options.problemShape, deviceA, deviceB, deviceC};
    MatmulAdapter matmulOp;
    if (matmulOp.CanImplement(arguments) != Catlass::Status::kSuccess) {
        std::cerr << "Unsupported matmul-permute shape on Ascend950.\n";
        aclReleaseEarly();
        return;
    }
    sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    matmulOp.Initialize(arguments, deviceWorkspace);
    matmulOp(stream, aicCoreUsed);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<fp16_t> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    auto hostGolden = ComputeMatmulPermute0213(options.problemShape, hostA, tagA, hostB, tagB);
    std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
        uint64_t printCount = std::min<uint64_t>(errorIndices.size(), 16);
        for (uint64_t i = 0; i < printCount; ++i) {
            uint64_t idx = errorIndices[i];
            std::cerr << "  mismatch[" << i << "] index=" << idx << " result=" << static_cast<float>(hostC[idx])
                      << " expect=" << hostGolden[idx] << std::endl;
        }
    }

    aclReleaseEarly();
}

} // namespace

int main(int argc, const char** argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    switch (SelectMatmulPermuteKernelPreset(options.problemShape.m(), options.problemShape.n())) {
        case MatmulPermuteKernelPreset::Basic:
            Run<BasicMatmulPermuteSelector>(options);
            break;
        case MatmulPermuteKernelPreset::VectorPack:
            Run<VectorKernelSelector>(options);
            break;
        case MatmulPermuteKernelPreset::AswtWindow:
            Run<WindowKernelSelector>(options);
            break;
    }
    return 0;
}
