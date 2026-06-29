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

///////////////////////////////////////////////////////////////////////////////
/// Example 75: Ascend950 FP8 Per-Token Per-Channel Quantized Matrix Multiplication
///               with Epilogue-based Quantization
///
/// Computes: out = (A @ B) * perTokenScale * perChannelScale
///
/// Where:
///   - A: float8_e4m3_t, shape (M, K)
///   - B: float8_e4m3_t, shape (K, N)
///   - perTokenScale: float8_e4m3_t, shape (M,)
///   - perChannelScale: float8_e4m3_t, shape (N,)
///   - out: float, shape (M, N)
///
/// Architecture: Ascend950, epilogue-based quantization
///   AIC core: reads A and B from GM, performs matmul, writes intermediate C to GM workspace
///   AIV core: reads C from workspace, applies per-token and per-channel scales, writes output to GM
///
/// This example uses raw element types and layout tags instead of GemmType wrappers,
/// following the TLA pattern.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/matmul_per_token_per_channel_epilogue_tla.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"

#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;
using namespace tla;
using Options = GemmOptions;

static const std::string kDataRoot = "../../examples/75_ascend950_fp8_epilogue_quant_matmul/data";

///////////////////////////////////////////////////////////////////////////////
/// Run: Main execution function
///////////////////////////////////////////////////////////////////////////////
static void Run(const Options &options)
{
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    // ========================================
    // Step 1: Type definitions
    // ========================================

    using ArchTag = Arch::Ascend950;

    // Input data types
    using ElementA = float8_e4m3_t;
    using ElementB = float8_e4m3_t;
    using ElementC = float;  // Intermediate result type (matmul accumulator)
    using ElementD = float;   // Output type

    // Scale types
    using ElementPerTokenScale = float8_e4m3_t;
    using ElementPerChannelScale = float8_e4m3_t;

    // Layout tags (TLA layout types for Ascend950)
    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::RowMajor;
    using LayoutTagC = layout::RowMajor;
    using LayoutTagD = layout::RowMajor;
    using LayoutTagScale = layout::VectorLayout;
    using LayoutTagPerTokenScale = layout::VectorLayout;

    LayoutTagA tagA{m, k};
    LayoutTagB tagB{k, n};
    LayoutTagC tagC{m, n};

    LayoutTagD tagD{m, n};
    auto layoutA = MakeLayoutFromTag(tagA);
    auto layoutB = MakeLayoutFromTag(tagB);
    
    LayoutTagD layoutD{m, n};
    LayoutTagScale layoutPerChannelScale{n};
    LayoutTagPerTokenScale layoutPerTokenScale{m};

    // ========================================
    // Step 2: Tile shapes
    // ========================================

    // L1/L0 tile shapes optimized for Ascend950
    using L1TileShape = Shape<Int<256>, Int<256>, Int<512>>;
    using L0TileShape = Shape<Int<256>, Int<256>, Int<64>>;

    // Epilogue tile shape for UB operations
    using EpilogueTileShape = MatrixShape<128, 256>;

    // ========================================
    // Step 3: Tile compute operations for epilogue
    // ========================================

    // Compute type for epilogue
    using ComputeType = Gemm::GemmType<float, LayoutTagC>;

    // Broadcast ops for per-token scale (broadcast along columns)
    using TileBroadcastOneBlkOp = Epilogue::Tile::TileBroadcastOneBlk<ArchTag, ComputeType, EpilogueTileShape::ROW>;
    using TileOneBlkColumnBroadcastMulOp =
        Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag, ComputeType, EpilogueTileShape>;

    // Broadcast ops for per-channel scale (broadcast along rows)
    using TileRowBroadcastMulOp = Epilogue::Tile::TileRowBroadcastMul<ArchTag, ComputeType, EpilogueTileShape>;

    // ========================================
    // Step 4: Tile copy operations for epilogue
    // ========================================

    // GemmType wrappers for copy operations
    using CType = Gemm::GemmType<ElementC, LayoutTagC>;
    constexpr uint32_t ubStages = 1;

    using EpilogueDispatchPolicy = Epilogue::EpilogueAscend950PerTokenDequant<ubStages>;
    using ScaleType = Gemm::GemmType<ElementPerChannelScale, layout::VectorLayout>;
    using PerTokenScaleType = Gemm::GemmType<ElementPerTokenScale, layout::VectorLayout>;
    using DType = Gemm::GemmType<ElementD, layout::RowMajor>;

    using RowBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
    using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
    using OneBlkColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;

    using TileRowBroadcastMul = Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;
    using TileBroadcastOneBlk =
        Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType, EpilogueTileShape::ROW>;
    using TileOneBlkColumnBroadcastMul =
        Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag, OneBlkColumnBroadcastMulType, EpilogueTileShape>;
    using TileCopyEpilogue = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, DType>;
    using TileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    using BlockEpilogue = Epilogue::Block::BlockEpilogue<
        EpilogueDispatchPolicy, CType, ScaleType, PerTokenScaleType, DType, TileRowBroadcastMul, TileBroadcastOneBlk,
        TileOneBlkColumnBroadcastMul, TileCopyEpilogue, TileScheduler>;

    // ========================================
    // Step 6: Block MMAD (AIC core)
    // ========================================

    // Tile copy for matmul
    using TileCopy =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;

    constexpr bool enableUnitFlag = true;
    constexpr bool useHF32 = false;
    using DispatchPolicy = Gemm::MmadPingpongPreLoad<ArchTag, enableUnitFlag, useHF32>;

    // Block MMAD
    using BlockMmadTla = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;


    // ========================================
    // Step 7: Kernel
    // ========================================

    using BlockScheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    using GemmKernel = Gemm::Kernel::MatmulPerTokenPerChannelEpilogueTla<BlockMmadTla, BlockEpilogue, BlockScheduler, ubStages>;
    using GemmAdapter = Gemm::Device::DeviceGemm<GemmKernel>;

    // ========================================
    // Step 8: Memory allocation
    // ========================================

    size_t lenInputA = static_cast<size_t>(m) * k;
    size_t lenInputB = static_cast<size_t>(k) * n;
    size_t lenPerTokenScale = static_cast<size_t>(m);
    size_t lenPerChannelScale = static_cast<size_t>(n);
    size_t lenD = static_cast<size_t>(m) * n;

    size_t sizeInputA = lenInputA * sizeof(ElementA);
    size_t sizeInputB = lenInputB * sizeof(ElementB);
    size_t sizePerTokenScale = lenPerTokenScale * sizeof(ElementPerTokenScale);
    size_t sizePerChannelScale = lenPerChannelScale * sizeof(ElementPerChannelScale);
    size_t sizeD = lenD * sizeof(ElementD);

    // Host memory
    std::vector<uint8_t> hostA(lenInputA * sizeof(ElementA));
    std::vector<uint8_t> hostB(lenInputB * sizeof(ElementB));
    std::vector<uint8_t> hostPerTokenScale(lenPerTokenScale * sizeof(ElementPerTokenScale));
    std::vector<uint8_t> hostPerChannelScale(lenPerChannelScale * sizeof(ElementPerChannelScale));

    // Read input data from files
    const auto releaseAclEarly = [&]() {
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
    };
    if (!ReadFile(kDataRoot + "/input/a_8.bin", hostA.data(), sizeInputA)) {
        releaseAclEarly();
        return;
    }
    if (!ReadFile(kDataRoot + "/input/b_8.bin", hostB.data(), sizeInputB)) {
        releaseAclEarly();
        return;
    }
    if (!ReadFile(kDataRoot + "/input/a_scale.bin", hostPerTokenScale.data(), sizePerTokenScale)) {
        releaseAclEarly();
        return;
    }
    if (!ReadFile(kDataRoot + "/input/b_scale.bin", hostPerChannelScale.data(), sizePerChannelScale)) {
        releaseAclEarly();
        return;
    }

    // Device memory
    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeInputA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeInputA, hostA.data(), sizeInputA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeInputB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeInputB, hostB.data(), sizeInputB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *devicePerTokenScale{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devicePerTokenScale), sizePerTokenScale, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(devicePerTokenScale, sizePerTokenScale, hostPerTokenScale.data(), sizePerTokenScale, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *devicePerChannelScale{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devicePerChannelScale), sizePerChannelScale, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(devicePerChannelScale, sizePerChannelScale, hostPerChannelScale.data(), sizePerChannelScale, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceD{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceD), sizeD, ACL_MEM_MALLOC_HUGE_FIRST));

    // ========================================
    // Step 9: Kernel configuration and execution
    // ========================================

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    uint32_t taskNum = CeilDiv(m, static_cast<uint32_t>(tla::get<0>(L1TileShape{}))) *
                       CeilDiv(n, static_cast<uint32_t>(tla::get<1>(L1TileShape{})));

    std::cout << "taskNum: " << taskNum << std::endl;
    uint32_t aicCoreUsed = min(aicCoreNum, taskNum);

    // Set up kernel arguments (using TLA layouts defined earlier)
    typename GemmKernel::Arguments arguments{
        options.problemShape,
        deviceA, layoutA,
        deviceB, layoutB,
        devicePerTokenScale, layoutPerTokenScale,
        devicePerChannelScale, layoutPerChannelScale,
        deviceD, layoutD,
        aicCoreUsed
    };

    GemmAdapter gemmOp;

    // Check if kernel can be implemented
    if (gemmOp.CanImplement(arguments) == Status::kInvalid) {
        std::cerr << "Gemm op cannot be implemented. Please check shape requirements." << std::endl;
        return;
    }

    // Allocate workspace for intermediate C
    size_t sizeWorkspace = gemmOp.GetWorkspaceSize(arguments);
    uint8_t *deviceWorkspace{nullptr};
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // Initialize and run
    gemmOp.Initialize(arguments, deviceWorkspace);
    gemmOp(stream, aicCoreUsed);

    ACL_CHECK(aclrtSynchronizeStream(stream));

    // ========================================
    // Step 10: Result readback and save
    // ========================================

    std::vector<float> hostD(lenD);
    ACL_CHECK(aclrtMemcpy(hostD.data(), sizeD, deviceD, sizeD, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenD);
    if (!ReadFile(kDataRoot + "/golden/expected_data.bin", hostGolden.data(), sizeD)) {
        ACL_CHECK(aclrtFree(deviceA));
        ACL_CHECK(aclrtFree(deviceB));
        ACL_CHECK(aclrtFree(devicePerTokenScale));
        ACL_CHECK(aclrtFree(devicePerChannelScale));
        ACL_CHECK(aclrtFree(deviceD));
        if (deviceWorkspace != nullptr) {
            ACL_CHECK(aclrtFree(deviceWorkspace));
        }
        releaseAclEarly();
        return;
    }

    std::vector<uint64_t> errorIndices = golden::CompareData(hostD, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    std::cout << "Ascend950 FP8 Per-Token Per-Channel Quantized Matmul (Epilogue) completed." << std::endl;
    std::cout << "Problem shape: M=" << m << " N=" << n << " K=" << k << std::endl;
    std::cout << "Workspace size: " << sizeWorkspace << " bytes" << std::endl;

    // ========================================
    // Step 11: Cleanup
    // ========================================

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(devicePerTokenScale));
    ACL_CHECK(aclrtFree(devicePerChannelScale));
    ACL_CHECK(aclrtFree(deviceD));
    if (deviceWorkspace != nullptr) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

///////////////////////////////////////////////////////////////////////////////
/// Main
///////////////////////////////////////////////////////////////////////////////
int main(int argc, const char **argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }

    std::cout << "Running Ascend950 FP8 Per-Token Per-Channel Quantized Matmul (Epilogue)..." << std::endl;
    Run(options);

    return 0;
}
