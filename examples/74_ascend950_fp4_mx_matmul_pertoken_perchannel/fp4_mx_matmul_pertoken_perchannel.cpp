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

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0,
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/mx_matmul_pertoken_perchannel_tla.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "tla/layout.hpp"
#include "helper.hpp"
#include "golden.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"

using namespace Catlass;
using namespace tla;

template <class Dtype>
void MatmulKernelRun(GM_ADDR deviceA, GM_ADDR deviceB, GM_ADDR deviceMxScaleA, GM_ADDR deviceMxScaleB,
    GM_ADDR deviceScale, GM_ADDR devicePerTokenScale, GM_ADDR deviceD,
    uint8_t *deviceWorkspace, uint32_t m, uint32_t n, uint32_t k, aclrtStream stream)
{
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    constexpr uint32_t workspaceStages = 2;
    size_t sizeWorkspace = 0;
    uint32_t mxScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(k);
    uint32_t mxScaleAlignedK = RoundUp<2>(mxScaleK);

    using ElementA = float4_e2m1x2_t;
    using ElementB = float4_e2m1x2_t;
    using ElementC = float;
    using ElementMxScale = float8_e8m0_t;
    using ElementScale = float8_e4m3_t;        // per-channel scale
    using ElementPerTokenScale = float8_e4m3_t; // per-token scale
    using ElementD = Dtype;

    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::RowMajor;
    using LayoutTagC = layout::RowMajor;
    using LayoutTagD = layout::RowMajor;
    using LayoutTagScale = layout::VectorLayout;
    using LayoutTagPerTokenScale = layout::VectorLayout;

    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagA, false>(m, mxScaleK);
    auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagB, true>(mxScaleK, n);

    LayoutTagD layoutD{m, n};
    LayoutTagScale layoutScale{n};
    LayoutTagPerTokenScale layoutPerTokenScale{m};

    using ArchTag = Arch::Ascend950;
    constexpr bool enableUnitFlag = true;
    using DispatchPolicy = Gemm::MmadMx<ArchTag, enableUnitFlag>;

    using L1TileShape = Shape<Int<256>, Int<256>, Int<512>>;
    using L0TileShape = Shape<Int<256>, Int<256>, Int<256>>;

    using TileCopy = Gemm::Tile::PackedMxTileCopyTla<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementMxScale, decltype(layoutMxScaleA),
        ElementMxScale, decltype(layoutMxScaleB), ElementC, LayoutTagC, void>;

    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;

    using CType = Gemm::GemmType<ElementC, layout::RowMajor>;

    constexpr uint32_t ubStages = 2;
    using EpilogueDispatchPolicy = Epilogue::EpilogueAscend950PerTokenPerChannelQuant<ubStages>;
    using ScaleType = Gemm::GemmType<ElementScale, layout::VectorLayout>;
    using PerTokenScaleType = Gemm::GemmType<ElementPerTokenScale, layout::VectorLayout>;
    using DType = Gemm::GemmType<ElementD, layout::RowMajor>;

    using RowBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
    using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
    using OneBlkColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;

    using EpilogueTileShape = MatrixShape<32, 256>;
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

    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

    using MatmulKernel = Gemm::Kernel::MxMatmulPerTokenPerChannelTla<
        BlockMmad, BlockEpilogue, BlockScheduler, workspaceStages>;

    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

    GemmCoord problemShape{m, n, k};
    typename MatmulKernel::Arguments arguments{
        problemShape, aicCoreNum,
        deviceA, layoutA,
        deviceB, layoutB,
        deviceMxScaleA, layoutMxScaleA,
        deviceMxScaleB, layoutMxScaleB,
        deviceScale, layoutScale,
        devicePerTokenScale, layoutPerTokenScale,
        deviceD, layoutD
    };

    MatmulAdapter matmulOp;
    matmulOp.CanImplement(arguments);
    sizeWorkspace = matmulOp.GetWorkspaceSize(arguments);
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    matmulOp.Initialize(arguments, deviceWorkspace);
    matmulOp(stream, aicCoreNum);
}

using Options = GemmOptions;

static const std::string kDataRoot = "../../examples/74_ascend950_fp4_mx_matmul_pertoken_perchannel/data";

void Run(const Options &options)
{
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();
    uint32_t mxScaleK = CeilDiv<MX_SCALE_GROUP_NUM>(k);
    uint32_t mxScaleAlignedK = RoundUp<2>(mxScaleK);

    using ElementA = float4_e2m1x2_t;
    using ElementB = float4_e2m1x2_t;
    using ElementMxScale = float8_e8m0_t;
    using ElementScale = float8_e4m3_t;
    using ElementPerTokenScale = float8_e4m3_t;

    using LayoutTagA = layout::RowMajor;
    using LayoutTagB = layout::ColumnMajor;

    LayoutTagA tagA = LayoutTagA::MakeLayout<ElementA>(m, k);
    LayoutTagB tagB = LayoutTagB::MakeLayout<ElementB>(k, n);

    size_t lenA = tagA.Capacity();
    size_t lenB = tagB.Capacity();
    uint32_t lenMxScaleA = m * mxScaleAlignedK;
    uint32_t lenMxScaleB = mxScaleAlignedK * n;
    size_t lenD = static_cast<size_t>(m) * n;
    size_t lenScale = static_cast<size_t>(n);
    size_t lenPerTokenScale = static_cast<size_t>(m);

    size_t sizeA = lenA / 2;
    size_t sizeB = lenB / 2;
    size_t sizeMxScaleA = lenMxScaleA * sizeof(ElementMxScale);
    size_t sizeMxScaleB = lenMxScaleB * sizeof(ElementMxScale);
    size_t sizeD = lenD * sizeof(float); // default float output
    size_t sizeScale = lenScale * sizeof(ElementScale);
    size_t sizePerTokenScale = lenPerTokenScale * sizeof(ElementPerTokenScale);

    std::vector<int8_t> hostA(sizeA);
    std::vector<int8_t> hostB(sizeB);
    // MxScale 固定为 1: 用 fp8_e8m0 编码的 1.0，即指数 = 127，编码 = 127
    // 但直接填 0x7F (127) 即 2^(127-128) = 2^(-1) = 0.5，不对
    // fp8_e8m0: value = 2^(exp - 128)，exp=128 => value=1.0，但 exp 是 8 位无符号，范围 0~255
    // exp byte = 128 => 0x80 => value = 2^(128-128) = 2^0 = 1.0
    std::vector<uint8_t> hostMxScaleA(lenMxScaleA,0x7F); // all 1.0
    std::vector<uint8_t> hostMxScaleB(lenMxScaleB, 0x7F); // all 1.0
    std::vector<int8_t> hostScale(lenScale);
    std::vector<int8_t> hostPerTokenScale(lenPerTokenScale);

    const auto releaseAclEarly = [&]() {
        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(options.deviceId));
        ACL_CHECK(aclFinalize());
    };
    if (!ReadFile(kDataRoot + "/input/a_4.bin", hostA.data(), sizeA)) {
        releaseAclEarly();
        return;
    }
    if (!ReadFile(kDataRoot + "/input/b_4.bin", hostB.data(), sizeB)) {
        releaseAclEarly();
        return;
    }
    if (!ReadFile(kDataRoot + "/input/a_scale4.bin", hostPerTokenScale.data(), sizePerTokenScale)) {
        releaseAclEarly();
        return;
    }
    if (!ReadFile(kDataRoot + "/input/b_scale4.bin", hostScale.data(), sizeScale)) {
        releaseAclEarly();
        return;
    }

    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceMxScaleA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceMxScaleA), sizeMxScaleA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceMxScaleA, sizeMxScaleA, hostMxScaleA.data(), sizeMxScaleA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceMxScaleB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceMxScaleB), sizeMxScaleB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceMxScaleB, sizeMxScaleB, hostMxScaleB.data(), sizeMxScaleB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceScale{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceScale), sizeScale, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceScale, sizeScale, hostScale.data(), sizeScale, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *devicePerTokenScale{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devicePerTokenScale), sizePerTokenScale, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(devicePerTokenScale, sizePerTokenScale, hostPerTokenScale.data(), sizePerTokenScale,
        ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceD{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceD), sizeD, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *deviceWorkspace{nullptr};

    MatmulKernelRun<float>(deviceA, deviceB, deviceMxScaleA, deviceMxScaleB, deviceScale, devicePerTokenScale,
        deviceD, deviceWorkspace, m, n, k, stream);

    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<float> hostD(lenD);
    ACL_CHECK(aclrtMemcpy(hostD.data(), sizeD, deviceD, sizeD, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenD);
    if (!ReadFile(kDataRoot + "/golden/expected_data.bin", hostGolden.data(), sizeD)) {
        ACL_CHECK(aclrtFree(deviceA));
        ACL_CHECK(aclrtFree(deviceB));
        ACL_CHECK(aclrtFree(deviceMxScaleA));
        ACL_CHECK(aclrtFree(deviceMxScaleB));
        ACL_CHECK(aclrtFree(deviceScale));
        ACL_CHECK(aclrtFree(devicePerTokenScale));
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

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceMxScaleA));
    ACL_CHECK(aclrtFree(deviceMxScaleB));
    ACL_CHECK(aclrtFree(deviceScale));
    ACL_CHECK(aclrtFree(devicePerTokenScale));
    ACL_CHECK(aclrtFree(deviceD));
    if (deviceWorkspace != nullptr) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }
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
