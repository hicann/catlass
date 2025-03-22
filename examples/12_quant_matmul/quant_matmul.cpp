/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cstdlib>

#include "helper.hpp"
#include "golden.hpp"
#include "bfloat16.h"

#include "AscendCT/AscendCT.hpp"
#include "AscendCT/arch/arch.hpp"
#include "AscendCT/epilogue/block/block_epilogue.hpp"
#include "AscendCT/epilogue/dispatch_policy.hpp"
#include "AscendCT/epilogue/tile/tile_broadcast_mul.hpp"
#include "AscendCT/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "AscendCT/epilogue/tile/tile_swizzle.hpp"
#include "AscendCT/gemm/block/block_mmad.hpp"
#include "AscendCT/gemm/block/block_swizzle.hpp"
#include "AscendCT/gemm/dispatch_policy.hpp"
#include "AscendCT/gemm/kernel/quant_matmul_multistage_workspace.hpp"
#include "AscendCT/gemm/matmul_type.hpp"
#include "AscendCT/layout/layout.hpp"

#include "AscendCT/status.hpp"
#include "AscendCT/gemm/device/matmul_universal_adapter.hpp"

using namespace AscendCT;
using bfloat16 = op::bfloat16;

using L1TileShape = MatmulShape<128, 256, 512>;
constexpr uint32_t workspaceStages = 2;

struct Options {
    const std::string HELPER = "12_quant_matmul m n k [device_id]";

    MatmulCoord problemShape{128, 128, 128};
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

void Run(Options const & options)
{
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenScale = static_cast<size_t>(n);
    size_t lenPerTokenScale = static_cast<size_t>(m);
    size_t lenD = static_cast<size_t>(m) * n;

    size_t sizeA = lenA * sizeof(int8_t);
    size_t sizeB = lenB * sizeof(int8_t);
    size_t sizeScale = lenScale * sizeof(bfloat16);
    size_t sizePerTokenScale = lenPerTokenScale * sizeof(bfloat16);
    size_t sizeD = lenD * sizeof(bfloat16);
    size_t sizeWorkspace;

    std::vector<int8_t> hostA(lenA);
    std::vector<int8_t> hostB(lenB);
    std::vector<bfloat16> hostScale(lenScale);
    std::vector<bfloat16> hostPerTokenScale(lenPerTokenScale);
    golden::FillRandomData(hostA, -16, 16);
    golden::FillRandomData(hostB, -16, 16);
    golden::FillRandomData(hostScale, 0.0, 1.0);
    golden::FillRandomData(hostPerTokenScale, 0.0, 1.0);

    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceScale{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceScale), sizeScale, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceScale, sizeScale, hostScale.data(), sizeScale, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *devicePerTokenScale{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devicePerTokenScale), sizePerTokenScale,
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(devicePerTokenScale, sizePerTokenScale, hostPerTokenScale.data(), sizePerTokenScale,
        ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceD{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceD), sizeD, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *deviceWorkspace{nullptr};
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::ColumnMajor;
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    layout::VectorLayout layoutScale{n};
    layout::VectorLayout layoutPerTokenScale{m};
    layout::RowMajor layoutD{m, n};

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    using ArchTag = arch::AtlasA2;
    constexpr uint32_t preloadStages = 1;
    constexpr uint32_t l1Stages = 2;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;
    constexpr bool enableUnitFlag = false;
    constexpr bool enableShuffleK = true;
    using DispatchPolicy = gemm::MmadAtlasA2PreloadAsyncWithCallback<
        preloadStages,
        l1Stages, l0AStages, l0BStages, l0CStages,
        enableUnitFlag, enableShuffleK
    >;
    using L0TileShape = MatmulShape<128, 256, 128>;

    using AType = gemm::MatmulType<int8_t, LayoutA>;
    using BType = gemm::MatmulType<int8_t, LayoutB>;
    using CType = gemm::MatmulType<int32_t, layout::RowMajor>;

    using BlockMmad = gemm::block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    constexpr uint32_t ubStages = 2;
    using EpilogueDispatchPolicy = epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
    using ScaleType = gemm::MatmulType<uint16_t, layout::VectorLayout>;
    using PerTokenScaleType = gemm::MatmulType<uint16_t, layout::VectorLayout>;
    using DType = gemm::MatmulType<uint16_t, layout::RowMajor>;

    using RowBroadcastMulType = gemm::MatmulType<float, layout::RowMajor>;
    using BroadcastOneBlkType = gemm::MatmulType<float, layout::RowMajor>;
    using OneBlkColumnBroadcastMulType = gemm::MatmulType<float, layout::RowMajor>;

    using EpilogueTileShape = MatrixShape<32, 256>;
    using TileRowBroadcastMul = epilogue::tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;
    using TileBroadcastOneBlk = epilogue::tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType,
        EpilogueTileShape::ROW>;
    using TileOneBlkColumnBroadcastMul = epilogue::tile::TileOneBlkColumnBroadcastMul<ArchTag,
        OneBlkColumnBroadcastMulType, EpilogueTileShape>;
    using TileCopy = epilogue::tile::TileCopyBf16<ArchTag, CType, ScaleType, PerTokenScaleType, DType>;
    using TileScheduler = epilogue::tile::EpilogueHorizontalTileSwizzle;

    using BlockEpilogue = epilogue::block::BlockEpilogue<EpilogueDispatchPolicy, CType, ScaleType, PerTokenScaleType,
        DType, TileRowBroadcastMul, TileBroadcastOneBlk, TileOneBlkColumnBroadcastMul, TileCopy, TileScheduler>;

    if (options.problemShape.m() > options.problemShape.n()) {
        using BlockScheduler = typename gemm::block::MatmulIdentityBlockSwizzle<3, 0>;

        // kernel level
        using MatmulKernel = gemm::kernel::QuantMatmulMultiStageWorkspace<BlockMmad, BlockEpilogue,
            BlockScheduler, workspaceStages>;

        using MatmulAdapter = gemm::device::MatmulUniversalAdapter<MatmulKernel>;
    
        MatmulKernel::Arguments arguments{
            options.problemShape, aicCoreNum, deviceA, deviceB, deviceScale,
            devicePerTokenScale, deviceD};

        MatmulAdapter matmul_op;
        matmul_op.CanImplement(arguments);
        sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
        if (sizeWorkspace > 0) {
            ACL_CHECK(
                aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
            );
        }
        matmul_op.Initialize(arguments, deviceWorkspace);
        matmul_op(stream, aicCoreNum, fftsAddr);
    } else {
        using BlockScheduler = typename gemm::block::MatmulIdentityBlockSwizzle<3, 1>;

        // kernel level
        using MatmulKernel = gemm::kernel::QuantMatmulMultiStageWorkspace<BlockMmad, BlockEpilogue,
            BlockScheduler, workspaceStages>;

        using MatmulAdapter = gemm::device::MatmulUniversalAdapter<MatmulKernel>;
    
        MatmulKernel::Arguments arguments{
            options.problemShape, aicCoreNum, deviceA, deviceB, deviceScale,
            devicePerTokenScale, deviceD};

        MatmulAdapter matmul_op;
        matmul_op.CanImplement(arguments);
        sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
        if (sizeWorkspace > 0) {
            ACL_CHECK(
                aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)
            );
        }
        matmul_op.Initialize(arguments, deviceWorkspace);
        matmul_op(stream, aicCoreNum, fftsAddr);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<bfloat16> hostD(lenD);
    ACL_CHECK(aclrtMemcpy(hostD.data(), sizeD, deviceD, sizeD, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenD);
    golden::QuantMatmul(
        options.problemShape,
        hostA, layoutA,
        hostB, layoutB,
        hostScale, layoutScale,
        hostPerTokenScale, layoutPerTokenScale,
        hostGolden, layoutD);

    std::vector<uint64_t> errorIndices = golden::CompareData(hostD, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceScale));
    ACL_CHECK(aclrtFree(devicePerTokenScale));
    ACL_CHECK(aclrtFree(deviceD));
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char **argv)
{
    Options options;
    if (options.Parse(argc, argv) == 0) {
        Run(options);
    }
    return 0;
}