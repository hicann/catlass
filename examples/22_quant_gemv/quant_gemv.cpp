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
#include "AscendCT/gemv/block/block_gemv.hpp"
#include "AscendCT/gemv/block/block_swizzle.hpp"

#include "AscendCT/gemv/kernel/kernel_quant_gemv.hpp"
#include "AscendCT/gemv/tile/tile_copy.hpp"

#include "AscendCT/gemm/dispatch_policy.hpp"
#include "AscendCT/gemm/matmul_type.hpp"

#include "AscendCT/epilogue/block/block_epilogue.hpp"
#include "AscendCT/epilogue/dispatch_policy.hpp"
#include "AscendCT/epilogue/tile/tile_broadcast_mul.hpp"
#include "AscendCT/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "AscendCT/epilogue/tile/tile_elemwise_add.hpp"
#include "AscendCT/epilogue/tile/tile_elemwise_muls.hpp"
#include "AscendCT/epilogue/tile/tile_copy.hpp"

#include "AscendCT/status.hpp"
#include "AscendCT/gemv/device/gemv_universal_adapter.hpp"
#include "AscendCT/layout/layout.hpp"

using namespace AscendCT;
using bfloat16 = op::bfloat16;

using ScalarType = float;


typedef struct Options {
    const std::string HELPER = "22_quant_gemv m n [device_id]";

    GemvCoord problemShape{128, 128};
    int32_t deviceId{7};

    Options() = default;

    int Parse(int argc, const char** argv) {
        enum ArgsIndex {
            M_INDEX = 1,
            N_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };
        if (argc > ARGS_MAX || argc < N_INDEX) {
            std::cerr << HELPER << std::endl;
            return -1;
        }
        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);
        if (argc == ARGS_MAX) {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        return 0;
    }
} Options;


template <class Adapter>
void RunAdapter(Adapter gemv_op, typename Adapter::Arguments args, aclrtStream stream,
    uint32_t aicCoreNum, uint64_t fftsAddr)
{
    size_t sizeWorkspace = gemv_op.GetWorkspaceSize(args);
    uint8_t *deviceWorkspace = nullptr;
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    gemv_op.Initialize(args, deviceWorkspace);
    gemv_op(stream, aicCoreNum, fftsAddr);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }
}

void Run(Options options) {
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();

    size_t lenA = static_cast<size_t>(m) * n;
    size_t lenX = static_cast<size_t>(n) * 1;
    size_t lenZ = static_cast<size_t>(m) * 1;

    // dequant vector
    size_t lenScale = static_cast<size_t>(m);
    size_t lenBias = static_cast<size_t>(m);
    size_t lenPerTokenScale = 1;

    size_t sizeA = lenA * sizeof(int8_t);
    size_t sizeX = lenX * sizeof(int8_t);
    size_t sizeZ = lenZ * sizeof(bfloat16);
    size_t sizeScale = lenScale * sizeof(bfloat16);
    size_t sizeBias = lenBias * sizeof(bfloat16);
    size_t sizePerTokenScale = lenPerTokenScale * sizeof(float);
    size_t sizeWorkspace = lenZ * sizeof(int32_t);

    using LayoutX = layout::RowMajor;
    using LayoutA = layout::RowMajor;
    using LayoutZ = layout::VectorLayout;
    using LayoutY = layout::RowMajor;

    layout::VectorLayout layoutScale{m};
    layout::VectorLayout layoutBias{m};


    LayoutX layoutX{1, n};
    LayoutA layoutA{m, n};
    LayoutZ layoutZ{m};

    LayoutY layoutY_r{m, 1};
    LayoutX layoutX_r{n, 1};

    std::vector<ScalarType> hostPerTokenScale(lenPerTokenScale);
    golden::FillRandomData(hostPerTokenScale, 1.0, 1.0);


    std::vector<int8_t> hostA(lenA);
    std::vector<int8_t> hostX(lenX);
    std::vector<bfloat16> hostScale(lenScale);
    std::vector<bfloat16> hostBias(lenBias);
    golden::FillRandomData(hostA, -8, 8);
    golden::FillRandomData(hostX, -8, 8);
    golden::FillRandomData(hostScale, -1.0, 1.0);
    golden::FillRandomData(hostBias, -1.0, 1.0);

    std::vector<float> hostY(lenZ);
    golden::FillRandomData(hostY, -1.0f, 1.0f);


    uint8_t* deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceX{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceX), sizeX, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceX, sizeX, hostX.data(), sizeX, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceScale{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceScale), sizeScale, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceScale, sizeScale, hostScale.data(), sizeScale, ACL_MEMCPY_HOST_TO_DEVICE));


    uint8_t *deviceBias{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceBias), sizeBias,
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceBias, sizeBias, hostBias.data(), sizeBias,
        ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceZ{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceZ), sizeZ, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t* deviceWorkspace{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    
    using ArchTag = arch::AtlasA2;
    using LayoutTemp = layout::RowMajor;

    // Block level, define BlockGemv
    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    using DispatchPolicy = gemm::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;
    using L1TileShape = GemvShape<32, 1024>;
    using L0TileShape = GemvShape<32, 512>;
    using AType = gemm::MatmulType<int8_t, LayoutA>;
    using XType = gemm::MatmulType<int8_t, LayoutX>;
    using TempType = gemm::MatmulType<int32_t, LayoutTemp>;
    using BiasType = void;
    using TileCopy = gemv::tile::TileCopyGemvAic<typename DispatchPolicy::ArchTag, AType, XType, TempType, BiasType>;
    using TileMmad = gemm::tile::TileMmad<typename DispatchPolicy::ArchTag, XType, AType, BiasType>;

    using BlockGemv = gemv::block::BlockGemv<DispatchPolicy, L1TileShape, L0TileShape, AType, XType, TempType, BiasType, TileCopy, TileMmad>;

    // Block level, define BlockEpilogue
    constexpr uint32_t ubStages = 2;
    using EpilogueDispatchPolicy = epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
    using ScaleType = gemm::MatmulType<bfloat16_t, layout::VectorLayout>;
    using biasType = gemm::MatmulType<bfloat16_t, layout::VectorLayout>;

    using ZType = gemm::MatmulType<bfloat16_t, LayoutZ>;
    using AXType = gemm::MatmulType<int32_t, LayoutZ>;

    using ComputeType = gemm::MatmulType<float, LayoutZ>;
    constexpr uint32_t computeLength = 4096;

    using TileElemWiseAddGemv = epilogue::tile::TileElemWiseAdd<ArchTag, ComputeType, computeLength>;
    using TileElemWiseMulsGemv = epilogue::tile::TileElemWiseMuls<ArchTag, ComputeType, computeLength>;
    using RowBroadcastMulType = gemm::MatmulType<float, layout::VectorLayout>;

    using EpilogueTileShape = MatrixShape<1, 32>;
    using TileRowBroadcastMul = epilogue::tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;

    using EpilogueTileCopy = epilogue::tile::TileCopyBf16<ArchTag, AXType, ScaleType, biasType, ZType>;

    using BlockEpilogue = epilogue::block::BlockEpilogue<EpilogueDispatchPolicy, AXType, ScaleType, biasType, ZType, TileRowBroadcastMul, TileElemWiseAddGemv, TileElemWiseMulsGemv, EpilogueTileCopy>;

    using TileScheduler = typename gemv::block::GemvIdentityBlockSwizzle<3, 0>;

    // kernle levels
    using GemvKernel = gemv::kernel::QuantGemv<BlockGemv, BlockEpilogue, TileScheduler>;

    // Prepare params
    typename BlockEpilogue::Params epilogueParams{deviceScale, layoutScale, hostPerTokenScale[0], deviceBias, layoutBias, deviceZ, layoutZ};
    using GemvAdapter = gemv::device::GemvUniversalAdapter<GemvKernel>;
    GemvKernel::Arguments arguments{options.problemShape, deviceScale, layoutScale, hostPerTokenScale[0], deviceBias, layoutBias, sizeof(int32_t), deviceX, deviceA, deviceZ, epilogueParams};
    GemvAdapter gemv_op;
    RunAdapter(gemv_op, arguments, stream, aicCoreNum, fftsAddr);

    std::vector<bfloat16> hostRes(lenZ);
    ACL_CHECK(aclrtMemcpy(hostRes.data(), sizeZ, deviceZ, sizeZ, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenZ);
    golden::QuantGemv(
        options.problemShape, 
        hostA, layoutA, 
        hostX, layoutX_r, 
        hostScale, 
        hostPerTokenScale[0],
        hostBias,
        hostGolden, layoutY_r);
    std::vector<uint64_t> errorIndices = golden::CompareData(hostRes, hostGolden, m);

    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceX));
    ACL_CHECK(aclrtFree(deviceZ));
    ACL_CHECK(aclrtFree(deviceWorkspace));
    ACL_CHECK(aclrtFree(deviceScale));
    ACL_CHECK(aclrtFree(deviceBias));

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char** argv) {
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}