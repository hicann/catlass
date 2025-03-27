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

#include "helper.hpp"
#include "golden.hpp"

#include "AscendCT/AscendCT.hpp"
#include "AscendCT/arch/arch.hpp"
#include "AscendCT/gemv/block/block_gemv.hpp"
#include "AscendCT/gemv/block/block_swizzle.hpp"

#include "AscendCT/gemv/kernel/kernel_gemv_aic.hpp"
#include "AscendCT/gemv/tile/tile_copy.hpp"

#include "AscendCT/gemm/dispatch_policy.hpp"
#include "AscendCT/gemm/matmul_type.hpp"

#include "AscendCT/epilogue/block/block_epilogue.hpp"
#include "AscendCT/epilogue/dispatch_policy.hpp"
#include "AscendCT/epilogue/tile/tile_copy.hpp"
#include "AscendCT/epilogue/tile/tile_elemwise_add.hpp"
#include "AscendCT/epilogue/tile/tile_elemwise_muls.hpp"

#include "AscendCT/layout/layout.hpp"
#include "AscendCT/gemv/device/gemv_universal_adapter.hpp"
#include "AscendCT/status.hpp"


using namespace AscendCT;

using ScalarType = float;



typedef struct Options {
    const std::string HELPER = "20_gemv_aic m n [device_id]";

    GemvCoord problemShape{128, 128};
    int32_t deviceId{1};

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
    size_t lenY = static_cast<size_t>(m) * 1;
    size_t lenZ = static_cast<size_t>(m) * 1;
    size_t scalarLen = 1;

    size_t sizeA = lenA * sizeof(float);
    size_t sizeX = lenX * sizeof(float);
    size_t sizeZ = lenZ * sizeof(float);
    size_t sizeY = lenY * sizeof(float);
    size_t sizeWorkspace;

    using LayoutX = layout::RowMajor;
    using LayoutA = layout::RowMajor;
    using LayoutZ = layout::VectorLayout;
    using LayoutY = layout::RowMajor;

    LayoutX layoutX{1, n};
    LayoutA layoutA{m, n};
    LayoutZ layoutZ{m};

    LayoutY layoutY_r{m, 1};
    LayoutX layoutX_r{n, 1};

    size_t scalarSize = scalarLen * sizeof(ScalarType);
    std::vector<ScalarType> hostAlpha(scalarLen);
    std::vector<ScalarType> hostBeta(scalarLen);
    golden::FillRandomData(hostAlpha, -1.0f, 1.0f);
    golden::FillRandomData(hostBeta, -1.0f, 1.0f);

    std::vector<float> hostA(lenA);
    std::vector<float> hostX(lenX);
    std::vector<float> hostY(lenY);
    golden::FillRandomData(hostA, -1.0f, 1.0f);
    golden::FillRandomData(hostX, -1.0f, 1.0f);
    golden::FillRandomData(hostY, -1.0f, 1.0f);
    uint8_t* deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceX{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceX), sizeX, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceX, sizeX, hostX.data(), sizeX, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceZ{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceZ), sizeZ, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceZ, sizeZ, hostY.data(), sizeZ, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* deviceWorkspace{nullptr};

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    using ArchTag = arch::AtlasA2;

    // Block level, define BlockGemv
    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    using DispatchPolicy = gemm::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;

    using LayoutX = layout::RowMajor;
    using LayoutA = layout::RowMajor;
    using LayoutTemp = layout::RowMajor;
    using LayoutZ = layout::VectorLayout;

    using L1TileShape = GemvShape<32, 512>;
    using L0TileShape = GemvShape<32, 256>;
    using AType = gemm::MatmulType<float, LayoutA>;
    using XType = gemm::MatmulType<float, LayoutX>;
    using TempType = gemm::MatmulType<float, LayoutTemp>;
    using BiasType = void;
    using TileCopy = gemv::tile::TileCopyGemvAic<typename DispatchPolicy::ArchTag, AType, XType, TempType, BiasType>;
    using TileMmad = gemm::tile::TileMmad<typename DispatchPolicy::ArchTag, XType, AType, BiasType>;

    using BlockGemv = gemv::block::BlockGemv<DispatchPolicy, L1TileShape, L0TileShape, AType, XType, TempType, BiasType, TileCopy, TileMmad>;

    // Block level, define BlockEpilogue
    using EpilogueBlockDispatchPolicy = epilogue::EpilogueAtlasA2ElemWiseOneSource;
    using YType = gemm::MatmulType<float, LayoutZ>;
    using ZType = gemm::MatmulType<float, LayoutZ>;
    using AXType = gemm::MatmulType<float, LayoutZ>;

    using ComputeType = AXType;
    constexpr uint32_t computeLength = 8192;

    using TileElemWiseAddGemv = epilogue::tile::TileElemWiseAdd<ArchTag, ComputeType, computeLength>;
    using TileElemWiseMulsGemv = epilogue::tile::TileElemWiseMuls<ArchTag, ComputeType, computeLength>;

    using EpilogueTileCopy = epilogue::tile::TileCopy<ArchTag, YType, AXType, ZType>;

    using BlockEpilogue = epilogue::block::BlockEpilogue<EpilogueBlockDispatchPolicy, AXType, YType, ZType, TileElemWiseAddGemv, TileElemWiseMulsGemv, EpilogueTileCopy>;

    using TileScheduler = typename gemv::block::GemvIdentityBlockSwizzle<3, 0>;

    // kernle levels
    using GemvKernel = gemv::kernel::GemvEpilogue<BlockGemv, BlockEpilogue, TileScheduler>;

    // TODO:  use adapter to activate the kernel
    using GemvAdapter = gemv::device::GemvUniversalAdapter<GemvKernel>;
    GemvKernel::Arguments arguments{options.problemShape, hostAlpha[0], hostBeta[0], sizeof(float), deviceX, deviceA, deviceZ};
    GemvAdapter gemv_op;
    RunAdapter(gemv_op, arguments, stream, aicCoreNum, fftsAddr);

    std::vector<float> hostRes(lenZ);
    ACL_CHECK(aclrtMemcpy(hostRes.data(), sizeZ, deviceZ, sizeZ, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenZ);

    golden::ComputeGemvAic(options.problemShape, hostAlpha[0], hostBeta[0], hostA, layoutA, hostX, layoutX_r, hostY, layoutY_r, hostGolden, layoutY_r);
    std::vector<uint64_t> errorIndices = golden::CompareData(hostRes, hostGolden, m);

    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceX));
    ACL_CHECK(aclrtFree(deviceZ));

    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }

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