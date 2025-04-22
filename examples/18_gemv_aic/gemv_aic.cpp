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

#include "act/act.hpp"
#include "act/arch/arch.hpp"
#include "act/gemv/block/block_gemv.hpp"

#include "act/gemv/kernel/kernel_gemv_aic.hpp"
#include "act/gemv/tile/tile_copy.hpp"

#include "act/gemm/dispatch_policy.hpp"
#include "act/gemm/gemm_type.hpp"

#include "act/epilogue/block/block_epilogue.hpp"
#include "act/epilogue/dispatch_policy.hpp"
#include "act/epilogue/tile/tile_copy.hpp"
#include "act/epilogue/tile/tile_elemwise_add.hpp"
#include "act/epilogue/tile/tile_elemwise_muls.hpp"

#include "act/layout/layout.hpp"
#include "act/gemv/device/device_gemv.hpp"
#include "act/status.hpp"

using ScalarType = float;

typedef struct Options {
    const std::string HELPER = "20_gemv_aic m n [device_id]";

    Act::GemvCoord problemShape{128, 128};
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

    using LayoutX = Act::layout::RowMajor;
    using LayoutA = Act::layout::RowMajor;
    using LayoutZ = Act::layout::VectorLayout;
    using LayoutY = Act::layout::RowMajor;

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

    using ArchTag = Act::Arch::AtlasA2;

    // Block level, define BlockGemv
    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    using DispatchPolicy = Act::Gemm::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;

    using LayoutX = Act::layout::RowMajor;
    using LayoutA = Act::layout::RowMajor;
    using LayoutTemp = Act::layout::RowMajor;
    using LayoutZ = Act::layout::VectorLayout;

    using L1TileShape = Act::GemvShape<32, 512>;
    using L0TileShape = Act::GemvShape<32, 256>;
    using AType = Act::Gemm::GemmType<float, LayoutA>;
    using XType = Act::Gemm::GemmType<float, LayoutX>;
    using TempType = Act::Gemm::GemmType<float, LayoutTemp>;
    using BiasType = void;
    using TileCopy = Act::Gemv::Tile::TileCopyGemvAic<typename DispatchPolicy::ArchTag, AType, XType, TempType, BiasType>;
    using TileMmad = Act::Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, XType, AType, BiasType>;

    using BlockGemv = Gemv::Block::BlockGemv<DispatchPolicy, L1TileShape, L0TileShape, AType, XType, TempType, BiasType, TileCopy, TileMmad>;

    // Block level, define BlockEpilogue
    using EpilogueBlockDispatchPolicy = Act::Epilogue::EpilogueAtlasA2ElemWiseOneSource;
    using YType = Act::Gemm::GemmType<float, LayoutZ>;
    using ZType = Act::Gemm::GemmType<float, LayoutZ>;
    using AXType = Act::Gemm::GemmType<float, LayoutZ>;

    using ComputeType = AXType;
    constexpr uint32_t computeLength = 8192;

    using TileElemWiseAddGemv = Act::Epilogue::Tile::TileElemWiseAdd<ArchTag, ComputeType, computeLength>;
    using TileElemWiseMulsGemv = Act::Epilogue::Tile::TileElemWiseMuls<ArchTag, ComputeType, computeLength>;

    using EpilogueTileCopy = Act::Epilogue::Tile::TileCopy<ArchTag, YType, AXType, ZType>;

    using BlockEpilogue = Act::Epilogue::Block::BlockEpilogue<EpilogueBlockDispatchPolicy, AXType, YType, ZType, TileElemWiseAddGemv, TileElemWiseMulsGemv, EpilogueTileCopy>;

    // kernle levels
    using GemvKernel = Act::Gemv::Kernel::GemvEpilogue<BlockGemv, BlockEpilogue>;

    // TODO:  use adapter to activate the kernel
    using GemvAdapter = Act::Gemv::Device::DeviceGemv<GemvKernel>;
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