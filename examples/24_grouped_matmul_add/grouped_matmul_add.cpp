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
#include <cstdlib>

#include "helper.hpp"
#include "golden.hpp"
#include "fp16_t.h"

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/grouped_matmul_epilogue.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"

#include "catlass/status.hpp"
#include "catlass/gemm/device/device_gemm.hpp"

using namespace Catlass;
using fp16_t = op::fp16_t;

struct Options {
    const std::string HELPER = "24_grouped_matmul_add group_count m n k [device_id]";

    uint32_t groupCount{1};
    GemmCoord problemShape{128, 128, 128};
    int32_t deviceId{0};

    Options() = default;

    int Parse(int argc, const char **argv)
    {
        enum ArgsIndex {
            GROUP_COUNT_INDEX = 1,
            M_INDEX,
            N_INDEX,
            K_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > ARGS_MAX || argc <= K_INDEX) {
            std::cerr << HELPER << std::endl;
            return -1;
        }

        groupCount = std::atoi(argv[GROUP_COUNT_INDEX]);
        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);
        problemShape.k() = std::atoi(argv[K_INDEX]);
        if (argc == ARGS_MAX) {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        return 0;
    }
};

void Run(Options const &options)
{
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t problemCount = options.groupCount;
    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenD = static_cast<size_t>(m) * n * problemCount;

    size_t sizeA = lenA * sizeof(fp16_t);
    size_t sizeB = lenB * sizeof(fp16_t);
    size_t sizeD = lenD * sizeof(fp16_t);

    using LayoutA = layout::ColumnMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;

    const fp16_t fp16_tLower = -5.0;
    const fp16_t fp16_tUpper = 5.0;
    std::vector<fp16_t> hostA(lenA);
    std::vector<fp16_t> hostB(lenB);
    std::vector<fp16_t> hostX(lenD);
    golden::FillRandomData<fp16_t>(hostA, fp16_tLower, fp16_tUpper);
    golden::FillRandomData<fp16_t>(hostB, fp16_tLower, fp16_tUpper);
    golden::FillRandomData<fp16_t>(hostX, fp16_tLower, fp16_tUpper);
    auto groupList = golden::GenerateGroupList(k, problemCount);

    std::vector<GemmCoord> problemShapeList(problemCount);
    std::vector<LayoutA> layoutAList(problemCount);
    std::vector<LayoutB> layoutBList(problemCount);
    std::vector<LayoutC> layoutCList(problemCount);
    for (uint32_t i = 0; i < problemCount; ++i) {
        uint32_t currentK = (i == 0) ? groupList[0] : (groupList[i] - groupList[i - 1]);
        problemShapeList[i] = GemmCoord{m, n, currentK};
        layoutAList[i] = LayoutA{m, currentK};
        layoutBList[i] = LayoutB{currentK, n};
        layoutCList[i] = LayoutC{m, n};
    }

    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceD{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceD), sizeD, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceD, sizeD, hostX.data(), sizeD, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *problemShapeListDevice{nullptr};
    size_t sizeProblemShapeList = problemShapeList.size() * sizeof(GemmCoord);
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&problemShapeListDevice), sizeProblemShapeList,
        ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(problemShapeListDevice, sizeProblemShapeList,
        problemShapeList.data(), sizeProblemShapeList, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *layoutAListDevice{nullptr};
    size_t sizeLayoutAList = layoutAList.size() * sizeof(LayoutA);
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&layoutAListDevice), sizeLayoutAList, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(layoutAListDevice, sizeLayoutAList,
        layoutAList.data(), sizeLayoutAList, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *layoutBListDevice{nullptr};
    size_t sizeLayoutBList = layoutBList.size() * sizeof(LayoutB);
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&layoutBListDevice), sizeLayoutBList, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(layoutBListDevice, sizeLayoutBList,
        layoutBList.data(), sizeLayoutBList, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *layoutCListDevice{nullptr};
    size_t sizeLayoutCList = layoutCList.size() * sizeof(LayoutC);
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&layoutCListDevice), sizeLayoutCList, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(layoutCListDevice, sizeLayoutCList,
        layoutCList.data(), sizeLayoutCList, ACL_MEMCPY_HOST_TO_DEVICE));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    constexpr uint32_t preloadStages = 1;
    constexpr uint32_t l1Stages = 2;
    constexpr uint32_t l0AStages = 4;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;
    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;

    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsync<
        preloadStages,
        l1Stages, l0AStages, l0BStages, l0CStages,
        enableUnitFlag, enableShuffleK
    >;
    using L1TileShape = GemmShape<128, 256, 256>;
    using L0TileShape = GemmShape<128, 256, 64>;
    using AType = Gemm::GemmType<half, LayoutA>;
    using BType = Gemm::GemmType<half, LayoutB>;
    using CType = Gemm::GemmType<half, LayoutC>;
    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2ElemWiseOneSource;
    using XType = CType;
    using DType = CType;
    using ComputeType = CType;
    constexpr uint32_t computeLength = 16384;
    using TileElemWiseEpilogue = Epilogue::Tile::TileElemWiseAdd<ArchTag, ComputeType, computeLength>;
    using EpilogueTileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, XType, DType>;
    using BlockEpilogue = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, CType, XType, DType,
        TileElemWiseEpilogue, EpilogueTileCopy>;

    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
    using MatmulKernel = Gemm::Kernel::GroupedMatmulEpilogue<BlockMmad, BlockEpilogue, BlockScheduler>;
    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

    typename MatmulKernel::Arguments arguments{
        problemCount, problemShapeListDevice,
        deviceA, layoutAListDevice,
        deviceB, layoutBListDevice,
        deviceD, layoutCListDevice
    };

    MatmulAdapter matmul_op;
    // Workspace size equals output size
    uint8_t *deviceWorkspace{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeD, ACL_MEM_MALLOC_HUGE_FIRST));
    matmul_op.Initialize(arguments, deviceWorkspace);
    matmul_op(stream, aicCoreNum);

    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<fp16_t> hostD(lenD);
    ACL_CHECK(aclrtMemcpy(hostD.data(), sizeD, deviceD, sizeD, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenD);
    golden::ComputeGroupedMatmul(problemCount, problemShapeList, hostA, layoutAList,
        hostB, layoutBList, hostGolden, layoutCList);
    for (size_t idx = 0; idx < lenD; ++idx) {
        hostGolden[idx] += static_cast<float>(hostX[idx]);
    }

    std::vector<uint64_t> errorIndices = golden::CompareData(hostD, hostGolden, k, groupList, m * n);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceD));
    ACL_CHECK(aclrtFree(deviceWorkspace));
    ACL_CHECK(aclrtFree(problemShapeListDevice));
    ACL_CHECK(aclrtFree(layoutAListDevice));
    ACL_CHECK(aclrtFree(layoutBListDevice));
    ACL_CHECK(aclrtFree(layoutCListDevice));

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
