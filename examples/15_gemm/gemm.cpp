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
#include "act/gemm/block/block_mmad.hpp"
#include "act/gemm/kernel/gemm.hpp"
#include "act/gemm/gemm_type.hpp"
#include "act/layout/layout.hpp"
#include "act/gemm_coord.hpp"
#include "act/matrix_coord.hpp"
#include "act/gemm/dispatch_policy.hpp"
#include "act/epilogue/dispatch_policy.hpp"
#include "act/epilogue/tile/tile_copy.hpp"
#include "act/epilogue/tile/tile_elemwise_add.hpp"
#include "act/epilogue/tile/tile_elemwise_muls.hpp"
#include "act/epilogue/tile/tile_cast.hpp"
#include "act/epilogue/block/block_epilogue.hpp"

#include "act/status.hpp"
#include "act/gemm/device/device_gemm.hpp"

using namespace Act;

using ScalarType = float;

typedef struct Options{
    const std::string HELPER = "17_gemm m n k [device_id]";

    GemmCoord problemShape{128, 128, 128};
    int32_t deviceId{0};

    Options() = default;

    int Parse(int argc, const char **argv){
        enum ArgsIndex{
            M_INDEX = 1,
            N_INDEX,
            K_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };
        if(argc > ARGS_MAX || argc <= K_INDEX){
            std::cerr << HELPER << std::endl;
            return -1;
        }
        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);
        problemShape.k() = std::atoi(argv[K_INDEX]);
        if(argc == ARGS_MAX){
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        return 0;
    }
}Options;

layout::RowMajor GetWorkspaceLayout(layout::RowMajor layout, uint32_t align)
{
    if (align == 0) {
        return layout;
    }
    return layout::RowMajor(layout.shape(0), layout.shape(1),
        RoundUp(layout.shape(1), align));
}

layout::ColumnMajor GetWorkspaceLayout(layout::ColumnMajor layout, uint32_t align)
{
    if (align == 0) {
        return layout;
    }
    return layout::ColumnMajor(layout.shape(0), layout.shape(1),
        RoundUp(layout.shape(0), align));
}

size_t GetWorkspaceLen(layout::RowMajor layout)
{
    return layout.shape(0) * layout.stride(0);
}

size_t GetWorkspaceLen(layout::ColumnMajor layout)
{
    return layout.shape(1) * layout.stride(1);
}

bool IsSameStride(layout::RowMajor layout1, layout::RowMajor layout2)
{
    return layout1.stride(0) == layout2.stride(0);
}

bool IsSameStride(layout::ColumnMajor layout1, layout::ColumnMajor layout2)
{
    return layout1.stride(1) == layout2.stride(1);
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

void Run(Options options){
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
    size_t lenX = lenC; 
    size_t scalarLen = 1;
    
    size_t sizeA = lenA * sizeof(float);
    size_t sizeB = lenB * sizeof(float);
    size_t sizeC = lenC * sizeof(float);
    size_t sizeX = lenX * sizeof(float);

    const uint32_t align = 128;
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};
    LayoutA layoutWA = GetWorkspaceLayout(layoutA, align);
    LayoutB layoutWB = GetWorkspaceLayout(layoutB, align);
    size_t sizeWA = GetWorkspaceLen(layoutWA) * sizeof(float);
    size_t sizeWB = GetWorkspaceLen(layoutWB) * sizeof(float);

    size_t scalarSize = scalarLen * sizeof(ScalarType);
    std::vector<ScalarType> hostAlpha(scalarLen);
    std::vector<ScalarType> hostBeta(scalarLen);
    golden::FillRandomData(hostAlpha,  -1.0f, 1.0f);
    golden::FillRandomData(hostBeta,  -1.0f, 1.0f);
    std::vector<float> hostA(lenA);
    std::vector<float> hostB(lenB);
    std::vector<float> hostC(lenC);
    golden::FillRandomData(hostA,  -1.0f, 1.0f);
    golden::FillRandomData(hostB,  -1.0f, 1.0f);
    golden::FillRandomData(hostC,  -1.0f, 1.0f);
    float *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    float *deviceWA{nullptr};
    if (IsSameStride(layoutWA, layoutA)) {
        deviceWA = deviceA;
    } else {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWA), sizeWA, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    float *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    float *deviceWB{nullptr};
    if (IsSameStride(layoutWB, layoutB)) {
        deviceWB = deviceB;
    } else {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWB), sizeWB, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    float *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceC, sizeC, hostC.data(), sizeC, ACL_MEMCPY_HOST_TO_DEVICE));
    
    float *gmWorkspace{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&gmWorkspace), sizeX, ACL_MEM_MALLOC_HUGE_FIRST));

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    
    using ArchTag = Arch::AtlasA2;
    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    using GemmBlockDispatchPolicy = Act::Gemm::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;
    using EpilogueBlockDispatchPolicy = Act::Epilogue::EpilogueAtlasA2ElemWiseOneSource;
    using AType = Gemm::GemmType<float, LayoutA>;
    using BType = Gemm::GemmType<float, LayoutB>;
    using CType = Gemm::GemmType<float, LayoutC>;
    using XType = Gemm::GemmType<float, LayoutC>;
    using DType = CType;
    using ComputeType = XType;
    using L1TileShape = GemmShape<128, 128, 128>;
    using L0TileShape = GemmShape<128, 128, 64>;
    using TileShapeCast = MatrixShape<L1TileShape::M / 2, L1TileShape::N>;
    using GemmBlock = Gemm::Block::BlockGemm<GemmBlockDispatchPolicy, L1TileShape, L0TileShape, AType, BType, XType>;
    constexpr uint32_t computeLength = L1TileShape::MN / 2;
    using TileElemWiseAddGemm = Epilogue::Tile::TileElemWiseAdd<ArchTag, ComputeType, computeLength>;
    using TileElemWiseMulsGemm = Epilogue::Tile::TileElemWiseMuls<ArchTag, ComputeType, computeLength>;
    using TileElemWiseCastC = Epilogue::Tile::TileCast<ArchTag, ComputeType, CType, TileShapeCast>;
    using TileElemWiseCastD = Epilogue::Tile::TileCast<ArchTag, DType, ComputeType, TileShapeCast>;
    using EpilogueTileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, XType, DType>;
    using EpilogueBlock = Epilogue::Block::BlockEpilogue<EpilogueBlockDispatchPolicy, CType, XType, DType, TileElemWiseAddGemm, TileElemWiseMulsGemm, TileElemWiseCastC, TileElemWiseCastD, EpilogueTileCopy>;
    using GemmKernel = Gemm::Kernel::KernelGemm<GemmBlock, EpilogueBlock>;
    typename EpilogueBlock::Params epilogueParams{hostAlpha[0], hostBeta[0], (uint8_t*)deviceC, layoutC, (uint8_t*)deviceC, layoutC};
    typename GemmKernel::Arguments arguments{options.problemShape, align, (uint8_t*)deviceA, (uint8_t*)deviceB, (uint8_t*)gmWorkspace, (uint8_t*)deviceWA, (uint8_t*)deviceWB, epilogueParams};
    using GemmAdapter = Gemm::Device::DeviceGemm<GemmKernel>;
    GemmAdapter gemm_op;
    gemm_op.CanImplement(arguments);
    RunAdapter(gemm_op, arguments, stream, aicCoreNum, fftsAddr);

    std::vector<float> hostRes(lenC);
    ACL_CHECK(aclrtMemcpy(hostRes.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<float> hostGolden(lenC);
    golden::ComputeGemm(options.problemShape, hostAlpha[0], hostBeta[0], hostA, layoutA, hostB, layoutB, hostC, layoutC, hostGolden, layoutC);

    std::vector<uint64_t> errorIndices = golden::CompareData(hostRes, hostGolden, m * n);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    if (!IsSameStride(layoutWA, layoutA)) {
        ACL_CHECK(aclrtFree(deviceWA));
    }
    if (!IsSameStride(layoutWB, layoutB)) {
        ACL_CHECK(aclrtFree(deviceWB));
    }
    ACL_CHECK(aclrtFree(gmWorkspace));

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char **argv){
    Options options;
    if(options.Parse(argc, argv) != 0){
        return -1;
    }
    Run(options);
    return 0;
}