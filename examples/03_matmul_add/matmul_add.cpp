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
#include "fp16_t.h"

#include "act/act.hpp"
#include "act/arch/arch.hpp"
#include "act/epilogue/dispatch_policy.hpp"
#include "act/epilogue/block/block_epilogue.hpp"
#include "act/epilogue/tile/tile_copy.hpp"
#include "act/epilogue/tile/tile_elemwise_add.hpp"
#include "act/gemm/block/block_mmad.hpp"
#include "act/gemm/block/block_swizzle.hpp"
#include "act/gemm/dispatch_policy.hpp"
#include "act/gemm/kernel/matmul_epilogue.hpp"
#include "act/gemm/gemm_type.hpp"
#include "act/layout/layout.hpp"

#include "AscendCT/status.hpp"
#include "AscendCT/gemm/device/matmul_universal_adapter.hpp"

using namespace AscendCT;
using fp16_t = op::fp16_t;



struct Options {
    const std::string HELPER = "03_matmul_add m n k [device_id]";

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

void Run(Options const &options)
{
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    // Compute the length of each matrix and the size of each buffer
    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenD = static_cast<size_t>(m) * n;
    size_t lenX = lenD;

    size_t sizeA = lenA * sizeof(fp16_t);
    size_t sizeB = lenB * sizeof(fp16_t);
    size_t sizeD = lenD * sizeof(fp16_t);

    // Define the layout of each matrix
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutD{m, n};

    // Prepare input data A, B, and X
    std::vector<fp16_t> hostA(lenA);
    std::vector<fp16_t> hostB(lenB);
    std::vector<fp16_t> hostX(lenX);
    golden::FillRandomData<fp16_t>(hostA, -5.0f, 5.0f);
    golden::FillRandomData<fp16_t>(hostB, -5.0f, 5.0f);
    golden::FillRandomData<fp16_t>(hostX, -5.0f, 5.0f);

    // Allocate device memory and copy data from host to device
    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    // The data of X is stored on deviceD to save storage space
    uint8_t *deviceD{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceD), sizeD, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceD, sizeD, hostX.data(), sizeD, ACL_MEMCPY_HOST_TO_DEVICE));

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    // Define ArchTag
    using ArchTag = arch::AtlasA2;

    // Block level, define BlockMmad
    constexpr bool enableUnitFlag = true;
    using MmadDispatchPolicy = gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
    using L1TileShape = MatmulShape<128, 256, 256>;
    using L0TileShape = MatmulShape<128, 256, 64>;
    using AType = gemm::MatmulType<half, LayoutA>;
    using BType = gemm::MatmulType<half, LayoutB>;
    using CType = gemm::MatmulType<half, LayoutC>;
    using BlockMmad = gemm::block::BlockMmad<MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    // Block level, define BlockEpilogue
    using EpilogueDispatchPolicy = epilogue::EpilogueAtlasA2ElemWiseOneSource;
    using XType = CType;
    using DType = CType;
    using ComputeType = CType;
    constexpr uint32_t computeLength = 16384;
    using TileElemWiseEpilogue = epilogue::tile::TileElemWiseAdd<ArchTag, ComputeType, computeLength>;
    using EpilogueTileCopy = epilogue::tile::TileCopy<ArchTag, CType, XType, DType>;
    using BlockEpilogue = epilogue::block::BlockEpilogue<EpilogueDispatchPolicy, CType, XType, DType,
        TileElemWiseEpilogue, EpilogueTileCopy>;

    // Define BlockScheduler
    // Swizzle offset is 3 and direction is 0.
    using BlockScheduler = typename gemm::block::MatmulIdentityBlockSwizzle<3, 0>;

    // Kernel level
    using MatmulKernel = gemm::kernel::MatmulEpilogue<BlockMmad, BlockEpilogue, BlockScheduler>;

    // Prepare params
    typename MatmulKernel::Arguments arguments{
        options.problemShape, sizeof(half), deviceA, deviceB, deviceD};
    using MatmulAdapter = gemm::device::MatmulUniversalAdapter<MatmulKernel>;
    MatmulAdapter matmul_op;
    size_t sizeWorkspace = matmul_op.GetWorkspaceSize(arguments);
    uint8_t *deviceWorkspace{nullptr};
    if (sizeWorkspace > 0) {
        ACL_CHECK(
            aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace,ACL_MEM_MALLOC_HUGE_FIRST));
    }
    matmul_op.Initialize(arguments, deviceWorkspace);
    matmul_op(stream, aicCoreNum, fftsAddr);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }

    // Copy the result from device to host
    std::vector<fp16_t> hostD(lenD);
    ACL_CHECK(aclrtMemcpy(hostD.data(), sizeD, deviceD, sizeD, ACL_MEMCPY_DEVICE_TO_HOST));

    // Compute the golden result
    std::vector<float> hostGolden(lenD);
    golden::ComputeMatmulElemWiseAdd(options.problemShape, hostA, layoutA, hostB, layoutB, hostX, hostGolden, layoutD);

    // Compare the result
    std::vector<uint64_t> errorIndices = golden::CompareData(hostD, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceD));

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
