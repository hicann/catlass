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



#include <cstring>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include "tiling/platform/platform_ascendc.h"
#include <acl/acl.h>
#include <runtime/rt_ffts.h>
#include "tiling.h"
#include "csv.h"
#include "helper.h"

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/block/block_mmad_w4a8_local.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/w4a8_matmul_local.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/device/device_gemm.hpp"

using namespace Catlass;

const uint32_t ALIGN = 256;
const uint32_t BLOCKS = 20;

struct Options {
    const std::string HELPER = "23_w4a8_matmul [device_id] m n k transA transB ";

    GemmCoord problemShape{128, 128, 128};
    int32_t deviceId{0};
    int32_t transA;
    int32_t transB;

    Options() = default;

    int Parse(int argc, const char **argv)
    {
        enum ArgsIndex {
            DEVICE_ID_INDEX = 1,
            B_INDEX,
            M_INDEX,
            K_INDEX,
            N_INDEX,
            TRANS_A,
            TRANS_B,
            ARGS_MAX
        };

        if (argc != ARGS_MAX) {
            std::cerr << HELPER << std::endl;
            return -1;
        }

        deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);
        problemShape.k() = std::atoi(argv[K_INDEX]);
        transA = std::atoi(argv[trans_A])
        transB = std::atoi(argv[trans_B])
        return 0
    }
};

template<typename T>
uint64_t getSize(uint64_t m, uint64_t n) {
    return m * n * (uint64_t)sizeof(T);
}

void Run(Options const &options)
{
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();
    uint32_t transA = options.transA;
    uint32_t transB = options.transB;
    uint32_t verifyLevel = 1;
    uint32_t loopTimes = 1;

    uint64_t scalar = static_cast<uint64_t>(0x000000003FC00000);

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(m) * n;

    uint64_t sizeA = getSize<__fp16>(m, k);
    uint64_t sizeB = getSize<int8_t>(k, (n + 1) / 2);
    if (transB) {
        sizeB = getSize<int8_t>((k + 1) / 2, n);
    }
    uint64_t sizeC = getSize<__fp16>(m, n);
    uint64_t sizeExpected = getSize<float>(m, n);

    uint64_t sizeWksp = 256 * 512 * BLOCKS * 2 * sizeof(int8_t);
    if (transA && transB) {
        sizeWksp = 128 * 512 * BLOCKS * 2 * sizeof(int8_t);
    }

    uint8_t *hostA, *hostB, *hostC, *hExpected;
    if (verifyLevel) {
        ACL_CHECK(aclrtMallocHost((void **)(&hostA), sizeA));
        ACL_CHECK(aclrtMallocHost((void **)(&hostB), sizeB));
        ACL_CHECK(aclrtMallocHost((void **)(&hostC), sizeC));
        ACL_CHECK(aclrtMallocHost((void **)(&hExpected), sizeExpected));
        ReadFile("/home/c50053055/catlass-master/examples/23_w4a8_matmul/build/data/inputA.dat", hostA, sizeA);
        ReadFile("/home/c50053055/catlass-master/examples/23_w4a8_matmul/build/data/inputB.dat", hostB, sizeB);
        ReadFile("/home/c50053055/catlass-master/examples/23_w4a8_matmul/build/data/expected.dat", hExpected, sizeExpected);
    }

    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};

    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);

    uint8_t *deviceA, *deviceB, *deviceC, *workspace;
    for (uint32_t i = 0; i < (verifyLevel == 0 ? loopTimes : 1); ++i) {

        ACL_CHECK(aclrtMalloc((void **)&deviceA, sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMalloc((void **)&deviceB, sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMalloc((void **)&deviceC, sizeC, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMalloc((void **)&workspace, sizeWksp, ACL_MEM_MALLOC_HUGE_FIRST));

        if (verifyLevel) {
            ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
            ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB, sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
        }
        auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

        using ArchTag = Arch::AtlasA2;

        constexpr bool enableUnitFlag = false;
        constexpr bool enableShuffleK = true;
        using DispatchPolicy = Gemm::MmadAtlasA2W4A8Local<enableUnitFlag, enableShuffleK>;

        // if LayoutA and LayoutB is both ColumnMajor
        // L1TileShape using MatmulShape<256, 128, 256> can achieve better performance.
        using L1TileShape = std::conditional_t<std::is_same_v<LayoutA, layout::ColumnMajor> &&
            std::is_same_v<LayoutB, layout::ColumnMajor>, GemmShape<256, 128, 512>, GemmShape<128, 256, 512>>;
        using L0TileShape = std::conditional_t<std::is_same_v<LayoutA, layout::ColumnMajor> &&
            std::is_same_v<LayoutB, layout::ColumnMajor>, GemmShape<256, 128, 128>, GemmShape<128, 256, 128>>;
        using AType = Gemm::GemmType<int8_t, LayoutA>;
        using BType = Gemm::GemmType<int8_t, LayoutB>;
        using CType = Gemm::GemmType<half, LayoutC>;
        using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

        if (options.problemShape.m() > options.problemShape.n()) {
            using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
            using BlockEpilogue = void;
            // kernel level
            using MatmulKernel = Gemm::Kernel::W4A8MatmulLocal<BlockMmad, BlockEpilogue, BlockScheduler>;
            typename MatmulKernel::Arguments arguments{
                options.problemShape, deviceA, deviceB, deviceC, workspace, scalar};
            // call a kernel
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
            MatmulAdapter matmul_op;
            matmul_op.CanImplement(arguments);
            matmul_op.Initialize(arguments, deviceWorkspace);
            matmul_op(stream, aicCoreNum, fftsAddr);
        } else {
            using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
            using BlockEpilogue = void;
            // kernel level
            using MatmulKernel = Gemm::Kernel::W4A8MatmulLocal<BlockMmad, BlockEpilogue, BlockScheduler>;
            typename MatmulKernel::Arguments arguments{
                options.problemShape, deviceA, deviceB, deviceC, workspace, scalar};
            // call a kernel
            using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
            MatmulAdapter matmul_op;
            matmul_op.CanImplement(arguments);
            matmul_op.Initialize(arguments, workspace);
            matmul_op(stream, aicCoreNum, fftsAddr);
        }

        ACL_CHECK(aclrtSynchronizeStream(stream));

        ACL_CHECK(aclrtFree(deviceA));
        ACL_CHECK(aclrtFree(deviceB));
        if (verifyLevel) {
            ACL_CHECK(aclrtMemcpy(hostC, sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));
        }
        ACL_CHECK(aclrtFree(deviceC));
        ACL_CHECK(aclrtFree(workspace));
    }   

    if (verifyLevel) {
        WirteFile("/home/c50053055/catlass-master/examples/23_w4a8_matmul/build/data/outputC.dat", hostC, sizeC);
        CompareResults<__fp16, float>((__fp16*)hostC, (float*)hExpected, m, k, n);
        ACL_CHECK(aclrtFreeHost(hostA));
        ACL_CHECK(aclrtFreeHost(hostB));
        ACL_CHECK(aclrtFreeHost(hostC));
        ACL_CHECK(aclrtFreeHost(hExpected));
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