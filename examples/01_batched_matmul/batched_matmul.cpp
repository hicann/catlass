/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <acl/acl.h>
#include "helper.hpp"
#include "golden.hpp"
#include "fp16_t.h"

#include "AscendCT/AscendCT.hpp"
#include "AscendCT/arch/arch.hpp"
#include "AscendCT/gemm/block/block_mmad.hpp"
#include "AscendCT/gemm/block/block_swizzle.hpp"
#include "AscendCT/gemm/dispatch_policy.hpp"
#include "AscendCT/gemm/kernel/batched_matmul.hpp"
#include "AscendCT/gemm/GemmType"
#include "AscendCT/layout/layout.hpp"

using namespace AscendCT;
using fp16_t = op::fp16_t;

constexpr float DATA_UPPER_BOUND = 5;
constexpr float DATA_LOWER_BOUND = -5;

template <class LayoutA, class LayoutB, class LayoutC>
ASCENDCT_GLOBAL
void BatchedMatmul(uint32_t batchCount, GemmCoord problemShape,
                   GM_ADDR gmA, LayoutA layoutA,
                   GM_ADDR gmB, LayoutB layoutB,
                   GM_ADDR gmC, LayoutC layoutC)
{
    using ArchTag = arch::AtlasA2;
    using DispatchPolicy = gemm::MmadAtlasA2Pingpong<true>;
    using L1TileShape = GemmShape<128, 256, 256>;
    using L0TileShape = GemmShape<128, 256, 64>;

    using AType = gemm::GemmType<half, LayoutA>;
    using BType = gemm::GemmType<half, LayoutB>;
    using CType = gemm::GemmType<half, LayoutC>;

    using BlockMmad = gemm::block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockEpilogue = void;

    if (problemShape.m() > problemShape.n()) {
        // Swizzle offset is 3 and direction is 0.
        using BlockScheduler = typename gemm::block::GemmIdentityBlockSwizzle<3, 0>;

        // kernel level
        using MatmulKernel = gemm::kernel::BatchedMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;

        int64_t strideA = problemShape.m() * problemShape.k();
        int64_t strideB = problemShape.k() * problemShape.n();
        int64_t strideC = problemShape.m() * problemShape.n();
        typename MatmulKernel::Params params{
            batchCount, problemShape,
            gmA, layoutA, strideA,
            gmB, layoutB, strideB,
            gmC, layoutC, strideC
        };

        // call a kernel
        MatmulKernel matmul;
        matmul(params);
    } else {
        // Swizzle offset is 3 and direction is 1.
        using BlockScheduler = typename gemm::block::GemmIdentityBlockSwizzle<3, 1>;

        // kernel level
        using MatmulKernel = gemm::kernel::BatchedMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;

        int64_t strideA = problemShape.m() * problemShape.k();
        int64_t strideB = problemShape.k() * problemShape.n();
        int64_t strideC = problemShape.m() * problemShape.n();
        typename MatmulKernel::Params params{
            batchCount, problemShape,
            gmA, layoutA, strideA,
            gmB, layoutB, strideB,
            gmC, layoutC, strideC
        };

        // call a kernel
        MatmulKernel matmul;
        matmul(params);
    }
}

struct Options {
    const std::string HELPER = "01_batched_matmul b m n k [device_id]";

    uint32_t batchCount{1};
    uint32_t m{128};
    uint32_t n{128};
    uint32_t k{128};
    uint32_t deviceId{0};

    Options() = default;

    int Parse(int argc, const char **argv)
    {
        enum ArgsIndex {
            B_INDEX = 1,
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

        batchCount = std::atoi(argv[B_INDEX]);
        m = std::atoi(argv[M_INDEX]);
        n = std::atoi(argv[N_INDEX]);
        k = std::atoi(argv[K_INDEX]);
        if (argc == ARGS_MAX) {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        return 0;
    }
};


void resourceInit(uint32_t deviceId, aclrtStream *stream)
{
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    ACL_CHECK(aclrtCreateStream(stream));
}

void resourceDestroy(uint32_t deviceId, aclrtStream stream)
{
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());
}

void freeTensor(uint8_t *deviceA, uint8_t *deviceB, uint8_t *deviceC)
{
    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
}

void Run(Options const &options)
{
    aclrtStream stream{nullptr};

    resourceInit(options.deviceId, &stream);

    uint32_t batchCount = options.batchCount;
    uint32_t m = options.m;
    uint32_t n = options.n;
    uint32_t k = options.k;
    GemmCoord problemShape{m, n, k};

    size_t lenA = static_cast<size_t>(m) * k * batchCount;
    size_t lenB = static_cast<size_t>(k) * n * batchCount;
    size_t lenC = static_cast<size_t>(m) * n * batchCount;

    size_t sizeA = lenA * sizeof(fp16_t);
    size_t sizeB = lenB * sizeof(fp16_t);
    size_t sizeC = lenC * sizeof(fp16_t);

    // allocate memory of A and copy to device side
    std::vector<fp16_t> hostA(lenA, 1.0);
    golden::FillRandomData<fp16_t>(hostA, DATA_LOWER_BOUND, DATA_UPPER_BOUND);
    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    // allocate memory of B and copy to device side
    std::vector<fp16_t> hostB(lenB, 1.0);
    golden::FillRandomData<fp16_t>(hostB, DATA_LOWER_BOUND, DATA_UPPER_BOUND);
    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    // allocate memory of C
    std::vector<fp16_t> hostC(lenC);
    uint8_t *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    using LayoutA = layout::RowMajor; // can be RowMajor or ColumnMajor
    using LayoutB = layout::RowMajor; // can be RowMajor or ColumnMajor
    using LayoutC = layout::RowMajor; // must be RowMajor
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    BatchedMatmul<<<aicCoreNum, nullptr, stream>>>(batchCount, problemShape,
        deviceA, layoutA, deviceB, layoutB, deviceC, layoutC);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    // comparison of precision with matmul computed on cpu
    std::vector<float> hostGolden(lenC);
    golden::ComputeBatchedMatmul(batchCount, problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);

    std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    freeTensor(deviceA, deviceB, deviceC);
    resourceDestroy(options.deviceId, stream);
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
