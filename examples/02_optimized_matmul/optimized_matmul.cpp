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

#include "actlass/arch/arch.hpp"
#include "actlass/layout/layout.hpp"
#include "actlass/matmul/block/block_mmad.hpp"
#include "actlass/matmul/block/block_swizzle.hpp"
#include "actlass/matmul/dispatch_policy.hpp"
#include "actlass/matmul/kernel/optimized_matmul.hpp"
#include "actlass/matmul/matmul_type.hpp"
#include "actlass/actlass.hpp"

using namespace actlass;

template <
    class LayoutA,
    class LayoutB,
    class LayoutC
>
ACTLASS_GLOBAL
void OptimizedMatmul(
    uint64_t fftsAddr,
    MatmulCoord problemShape,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmB, LayoutB layoutB,
    GM_ADDR gmC, LayoutC layoutC,
    GM_ADDR gmWA, LayoutA layoutWA,
    GM_ADDR gmWB, LayoutB layoutWB
)
{
    using ArchTag = arch::AtlasA2;
    AscendC::SetSyncBaseAddr(fftsAddr);

    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    using DispatchPolicy = matmul::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;
    using AType = matmul::MatmulType<half, LayoutA>;
    using BType = matmul::MatmulType<half, LayoutB>;
    using CType = matmul::MatmulType<half, LayoutC>;

    if constexpr (std::is_same_v<LayoutA, layout::ColumnMajor> && std::is_same_v<LayoutB, layout::ColumnMajor>) {
        using L1TileShape = MatmulShape<256, 128, 256>;
        using L0TileShape = MatmulShape<256, 128, 64>;
        using BlockMmad = matmul::block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        using BlockEpilogue = void;
        if (problemShape.m() > problemShape.n()) {
            using TileScheduler = typename matmul::block::MatmulIdentityBlockSwizzle<3, 0>;
            // kernel level
            using MatmulKernel = matmul::kernel::OptimizedMatmul<BlockMmad, BlockEpilogue, TileScheduler>;
            typename MatmulKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC,
                gmWA, layoutWA, gmWB, layoutWB};
            // call a kernel
            MatmulKernel matmul;
            matmul(params);
        } else {
            using TileScheduler = typename matmul::block::MatmulIdentityBlockSwizzle<3, 1>;
            // kernel level
            using MatmulKernel = matmul::kernel::OptimizedMatmul<BlockMmad, BlockEpilogue, TileScheduler>;
            typename MatmulKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC,
                gmWA, layoutWA, gmWB, layoutWB};

            // call a kernel
            MatmulKernel matmul;
            matmul(params);
        }
    } else {
        using L1TileShape = MatmulShape<128, 256, 256>;
        using L0TileShape = MatmulShape<128, 256, 64>;
        using BlockMmad = matmul::block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        using BlockEpilogue = void;
        if (problemShape.m() > problemShape.n()) {
            using TileScheduler = typename matmul::block::MatmulIdentityBlockSwizzle<3, 0>;
            // kernel level
            using MatmulKernel = matmul::kernel::OptimizedMatmul<BlockMmad, BlockEpilogue, TileScheduler>;
            typename MatmulKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC,
                gmWA, layoutWA, gmWB, layoutWB};
            // call a kernel
            MatmulKernel matmul;
            matmul(params);
        } else {
            using TileScheduler = typename matmul::block::MatmulIdentityBlockSwizzle<3, 1>;
            // kernel level
            using MatmulKernel = matmul::kernel::OptimizedMatmul<BlockMmad, BlockEpilogue, TileScheduler>;
            typename MatmulKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC,
                gmWA, layoutWA, gmWB, layoutWB};
            // call a kernel
            MatmulKernel matmul;
            matmul(params);
        }
    }
}

struct Options {
    const std::string HELPER = "06_optimizd_matmul m n k [device_id]";

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

layout::RowMajor GetWorkspaceLayout(layout::RowMajor layout, uint32_t align)
{
    if (align == 0) {
        return layout;
    }
    return layout::RowMajor(layout.shape(0), layout.shape(1),
        (layout.shape(1) + align - 1) / align * align);
}

layout::ColumnMajor GetWorkspaceLayout(layout::ColumnMajor layout, uint32_t align)
{
    if (align == 0) {
        return layout;
    }
    return layout::ColumnMajor(layout.shape(0), layout.shape(1),
        (layout.shape(0) + align - 1) / align * align);
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

void Run(Options const &options)
{
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

    size_t sizeA = lenA * sizeof(half);
    size_t sizeB = lenB * sizeof(half);
    size_t sizeC = lenC * sizeof(half);

    const uint32_t align = 256;
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::ColumnMajor;
    using LayoutC = layout::RowMajor;
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};
    LayoutA layoutWA = GetWorkspaceLayout(layoutA, align);
    LayoutB layoutWB = GetWorkspaceLayout(layoutB, align);
    size_t sizeWA = GetWorkspaceLen(layoutWA) * sizeof(half);
    size_t sizeWB = GetWorkspaceLen(layoutWB) * sizeof(half);

    std::vector<half> hostA(lenA);
    std::vector<half> hostB(lenB);
    golden::FillRandomData(hostA, -5.0f, 5.0f);
    golden::FillRandomData(hostB, -5.0f, 5.0f);

    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *deviceWA{nullptr};
    // If layoutWA has the same stride with layoutA, no need to padding A
    if (IsSameStride(layoutWA, layoutA)) {
        deviceWA = deviceA;
    } else {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWA), sizeWA, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    uint8_t *deviceWB{nullptr};
    // If layoutWB has the same stride with layoutB, no need to padding B
    if (IsSameStride(layoutWB, layoutB)) {
        deviceWB = deviceB;
    } else {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWB), sizeWB, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    OptimizedMatmul<<<aicCoreNum, nullptr, stream>>>(
        fftsAddr,
        options.problemShape, deviceA, layoutA, deviceB, layoutB, deviceC, layoutC,
        deviceWA, layoutWA, deviceWB, layoutWB);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<half> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenC);
    golden::ComputeMatmul(options.problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);

    std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
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
