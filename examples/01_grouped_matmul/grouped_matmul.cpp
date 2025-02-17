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

#include "actlass/actlass.hpp"
#include "actlass/arch/arch.hpp"
#include "actlass/matmul/block/block_mmad.hpp"
#include "actlass/matmul/block/block_swizzle.hpp"
#include "actlass/matmul/dispatch_policy.hpp"
#include "actlass/matmul/kernel/grouped_matmul.hpp"
#include "actlass/matmul/matmul_type.hpp"
#include "actlass/layout/layout.hpp"

using namespace actlass;

constexpr float DATA_UPPER_BOUND = 5;
constexpr float DATA_LOWER_BOUND = -5;

template <
    class LayoutA,
    class LayoutB,
    class LayoutC
>
ACTLASS_GLOBAL
void GroupedMatmul(
    uint32_t problemCount, GM_ADDR ptrProblemShape,
    GM_ADDR ptrA, GM_ADDR ptrLayoutA,
    GM_ADDR ptrB, GM_ADDR ptrLayoutB,
    GM_ADDR ptrC, GM_ADDR ptrLayoutC
)
{
    using ArchTag = arch::AtlasA2;
    using DispatchPolicy = matmul::MmadAtlasA2Preload<true>;
    using L1TileShape = MatmulShape<128, 256, 256>;
    using L0TileShape = MatmulShape<128, 256, 64>;

    using AType = matmul::MatmulType<half, LayoutA>;
    using BType = matmul::MatmulType<half, LayoutB>;
    using CType = matmul::MatmulType<half, LayoutC>;

    using BlockMmad = matmul::block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockEpilogue = void;
    using TileScheduler = typename matmul::block::MatmulIdentityBlockSwizzle<3, 0>;

    // kernel level
    using MatmulKernel = matmul::kernel::GroupedMatmul<BlockMmad, BlockEpilogue, TileScheduler>;

    typename MatmulKernel::Params params{
        problemCount, ptrProblemShape, ptrA, ptrLayoutA, ptrB, ptrLayoutB, ptrC, ptrLayoutC
    };

    // call a kernel
    MatmulKernel matmul;
    matmul(params);
}

struct GroupedMatmulArguments {
    uint32_t problemCount;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    std::vector<uint32_t> groupList;

    GroupedMatmulArguments(uint32_t problemCount, uint32_t m, uint32_t n, uint32_t k,
        std::vector<uint32_t>& groupList)
        : problemCount(problemCount), m(m), n(n), k(k), groupList(groupList) {}

    bool CheckArguments()
    {
        if (groupList.size() != problemCount) {
            std::cerr << "Invalid parameter. " << std::endl;
            return false;
        }
        if (groupList.back() != m) {
            std::cerr << "Invalid parameter. " << std::endl;
            return false;
        }
        if (problemCount == 1) {
            return true;
        }
        for (int i = 1; i < groupList.size(); ++i) {
            if (groupList[i] > groupList[i - 1]) {
                continue;
            } else {
                std::cerr << "Invalid parameter. " << std::endl;
                return false;
            }
        }
        return true;
    }
};

void Run(GroupedMatmulArguments& groupedMatmulArguments)
{
    uint32_t problemCount = groupedMatmulArguments.problemCount;
    uint32_t m = groupedMatmulArguments.m;
    uint32_t n = groupedMatmulArguments.n;
    uint32_t k = groupedMatmulArguments.k;
    std::vector<uint32_t>& groupList = groupedMatmulArguments.groupList;

    aclrtStream stream{nullptr};
    uint32_t deviceId = 0;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n * problemCount;
    size_t lenC = static_cast<size_t>(m) * n;

    size_t sizeA = lenA * sizeof(half);
    size_t sizeB = lenB * sizeof(half);
    size_t sizeC = lenC * sizeof(half);

    using LayoutA = layout::RowMajor;
    using LayoutB = layout::ColumnMajor;
    using LayoutC = layout::RowMajor;

    std::vector<half> hostA(lenA);
    std::vector<half> hostB(lenB);
    golden::FillRandomData(hostA, DATA_LOWER_BOUND, DATA_UPPER_BOUND);
    golden::FillRandomData(hostB, DATA_LOWER_BOUND, DATA_UPPER_BOUND);

    // crate grouped matmul problem shapes and layouts
    std::vector<MatmulCoord> problemShapeList(problemCount);
    std::vector<LayoutA> layoutAList(problemCount);
    std::vector<LayoutB> layoutBList(problemCount);
    std::vector<LayoutC> layoutCList(problemCount);
    uint32_t prev = 0;
    for (uint32_t i = 0; i < problemCount; ++i) {
        uint32_t currentM = groupList[i] - prev;
        prev = groupList[i];
        problemShapeList[i] = MatmulCoord{currentM, n, k};
        layoutAList[i] = LayoutA{currentM, k};
        layoutBList[i] = LayoutB{k, n};
        layoutCList[i] = LayoutC{currentM, n};
    }

    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *problemShapeListDevice{nullptr};
    size_t sizeProblemShapeList = problemShapeList.size() * sizeof(MatmulCoord);
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

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    GroupedMatmul<LayoutA, LayoutB, LayoutC><<<aicCoreNum, nullptr, stream>>>(
        problemCount, problemShapeListDevice,
        deviceA, layoutAListDevice,
        deviceB, layoutBListDevice,
        deviceC, layoutCListDevice);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<half> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenC);
    golden::ComputeGroupedMatmul(problemCount, problemShapeList, hostA, layoutAList,
        hostB, layoutBList, hostGolden, layoutCList);

    std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtFree(problemShapeListDevice));
    ACL_CHECK(aclrtFree(layoutAListDevice));
    ACL_CHECK(aclrtFree(layoutBListDevice));
    ACL_CHECK(aclrtFree(layoutCListDevice));

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char **argv)
{
    uint32_t problemCount = 8;
    uint32_t m = 1024;
    uint32_t n = 768;
    uint32_t k = 512;
    std::vector<uint32_t> groupList {128, 256, 512, 515, 568, 579, 678, 1024};
    GroupedMatmulArguments groupedMatmulArguments(problemCount, m, n, k, groupList);
    if (groupedMatmulArguments.CheckArguments()) {
        Run(groupedMatmulArguments);
    }
    return 0;
}