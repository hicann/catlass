/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <acl/acl.h>
#include <runtime/rt_ffts.h>

#include "AscendCTKernel.h"
#include "BasicMatmul.h"
#include "GroupedMatmulSliceK.h"
#include "GroupedMatmulSliceM.h"
#include "OptimizedMatmul.h"

namespace AscendCTKernel {

void BasicMatmul(uint32_t blockNum, aclrtStream stream, KernelInfo kernelInfo)
{
    Act::GemmCoord problemShape{kernelInfo.m, kernelInfo.n, kernelInfo.k};
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    LayoutA layoutA{kernelInfo.m, kernelInfo.k};
    LayoutB layoutB{kernelInfo.k, kernelInfo.n};
    LayoutC layoutC{kernelInfo.m, kernelInfo.n};
    if (kernelInfo.inputDataType == ACL_FLOAT16 && kernelInfo.outputDataType == ACL_FLOAT16) {
        basic_matmul<LayoutA, LayoutB, LayoutC, ACL_FLOAT16, ACL_FLOAT16><<<blockNum, nullptr, stream>>>(
            problemShape, kernelInfo.inputAddr.at(0), layoutA, kernelInfo.inputAddr.at(1), layoutB,
            kernelInfo.outputAddr.at(0), layoutC);
    } else if (kernelInfo.inputDataType == ACL_BF16 && kernelInfo.outputDataType == ACL_BF16) {
        basic_matmul<LayoutA, LayoutB, LayoutC, ACL_BF16, ACL_BF16><<<blockNum, nullptr, stream>>>(
            problemShape, kernelInfo.inputAddr.at(0), layoutA, kernelInfo.inputAddr.at(1), layoutB,
            kernelInfo.outputAddr.at(0), layoutC);
    }
}

void GroupedMatmul(uint32_t blockNum, aclrtStream stream, KernelInfo kernelInfo)
{
    const uint32_t problemCount = kernelInfo.groupList.size();

    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;

    std::vector<int32_t> groupList = kernelInfo.groupList;
    uint32_t m = kernelInfo.m;
    uint32_t n = kernelInfo.n;
    uint32_t k = kernelInfo.k;

    GemmCoord problemShape{m, n, k};
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};

    // layout and shape malloc and copy
    uint8_t *groupListDevice{nullptr};
    typename std::decay<decltype(groupList)>::type::value_type elemGroupList = 0;
    size_t sizeGroupListDevice = groupList.size() * sizeof(decltype(elemGroupList));
    aclrtMalloc(reinterpret_cast<void **>(&groupListDevice), sizeGroupListDevice, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(groupListDevice, sizeGroupListDevice, groupList.data(), sizeGroupListDevice, ACL_MEMCPY_HOST_TO_DEVICE);

    // execution
    if (kernelInfo.split == KernelInfo::GMMSplit::SPLIT_M) {
        grouped_matmul_slice_m<LayoutA, LayoutB, LayoutC><<<blockNum, nullptr, stream>>>(
            problemShape, problemCount, groupListDevice, kernelInfo.inputAddr.at(0), layoutA,
            kernelInfo.inputAddr.at(1), layoutB, kernelInfo.outputAddr.at(0), layoutC);
    } else if (kernelInfo.split == KernelInfo::GMMSplit::SPLIT_K) {
        grouped_matmul_slice_k<LayoutA, LayoutB, LayoutC><<<blockNum, nullptr, stream>>>(
            problemShape, problemCount, groupListDevice, kernelInfo.inputAddr.at(0), layoutA,
            kernelInfo.inputAddr.at(1), layoutB, kernelInfo.outputAddr.at(0), layoutC);
    }
    aclrtFree(groupListDevice);
}

void OptimizedMatmul(uint32_t blockNum, aclrtStream stream, KernelInfo kernelInfo)
{
    uint32_t m = kernelInfo.m;
    uint32_t n = kernelInfo.n;
    uint32_t k = kernelInfo.k;

    GemmCoord problemShape{kernelInfo.m, kernelInfo.n, kernelInfo.k};

    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};

    const uint32_t align = 256;
    bool isNeedPaddingA = IsNeedPadding(layoutA, align);
    bool isNeedPaddingB = IsNeedPadding(layoutB, align);

    // if LayoutA and LayoutB is both ColumnMajor,
    // L1TileShape using GemmShape<256, 128, 256> can achieve better performance.
    using L1TileShape =
        std::conditional_t<std::is_same_v<LayoutA, layout::ColumnMajor> && std::is_same_v<LayoutB, layout::ColumnMajor>,
                           GemmShape<256, 128, 256>, GemmShape<128, 256, 256>>;

    uint8_t *deviceA = kernelInfo.inputAddr.at(0);
    uint8_t *deviceB = kernelInfo.inputAddr.at(1);
    uint8_t *deviceC = kernelInfo.outputAddr.at(0);

    size_t sizeWA = GetWorkspaceLen(layoutA, L1TileShape::M, L1TileShape::K) * sizeof(half);
    size_t sizeWB = GetWorkspaceLen(layoutB, L1TileShape::K, L1TileShape::N) * sizeof(half);

    uint8_t *deviceWA{nullptr};
    if (isNeedPaddingA) {
        aclrtMalloc(reinterpret_cast<void **>(&deviceWA), sizeWA, ACL_MEM_MALLOC_HUGE_FIRST);
    } else {
        // no need to padding A
        deviceWA = deviceA;
    }

    uint8_t *deviceWB{nullptr};
    // If layoutWB has the same stride with layoutB, no need to padding B
    if (isNeedPaddingB) {
        aclrtMalloc(reinterpret_cast<void **>(&deviceWB), sizeWB, ACL_MEM_MALLOC_HUGE_FIRST);
    } else {
        // no need to padding B
        deviceWB = deviceB;
    }

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);

    optimized_matmul<<<blockNum, nullptr, stream>>>(fftsAddr, problemShape, deviceA, layoutA, deviceB, layoutB, deviceC,
                                                    layoutC, deviceWA, deviceWB);

    if (isNeedPaddingA) {
        aclrtFree(deviceWA);
    }
    if (isNeedPaddingB) {
        aclrtFree(deviceWB);
    }
}
} // namespace AscendCTKernel