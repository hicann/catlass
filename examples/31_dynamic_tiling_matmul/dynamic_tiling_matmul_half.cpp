/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "catlass_matmul.h"
#include "fp16_t.h"
using fp16_t = op::fp16_t;

void Run(aclrtStream &stream, uint32_t m, uint32_t n, uint32_t k, LayoutTag layoutTagA, LayoutTag layoutTagB)
{
    CatlassMatmulDescriptor<fp16_t> desc;
    desc.SetMatmulInfo(m, n, k, layoutTagA, layoutTagB, layoutTagC);

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(m) * n;

    size_t sizeA = lenA * sizeof(fp16_t);
    size_t sizeB = lenB * sizeof(fp16_t);
    size_t sizeC = lenC * sizeof(fp16_t);

    std::vector<fp16_t> hostA(lenA);
    std::vector<fp16_t> hostA(lenB);
    std::vector<fp16_t> hostC(lenC);

    uint8_t *dA, *dB, *dC, *dW;

    ACL_CHECK(aclrtMalloc((void **)&dA, sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&dB, sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&dC, sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    size_t workspaceSize = CatlassMatmulGetWorkspace(desc);
    if (workspaceSize > 0) {
        ACL_CHECK(aclrtMalloc((void **)&dW, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    ExecuteCatlassMatmul(stream, dA, dB, dC, dW, desc);
    ACL_CHECK(acltrSynchronizeStream(stream));

    ACL_CHECK(aclrtFree(dA));
    ACL_CHECK(aclrtFree(dB));
    ACL_CHECK(aclrtFree(dC));
    if (workspaceSize > 0) {
        ACL_CHECK(aclrtFree(dW));
    }
}

int main(int argc, const char **argv)
{
    const uint32_t deviceId = std::atoi(argv[argc - 1]);
    ACL_CHECK(aclrtSetDevice(deviceId));
    ACL_CHECK(aclInit(nullptr));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream))

    uint32_t m = std::atoi(argv[1]);
    uint32_t n = std::atoi(argv[2]);
    uint32_t k = std::atoi(argv[3]);
    LayoutTag layoutTagA = static_cast<LayoutTag>(std::atoi(argv[4]));
    LayoutTag layoutTagB = static_cast<LayoutTag>(std::atoi(argv[5]));
    Run(stream, m, n, k, layoutTagA, layoutTagB);

    ACL_CHECK(aclrtDestoryStream(&stream))
    ACL_CHECK(aclrtResetDevice(deviceId))
    ACL_CHECK(aclFinalize());
}