/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR dataA PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_MATMUL_H
#define CATLASS_MATMUL_H

#include "helper.hpp"
#include "get_tiling.h"

template <class DType>
struct CatlassMatmulDescriptor {
    TilingParams tilingParams;
    uint8_t *dTilingParams;
    TilingKey tilingKey;
    PlatformInfo platformInfo;

    CatlassMatmulDescriptor(uint32_t m, uint32_t n, uint32_t k, size_t strideA, size_t strideB, size_t strideC,
        LayoutTag layoutTagA, LayoutTag layoutTagB, LayoutTag layoutTagC)
    {
        ACL_CHECK(aclrtMalloc((void **)&dTilingParams, sizeof(TilingParams), ACL_MEM_MALLOC_HUGE_FIRST));
    }

    CatlassMatmulDescriptor()
    {
        ACL_CHECK(aclrtMalloc((void **)&dTilingParams, sizeof(TilingParams), ACL_MEM_MALLOC_HUGE_FIRST));
    }

    void SetMatmulInfo(
        uint32_t m, uint32_t n, uint32_t k, LayoutTag layoutTagA, LayoutTag layoutTagB, LayoutTag layoutTagC)
    {
        size_t strideA = k;
        size_t strideB = n;
        size_t strideC = n;
        if (static_cast<LayoutTag>(layoutTagA) == LayoutTag::TagColumnMajor) {
            strideA = m;
        }
        if (static_cast<LayoutTag>(layoutTagB) == LayoutTag::TagColumnMajor) {
            strideB = k;
        }
        if (static_cast<LayoutTag>(layoutTagC) == LayoutTag::TagColumnMajor) {
            strideC = m;
        }
        tilingParams.SetParams(m, n, k, strideA, strideB, strideC, layoutTagA, layoutTagB, layoutTagC);
        GetTiling<DType>(tilingParams, tilingKey, platformInfo);
    }

    void SetMatmulInfo(uint32_t m, uint32_t n, uint32_t k, size_t strideA, size_t strideB, size_t strideC,
        LayoutTag layoutTagA, LayoutTag layoutTagB, LayoutTag layoutTagC)
    {
        tilingParams.SetParams(m, n, k, strideA, strideB, strideC, layoutTagA, layoutTagB, layoutTagC);
        GetTiling<DType>(tilingParams, tilingKey, platformInfo);
    }

    ~CatlassMatmulDescriptor()
    {
        ACL_CHECK(aclrtFree(dTilingParams));
    }
};

template <class DType>
size_t CatlassMatmulGetWorkspace(CatlassMatmulDescriptor<DType> &desc)
{
    return getWorkspaceFuncMap[desc.tilingKey.value](desc.tilingParams);
}

template <class DType>
void ExecuteCatlassMatmul(
    aclrtStream &stream, uint8_t *dA, uint8_t *dB, uint8_t *dC, uint8_t *dW, CatlassMatmulDescriptor<DType> &desc)
{
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    ACL_CHECK(aclrtMemcpy(
        desc.dTilingParams, sizeof(TilingParams), &desc.tilingParams, sizeof(TilingParams), ACL_MEMCPY_HOST_TO_DEVICE));

    launchKernelFuncMap[desc.tilingKey.value](stream, fftsAddr, dA, dB, dC, dW, desc.dTilingParams, desc.tilingParams);
}

template <class DType>
void PrintCatlassMatmulInfo(CatlassMatmulDescriptor<DType> &desc)
{
    PrintTilingParams<DType>(desc.tilingParams);
    std::cout << "Kernel Func Name: " << funcNameMap[desc.tilingKey.value] << std::endl;
    std::cout << "TilingKey: " << std::hex << desc.tilingKey.value << std::endl;
}

#endif  // CATLASS_MATMUL_H