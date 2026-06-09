/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <acl/acl.h>
#include <iostream>

#include "catlass_kernel_prebuilt.h"
#include "kernel_operator.h"
#include "tiling/mla_tiling.h"

#include "mla_kernel.cpp"
#include "mla_kernel_tp1_spec.cpp"

extern "C" int rtGetC2cCtrlAddr(uint64_t *, uint32_t *);

namespace CatlassKernel {

#define ACL_CHECK(status)                                                                    \
    do {                                                                                     \
        aclError error = status;                                                             \
        if (error != ACL_ERROR_NONE) {                                                       \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << error << std::endl;  \
        }                                                                                    \
    } while (0)

#define RT_CHECK(status)                                                                     \
    do {                                                                                     \
        int32_t error = status;                                                              \
        if (error != 0) {                                                                    \
            std::cerr << __FILE__ << ":" << __LINE__ << " rtError:" << error << std::endl;   \
        }                                                                                    \
    } while (0)

static void CopyHostToDevice(uint8_t *dst, uint8_t *src, uint64_t size)
{
    ACL_CHECK(aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE));
}

template <typename DType>
void MLAImpl(const uint32_t blockNum, aclrtStream stream, MlaParams &params)
{
    int32_t batch = static_cast<int32_t>(params.batch);
    int32_t kvSeqlen = static_cast<int32_t>(params.kvSeqlen);
    int32_t numHeads = static_cast<int32_t>(params.numHeads);
    int32_t kvHeads = static_cast<int32_t>(params.kvHeads);
    int32_t embeddingSize = static_cast<int32_t>(params.embeddingSize);
    int32_t embeddingSizeRope = static_cast<int32_t>(params.qRopeHeadDim);
    int32_t numBlocks = static_cast<int32_t>(params.numBlocks);
    int32_t blockSize = static_cast<int32_t>(params.blockSize);
    int32_t maxKvSeqlen = kvSeqlen;
    int32_t numTokens = static_cast<int32_t>(params.qNtokens);

    uint32_t dTypeKey = (params.dataType == ACL_FLOAT16) ? 0 : 1;
    uint32_t specStraKey = (numHeads == MLATiling::NUM128) ? 1 : 0;
    uint32_t tilingKey = (specStraKey << 2) + dTypeKey;

    uint64_t qoSize = static_cast<uint64_t>(numTokens) * numHeads * embeddingSize * sizeof(DType);
    uint64_t qRopeSize = static_cast<uint64_t>(numTokens) * numHeads * embeddingSizeRope * sizeof(DType);
    uint64_t kvSize = static_cast<uint64_t>(numBlocks) * blockSize * kvHeads * embeddingSize * sizeof(DType);
    uint64_t kRopeSize = static_cast<uint64_t>(numBlocks) * blockSize * kvHeads * embeddingSizeRope * sizeof(DType);
    uint64_t blockTableSize = static_cast<uint64_t>(
        batch * ((maxKvSeqlen + blockSize - 1) / blockSize) * sizeof(int32_t));
    uint32_t tilingSize = (MLATiling::TILING_HEAD_SIZE + batch * MLATiling::TILING_PARA_SIZE) * sizeof(int32_t);
    if (specStraKey > 0) {
        tilingSize = (MLATiling::TILING_HEAD_SIZE + numTokens * MLATiling::TILING_PARA_SIZE) * sizeof(int32_t);
    }

    uint8_t *qDevice;
    uint8_t *qRopeDevice;
    uint8_t *kDevice;
    uint8_t *kRopeDevice;
    uint8_t *blockTableDevice;
    uint64_t outputHostBytes = qoSize;

    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&qDevice), qoSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&qRopeDevice), qRopeSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&kDevice), kvSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&kRopeDevice), kRopeSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&blockTableDevice), blockTableSize, ACL_MEM_MALLOC_HUGE_FIRST));

    CopyHostToDevice(qDevice, params.inputAddr.at(0), qoSize);
    CopyHostToDevice(qRopeDevice, params.inputAddr.at(1), qRopeSize);
    CopyHostToDevice(kDevice, params.inputAddr.at(2), kvSize);
    CopyHostToDevice(kRopeDevice, params.inputAddr.at(3), kRopeSize);
    CopyHostToDevice(blockTableDevice, params.inputAddr.at(4), blockTableSize);

    uint8_t *sDevice;
    ACL_CHECK(aclrtMalloc(
        reinterpret_cast<void **>(&sDevice),
        blockNum * MLATiling::WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * MLATiling::NUM2,
        ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *pDevice;
    ACL_CHECK(aclrtMalloc(
        reinterpret_cast<void **>(&pDevice),
        blockNum * MLATiling::WORKSPACE_BLOCK_SIZE_DB * sizeof(DType) * MLATiling::NUM2,
        ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *oTmpDevice;
    ACL_CHECK(aclrtMalloc(
        reinterpret_cast<void **>(&oTmpDevice),
        blockNum * MLATiling::WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * MLATiling::NUM2,
        ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *globaloDevice;
    ACL_CHECK(aclrtMalloc(
        reinterpret_cast<void **>(&globaloDevice),
        blockNum * MLATiling::WORKSPACE_BLOCK_SIZE_DB * sizeof(float),
        ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *tilingDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&tilingDevice), tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));

    void *tilingHost = nullptr;
    ACL_CHECK(aclrtMallocHost(&tilingHost, tilingSize));
    uint32_t blockDim = blockNum;

    MLATiling::MLAInfo mlaInfo;
    mlaInfo.numTokens = numTokens;
    mlaInfo.numHeads = numHeads;
    mlaInfo.embeddingSize = embeddingSize;
    mlaInfo.embeddingSizeRope = embeddingSizeRope;
    mlaInfo.numBlocks = numBlocks;
    mlaInfo.blockSize = blockSize;
    mlaInfo.maxKvSeqlen = maxKvSeqlen;
    mlaInfo.kvHeads = kvHeads;
    mlaInfo.batch = batch;
    mlaInfo.qSeqLen = const_cast<int32_t *>(params.qSeqHost.data());
    mlaInfo.kvSeqLen = const_cast<int32_t *>(params.kvSeqHost.data());
    MLATiling::GetMLATilingParam(mlaInfo, blockDim, reinterpret_cast<uint32_t *>(tilingHost));

    ACL_CHECK(aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint32_t kvSplitCoreNum = *(reinterpret_cast<uint32_t *>(tilingHost) + MLATiling::TILING_KVCORENUM);
    uint64_t oFdSize = embeddingSize * numHeads * numTokens * kvSplitCoreNum * sizeof(float);
    uint64_t lSize = numTokens * numHeads * kvSplitCoreNum * sizeof(float);

    uint8_t *oCoreTmpDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&oCoreTmpDevice), oFdSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *lDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&lDevice), lSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *oScratchDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&oScratchDevice), qoSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    if (tilingKey == 0) {
        MLAFp16<<<blockDim, nullptr, stream>>>(
            fftsAddr, qDevice, qRopeDevice, kDevice, kRopeDevice, blockTableDevice, oScratchDevice, sDevice, pDevice,
            oTmpDevice, globaloDevice, oCoreTmpDevice, lDevice, tilingDevice);
    } else if (tilingKey == 1) {
        MLABf16<<<blockDim, nullptr, stream>>>(
            fftsAddr, qDevice, qRopeDevice, kDevice, kRopeDevice, blockTableDevice, oScratchDevice, sDevice, pDevice,
            oTmpDevice, globaloDevice, oCoreTmpDevice, lDevice, tilingDevice);
    } else if (tilingKey == 4) {
        MLATp1SpecFp16<<<blockDim, nullptr, stream>>>(
            fftsAddr, qDevice, qRopeDevice, kDevice, kRopeDevice, blockTableDevice, oScratchDevice, sDevice, pDevice,
            oTmpDevice, globaloDevice, oCoreTmpDevice, lDevice, tilingDevice);
    } else if (tilingKey == 5) {
        MLATp1SpecBf16<<<blockDim, nullptr, stream>>>(
            fftsAddr, qDevice, qRopeDevice, kDevice, kRopeDevice, blockTableDevice, oScratchDevice, sDevice, pDevice,
            oTmpDevice, globaloDevice, oCoreTmpDevice, lDevice, tilingDevice);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    params.outputHost.resize(static_cast<size_t>(outputHostBytes));
    ACL_CHECK(aclrtMemcpy(
        params.outputHost.data(), outputHostBytes, oScratchDevice, outputHostBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    aclrtFree(qDevice);
    aclrtFree(qRopeDevice);
    aclrtFree(kDevice);
    aclrtFree(kRopeDevice);
    aclrtFree(blockTableDevice);
    aclrtFree(oScratchDevice);
    aclrtFree(sDevice);
    aclrtFree(pDevice);
    aclrtFree(oTmpDevice);
    aclrtFree(globaloDevice);
    aclrtFree(tilingDevice);
    aclrtFree(oCoreTmpDevice);
    aclrtFree(lDevice);
    aclrtFreeHost(tilingHost);
}

void Mla(uint32_t blockNum, aclrtStream stream, MlaParams &params)
{
    if (params.dataType == ACL_FLOAT16) {
        MLAImpl<half>(blockNum, stream, params);
    } else if (params.dataType == ACL_BF16) {
        MLAImpl<bfloat16_t>(blockNum, stream, params);
    }
}

} // namespace CatlassKernel
