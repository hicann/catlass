/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTEST_CATLASS_KERNEL_H
#define OPTEST_CATLASS_KERNEL_H

#include <cstdint>
#include <vector>

#include <acl/acl.h>

namespace CatlassKernel {
using ElementGroupList = int64_t;
struct KernelInfo {
    enum class GMMSplit : uint32_t {
        SPLIT_M = 0,
        SPLIT_K = 1,
        SPLIT_N = 2
    };
    aclDataType inputDataType = aclDataType::ACL_FLOAT16;
    aclDataType outputDataType = aclDataType::ACL_FLOAT16;
    aclDataType scaleDataType = aclDataType::ACL_FLOAT16;  // Scale tensor data type (for quant_matmul)
    aclDataType perTokenScaleDataType = aclDataType::ACL_FLOAT16;  // PerTokenScale tensor data type (for quant_matmul)
    uint32_t g = 1;
    uint32_t b = 1;
    uint32_t m = 1;
    uint32_t n = 1;
    uint32_t k = 1;
    uint32_t M = 1;
    uint32_t K = 1;
    bool transA = false;
    bool transB = false;
    bool formatA = false;  // 如果为true，表示矩阵A使用分块布局（zN/nZ格式）；如果为false，使用常规Row/Col格式
    bool formatB = false;  // 如果为true，表示矩阵B使用分块布局（zN/nZ格式）；如果为false，使用常规Row/Col格式
    std::vector<ElementGroupList> groupList;
    GMMSplit split = GMMSplit::SPLIT_M;
    std::vector<uint8_t *> inputAddr;
    std::vector<uint8_t *> outputAddr;
};

struct ConvKernelInfo {
    aclDataType inputDataType = aclDataType::ACL_FLOAT16;
    aclDataType biasDataType = aclDataType::ACL_FLOAT;
    aclDataType outputDataType = aclDataType::ACL_FLOAT16;

    std::vector<uint32_t> fmapRelated;
    std::vector<uint32_t> filterRelated;
    
    std::vector<uint32_t> strideList;
    std::vector<uint32_t> padList;
    std::vector<uint32_t> dilationList;

    std::vector<uint8_t *> inputAddr;
    std::vector<uint8_t *> outputAddr;
};

struct FAKernelInfo {
    uint32_t qNtokens{0};
    uint32_t batch{0};
    uint32_t qSeqlen{0};
    uint32_t kvSeqlen{0};
    uint32_t numHeads{0};
    uint32_t kvHeads{0};
    uint32_t embeddingSize{0};
    uint32_t isVariedLen{0};
    uint32_t maskType{0};
    uint32_t blockSize{128};
    aclDataType dataType = ACL_FLOAT16;

    std::vector<uint8_t *> inputAddr;
    std::vector<uint8_t *> outputAddr;
};

struct W4A4QuantMatmulKernelInfo {
    aclDataType inputDataType = aclDataType::ACL_INT4;
    aclDataType scaleDataType = aclDataType::ACL_UINT64;
    aclDataType perTokenScaleDataType = aclDataType::ACL_FLOAT;
    aclDataType outputDataType = aclDataType::ACL_BF16;
    
    uint32_t m = 1;
    uint32_t n = 1;
    uint32_t k = 1;
    bool transA = false;
    bool transB = false;
    bool formatA = false;
    bool formatB = false;
    std::vector<uint8_t *> inputAddr;
    std::vector<uint8_t *> outputAddr;
};

/** GEMM (alpha*A*B + beta*C), aligned with 3rdparty 15_gemm: float, RowMajor, Epilogue alpha/beta, C/X in-place. */
struct GemmKernelInfo {
    uint32_t m = 1;
    uint32_t n = 1;
    uint32_t k = 1;
    float alpha = 1.0f;
    float beta = 0.0f;
    std::vector<uint8_t *> inputAddr;   // [0]=A, [1]=B, [2]=C (X)
    std::vector<uint8_t *> outputAddr;  // [0]=D (may be same as C for in-place)
};

void BasicMatmul(const uint32_t blockNum, aclrtStream stream, const KernelInfo &kernelInfo);
void BasicMatmulTLA(const uint32_t aicCoreNum, aclrtStream stream, const KernelInfo &kernelInfo);
void BigMatmulTLA(const uint32_t aicCoreNum, aclrtStream stream, const KernelInfo &kernelInfo);
void MatmulAdd(const uint32_t blockNum, aclrtStream stream, const KernelInfo &kernelInfo);
void PaddingMatmul(const uint32_t blockNum, aclrtStream stream, const KernelInfo &kernelInfo);
void GroupedMatmul(const uint32_t aicCoreNum, aclrtStream stream, const KernelInfo &kernelInfo);
void GroupedMatmulMix(const uint32_t blockNum, aclrtStream stream, const KernelInfo &kernelInfo);
void OptimizedMatmul(const uint32_t blockNum, aclrtStream stream, const KernelInfo &kernelInfo);
void OptimizedMatmulTLA(const uint32_t blockNum, aclrtStream stream, const KernelInfo &kernelInfo);
void GroupedMatmulSliceMPerTokenDequant(const uint32_t blockNum, aclrtStream stream, const KernelInfo &kernelInfo);
void GroupedMatmulSliceKPerTokenDequant(const uint32_t aicCoreNum, aclrtStream stream, const KernelInfo &kernelInfo);
void QuantMatmul(const uint32_t blockNum, aclrtStream stream, const KernelInfo &kernelInfo);
void ConvBias(uint32_t blockNum, aclrtStream stream, ConvKernelInfo kernelInfo);
void FlashAttentionInfer(const uint32_t blockNum, aclrtStream stream, FAKernelInfo kernelInfo);
void W4A4QuantMatmul(const uint32_t blockNum, aclrtStream stream, const W4A4QuantMatmulKernelInfo &kernelInfo);
void SplitkMatmul(const uint32_t aicCoreNum, aclrtStream stream, const KernelInfo &kernelInfo);
void StreamkMatmul(const uint32_t aicCoreNum, aclrtStream stream, const KernelInfo &kernelInfo);
void PaddingSplitkMatmul(const uint32_t aicCoreNum, aclrtStream stream, const KernelInfo &kernelInfo);
void BasicMatmulPreloadZN(const uint32_t aicCoreNum, aclrtStream stream, const KernelInfo &kernelInfo);
void MatmulRelu(const uint32_t blockNum, aclrtStream stream, const KernelInfo &kernelInfo);
void MatmulGelu(const uint32_t blockNum, aclrtStream stream, const KernelInfo &kernelInfo);
void MatmulSilu(const uint32_t blockNum, aclrtStream stream, const KernelInfo &kernelInfo);
void MatmulFullLoadA(const uint32_t aicCoreNum, aclrtStream stream, const KernelInfo &kernelInfo);
void Gemm(const uint32_t aicCoreNum, aclrtStream stream, const GemmKernelInfo &kernelInfo);
void SmallMatmul(const uint32_t aicCoreNum, aclrtStream stream, const KernelInfo &kernelInfo);
void BatchedMatmul(const uint32_t aicCoreNum, aclrtStream stream, const KernelInfo &kernelInfo);
} // namespace CatlassKernel

#endif // SHARED_LIB_CATLASS_KERNEL_H