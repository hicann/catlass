/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef OPTEST_CATLASS_KERNEL_PREBUILT_H
#define OPTEST_CATLASS_KERNEL_PREBUILT_H

#include <cstdint>
#include <vector>

#include <acl/acl.h>

namespace CatlassKernel {

/**
 * @brief Runtime parameters shared by prebuilt numbered examples.
 */
struct PrebuiltParams {
    std::vector<uint8_t*> inputAddr;  ///< Input buffer addresses.
    std::vector<uint8_t*> outputAddr; ///< Output buffer addresses.
};

/**
 * @brief Runtime parameters for convolution examples.
 */
struct ConvParams : public PrebuiltParams {
    aclDataType inputDataType = aclDataType::ACL_FLOAT16;  ///< Input tensor element type.
    aclDataType biasDataType = aclDataType::ACL_FLOAT;     ///< Bias tensor element type.
    aclDataType outputDataType = aclDataType::ACL_FLOAT16; ///< Output tensor element type.
    std::vector<uint32_t> fmapRelated;                     ///< Feature-map dimensions.
    std::vector<uint32_t> filterRelated;                   ///< Filter dimensions.
    std::vector<uint32_t> strideList;                      ///< Convolution strides.
    std::vector<uint32_t> padList;                         ///< Padding values.
    std::vector<uint32_t> dilationList;                    ///< Dilation values.
};

/**
 * @brief Runtime parameters for flash-attention style examples.
 */
struct FlashAttentionParams : public PrebuiltParams {
    uint32_t qNtokens = 0;                  ///< Total Q tokens for variable-length input.
    uint32_t batch = 0;                     ///< Batch size.
    uint32_t qSeqlen = 0;                   ///< Q sequence length.
    uint32_t kvSeqlen = 0;                  ///< KV sequence length.
    uint32_t numHeads = 0;                  ///< Number of Q heads.
    uint32_t kvHeads = 0;                   ///< Number of KV heads.
    uint32_t embeddingSize = 0;             ///< Per-head embedding dimension.
    uint32_t isVariedLen = 0;               ///< Whether variable-length input is used.
    uint32_t maskType = 0;                  ///< Mask mode.
    uint32_t blockSize = 128;               ///< Tile block size.
    aclDataType dataType = ACL_FLOAT16;     ///< Input/output element type.
};

/**
 * @brief Runtime parameters for MLA examples.
 */
struct MlaParams : public FlashAttentionParams {
    uint32_t qRopeHeadDim = 0;  ///< Q rope head dimension.
    uint32_t kvRopeHeadDim = 0; ///< KV rope head dimension.
};

/**
 * @brief Reserved prebuilt interface for example 19_mla.
 */
void Mla(const uint32_t blockNum, aclrtStream stream, const MlaParams& params);

/**
 * @brief Reserved prebuilt interface for example 23_flash_attention_infer.
 */
void FlashAttentionInfer(const uint32_t blockNum, aclrtStream stream, const FlashAttentionParams& params);

/**
 * @brief Reserved prebuilt interface for example 24_conv_bias.
 */
void ConvBias(const uint32_t blockNum, aclrtStream stream, const ConvParams& params);

/**
 * @brief Reserved prebuilt interface for example 33_basic_conv2d.
 */
void BasicConv2d(const uint32_t blockNum, aclrtStream stream, const ConvParams& params);

/**
 * @brief Reserved prebuilt interface for example 40_flash_attention_infer_tla.
 */
void FlashAttentionInferTLA(const uint32_t blockNum, aclrtStream stream, const FlashAttentionParams& params);

/**
 * @brief Reserved prebuilt interface for example 49_ascend950_flash_attention_infer.
 */
void Ascend950FlashAttentionInfer(const uint32_t blockNum, aclrtStream stream, const FlashAttentionParams& params);

/**
 * @brief Reserved prebuilt interface for example 56_ascend950_basic_conv2d_tla.
 */
void Ascend950BasicConv2dTLA(const uint32_t blockNum, aclrtStream stream, const ConvParams& params);

} // namespace CatlassKernel

#endif // OPTEST_CATLASS_KERNEL_PREBUILT_H
