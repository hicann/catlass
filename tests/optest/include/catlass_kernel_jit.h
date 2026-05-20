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

#ifndef OPTEST_CATLASS_KERNEL_JIT_H
#define OPTEST_CATLASS_KERNEL_JIT_H

#include <cstdint>
#include <vector>

#include <acl/acl.h>

namespace CatlassKernel {

/**
 * @brief Compile-time JIT parameters shared by numbered matmul-family examples.
 */
struct MatmulTParams {
    aclDataType elementA = aclDataType::ACL_FLOAT16; ///< Matrix A element type.
    aclDataType elementB = aclDataType::ACL_FLOAT16; ///< Matrix B element type.
    aclDataType elementC = aclDataType::ACL_FLOAT16; ///< Matrix C/output element type.
    bool transA = false;                             ///< Whether A is read as transposed.
    bool transB = false;                             ///< Whether B is read as transposed.
    bool transC = false;                             ///< Reserved output transpose flag.
    bool useNzA = false;                             ///< Whether A uses NZ layout.
    bool useNzB = false;                             ///< Whether B uses NZ layout.
    bool useNzC = false;                             ///< Whether C uses NZ layout.
};

/**
 * @brief Runtime matrix parameters shared by numbered matmul-family examples.
 */
struct MatmulParams {
    uint32_t m = 1;                   ///< M dimension.
    uint32_t n = 1;                   ///< N dimension.
    uint32_t k = 1;                   ///< K dimension.
    uint32_t batch = 1;               ///< Batch dimension for batched variants.
    std::vector<uint8_t*> inputAddr;  ///< Input buffer addresses.
    std::vector<uint8_t*> outputAddr; ///< Output buffer addresses.
};

/**
 * @brief Runtime parameters for grouped matmul examples.
 */
struct GroupedMatmulParams : public MatmulParams {
    enum class SliceMode : uint32_t
    {
        M = 0,
        K = 1,
        N = 2
    };

    uint32_t groupNum = 1;                    ///< Number of groups.
    std::vector<int64_t> groupList; ///< Per-group problem sizes.
    SliceMode sliceMode = SliceMode::M; ///< Grouped matmul slice dimension.
};

/**
 * @brief Compile-time parameters for quantized matmul examples.
 */
struct QuantMatmulTParams : public MatmulTParams {
    aclDataType scaleDataType = aclDataType::ACL_FLOAT16;         ///< Scale tensor element type.
    aclDataType perTokenScaleDataType = aclDataType::ACL_FLOAT16; ///< Per-token scale element type.
};

/**
 * @brief Runtime parameters for GEMM examples with alpha and beta scaling.
 */
struct GemmParams : public MatmulParams {
    float alpha = 1.0f; ///< Alpha scale in D = alpha * A * B + beta * C.
    float beta = 0.0f;  ///< Beta scale in D = alpha * A * B + beta * C.
};

/**
 * @brief Reserved JIT interface for example 00_basic_matmul.
 */
void BasicMatmul(const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 01_batched_matmul.
 */
void BatchedMatmul(
    const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 02_grouped_matmul_slice_m.
 */
void GroupedMatmulSliceM(
    const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const GroupedMatmulParams& params);

/**
 * @brief Reserved JIT interface for example 03_matmul_add.
 */
void MatmulAdd(const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 04_padding_matmul.
 */
void PaddingMatmul(
    const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 05_grouped_matmul_slice_k.
 */
void GroupedMatmulSliceK(
    const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const GroupedMatmulParams& params);

/**
 * @brief Reserved JIT interface for example 06_optimized_matmul.
 */
void OptimizedMatmul(
    const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 07_grouped_matmul_slice_m_per_token_dequant_moe.
 */
void GroupedMatmulSliceMPerTokenDequantMoe(
    const uint32_t blockNum, aclrtStream stream, const QuantMatmulTParams& tParams,
    const GroupedMatmulParams& params);

/**
 * @brief Reserved JIT interface for example 08_grouped_matmul.
 */
void GroupedMatmul(
    const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const GroupedMatmulParams& params);

/**
 * @brief Reserved JIT interface for example 09_splitk_matmul.
 */
void SplitkMatmul(
    const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 10_grouped_matmul_slice_m_per_token_dequant.
 */
void GroupedMatmulSliceMPerTokenDequant(
    const uint32_t blockNum, aclrtStream stream, const QuantMatmulTParams& tParams,
    const GroupedMatmulParams& params);

/**
 * @brief Reserved JIT interface for example 11_grouped_matmul_slice_k_per_token_dequant.
 */
void GroupedMatmulSliceKPerTokenDequant(
    const uint32_t blockNum, aclrtStream stream, const QuantMatmulTParams& tParams,
    const GroupedMatmulParams& params);

/**
 * @brief Reserved JIT interface for example 12_quant_matmul.
 */
void QuantMatmul(
    const uint32_t blockNum, aclrtStream stream, const QuantMatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 13_basic_matmul_tla.
 */
void BasicMatmulTLA(
    const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 14_optimized_matmul_tla.
 */
void OptimizedMatmulTLA(
    const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 15_gemm.
 */
void Gemm(const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const GemmParams& params);

/**
 * @brief Reserved JIT interface for example 16_group_gemm.
 */
void GroupGemm(const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const GroupedMatmulParams& params);

/**
 * @brief Reserved JIT interface for example 17_gemv_aiv.
 */
void GemvAIV(const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 18_gemv_aic.
 */
void GemvAIC(const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 20_matmul_bias.
 */
void MatmulBias(const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 21_basic_matmul_preload_zN.
 */
void BasicMatmulPreloadZN(
    const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 22_padding_splitk_matmul.
 */
void PaddingSplitkMatmul(
    const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 25_matmul_full_loadA.
 */
void MatmulFullLoadA(
    const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 26_matmul_relu.
 */
void MatmulRelu(const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 27_matmul_gelu.
 */
void MatmulGelu(const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 28_matmul_silu.
 */
void MatmulSilu(const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 29_a2_fp8_e4m3_matmul.
 */
void A2Fp8E4M3Matmul(
    const uint32_t blockNum, aclrtStream stream, const QuantMatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 30_w8a16_matmul.
 */
void W8A16Matmul(
    const uint32_t blockNum, aclrtStream stream, const QuantMatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 31_small_matmul.
 */
void SmallMatmul(
    const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 32_w4a8_matmul.
 */
void W4A8Matmul(
    const uint32_t blockNum, aclrtStream stream, const QuantMatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 34_single_core_splitk_matmul.
 */
void SingleCoreSplitkMatmul(
    const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 35_w4a8_grouped_matmul_msd.
 */
void W4A8GroupedMatmulMSD(
    const uint32_t blockNum, aclrtStream stream, const QuantMatmulTParams& tParams, const GroupedMatmulParams& params);

/**
 * @brief Reserved JIT interface for example 36_w4a8_matmul_msd.
 */
void W4A8MatmulMSD(
    const uint32_t blockNum, aclrtStream stream, const QuantMatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 37_streamk_matmul.
 */
void StreamkMatmul(
    const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 38_w4a4_matmul_per_token_per_channel_dequant.
 */
void W4A4MatmulPerTokenPerChannelDequant(
    const uint32_t blockNum, aclrtStream stream, const QuantMatmulTParams& tParams,
    const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 39_big_matmul_tla.
 */
void BigMatmulTLA(
    const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 41_sparse_matmul_tla.
 */
void SparseMatmulTLA(
    const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 42_quant_optimized_matmul_tla.
 */
void QuantOptimizedMatmulTLA(
    const uint32_t blockNum, aclrtStream stream, const QuantMatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 43_ascend950_basic_matmul.
 */
void Ascend950BasicMatmul(
    const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 44_quant_matmul_full_loadA_tla.
 */
void QuantMatmulFullLoadATLA(
    const uint32_t blockNum, aclrtStream stream, const QuantMatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 45_strided_batched_matmul_tla.
 */
void StridedBatchedMatmulTLA(
    const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 46_ascend950_matmul_fixpipe_opti.
 */
void Ascend950MatmulFixpipeOpti(
    const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 47_ascend950_grouped_matmul_slice_m_per_token_dequant.
 */
void Ascend950GroupedMatmulSliceMPerTokenDequant(
    const uint32_t blockNum, aclrtStream stream, const QuantMatmulTParams& tParams,
    const GroupedMatmulParams& params);

/**
 * @brief Reserved JIT interface for example 48_ascend950_grouped_matmul_slice_m_per_tensor_per_channel_dequant.
 */
void Ascend950GroupedMatmulSliceMPerTensorPerChannelDequant(
    const uint32_t blockNum, aclrtStream stream, const QuantMatmulTParams& tParams,
    const GroupedMatmulParams& params);

/**
 * @brief Reserved JIT interface for example 50_ascend950_basic_matmul_gemv.
 */
void Ascend950BasicMatmulGemv(
    const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 51_ascend950_quant_matmul_per_group_per_block_tla.
 */
void Ascend950QuantMatmulPerGroupPerBlockTLA(
    const uint32_t blockNum, aclrtStream stream, const QuantMatmulTParams& tParams,
    const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 52_quant_multi_core_splitk_matmul_tla.
 */
void QuantMultiCoreSplitkMatmulTLA(
    const uint32_t blockNum, aclrtStream stream, const QuantMatmulTParams& tParams,
    const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 53_ascend950_fp8_mx_matmul.
 */
void Ascend950Fp8MxMatmul(
    const uint32_t blockNum, aclrtStream stream, const QuantMatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 54_ascend950_fp4_mx_matmul.
 */
void Ascend950Fp4MxMatmul(
    const uint32_t blockNum, aclrtStream stream, const QuantMatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 57_ascend950_matmul_full_dequant.
 */
void Ascend950MatmulFullDequant(
    const uint32_t blockNum, aclrtStream stream, const QuantMatmulTParams& tParams,
    const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 58_ascend950_fp8_mx_batch_matmul.
 */
void Ascend950Fp8MxBatchMatmul(
    const uint32_t blockNum, aclrtStream stream, const QuantMatmulTParams& tParams,
    const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 59_ascend950_a8w4_mx_matmul.
 */
void Ascend950A8W4MxMatmul(
    const uint32_t blockNum, aclrtStream stream, const QuantMatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 60_ascend950_grouped_matmul_slice_m.
 */
void Ascend950GroupedMatmulSliceM(
    const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const GroupedMatmulParams& params);

/**
 * @brief Reserved JIT interface for example 102_dynamic_optimized_matmul.
 */
void DynamicOptimizedMatmul(
    const uint32_t blockNum, aclrtStream stream, const MatmulTParams& tParams, const MatmulParams& params);

/**
 * @brief Reserved JIT interface for example 103_dynamic_optimized_quant_matmul_per_token_basic.
 */
void DynamicOptimizedQuantMatmulPerTokenBasic(
    const uint32_t blockNum, aclrtStream stream, const QuantMatmulTParams& tParams,
    const MatmulParams& params);

} // namespace CatlassKernel

#endif // OPTEST_CATLASS_KERNEL_JIT_H
