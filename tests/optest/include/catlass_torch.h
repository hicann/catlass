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

#ifndef OPTEST_CATLASS_TORCH_H
#define OPTEST_CATLASS_TORCH_H

#include <torch/extension.h>

/**
 * @brief PyTorch extension entry for CATLASS basic matrix multiplication.
 *
 * This declaration mirrors the schema registered in ``src/catlass_torch.cpp``.
 * The implementation is provided by ``MatmulLike<BasicMatmul>::Run``.
 *
 * @param mat1 Left input matrix.
 * @param mat2 Right input matrix.
 * @param outDType Output scalar dtype.
 * @param transA Whether to read mat1 as transposed.
 * @param transB Whether to read mat2 as transposed.
 * @param formatA Whether mat1 uses CATLASS NZ block format.
 * @param formatB Whether mat2 uses CATLASS NZ block format.
 * @return Output matrix tensor.
 */
at::Tensor basic_matmul(
    const at::Tensor& mat1, const at::Tensor& mat2, const c10::ScalarType& outDType, const bool transA,
    const bool transB, const bool formatA, const bool formatB);

/**
 * @brief PyTorch extension entry for CATLASS quantized optimized matmul (TLA).
 *
 * Source: example 42_quant_optimized_matmul_tla.
 * Implementation provided by ``QuantMatmulLike<QuantOptimizedMatmulTLA>::Run``.
 *
 * @param mat1 Left input matrix (int8).
 * @param mat2 Right input matrix (int8).
 * @param scale Per-column scale tensor (float).
 * @param perTokenScale Per-row scale tensor (float).
 * @param outDType Output scalar dtype.
 * @param transA Whether to read mat1 as transposed.
 * @param transB Whether to read mat2 as transposed.
 * @param formatA Whether mat1 uses CATLASS NZ block format.
 * @param formatB Whether mat2 uses CATLASS NZ block format.
 * @return Output matrix tensor.
 */
at::Tensor quant_optimized_matmul_tla(
    const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& scale, const at::Tensor& perTokenScale,
    const c10::ScalarType& outDType, const bool transA, const bool transB, const bool formatA, const bool formatB);

/**
 * @brief PyTorch extension entry for CATLASS Ascend950 MX FP8 matmul ASWT (TLA).
 *
 * Source: example 53_ascend950_fp8_mx_matmul_aswt.
 */
at::Tensor ascend950_fp8_mx_matmul_aswt(
    const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& mx_scale_a, const at::Tensor& mx_scale_b,
    const bool transA, const bool transB);

/**
 * @brief PyTorch extension entry for CATLASS Ascend950 MX FP4 matmul ASWT (TLA).
 *
 * Source: example 54_ascend950_fp4_mx_matmul_aswt.
 */
at::Tensor ascend950_fp4_mx_matmul_aswt(
    const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& mx_scale_a, const at::Tensor& mx_scale_b,
    const bool transA, const bool transB);

#endif
