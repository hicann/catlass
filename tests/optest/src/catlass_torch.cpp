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

#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "catlass_kernel_jit.h"
#include "catlass_kernel_prebuilt.h"
#include "common/register.h"
#include "template/flash_attention.h"
#include "template/matmul.h"
#include "template/matmul_extra.h"
#include "template/mx_matmul.h"
#include "template/quant_matmul.h"
#include "template/sparse_matmul.h"
#include "template/strided_batched_matmul.h"
#include "template/w4a8_matmul.h"

namespace CatlassKernelWrapper {

using BasicMatmulOp = MatmulLike<CatlassKernel::BasicMatmul>;
static auto& basic_matmul = BasicMatmulOp::Run;
REGISTER_TORCH_FUNC(basic_matmul);

using MatmulAddOp = MatmulExtraLike<CatlassKernel::MatmulAdd, false>;
static auto& matmul_add = MatmulAddOp::Run;
REGISTER_TORCH_FUNC(matmul_add);

using PaddingMatmulOp = MatmulLike<CatlassKernel::PaddingMatmul>;
static auto& padding_matmul = PaddingMatmulOp::Run;
REGISTER_TORCH_FUNC(padding_matmul);

using OptimizedMatmulOp = MatmulLike<CatlassKernel::OptimizedMatmul>;
static auto& optimized_matmul = OptimizedMatmulOp::Run;
REGISTER_TORCH_FUNC(optimized_matmul);

using MatmulBiasOp = MatmulExtraLike<CatlassKernel::MatmulBias, true>;
static auto& matmul_bias = MatmulBiasOp::Run;
REGISTER_TORCH_FUNC(matmul_bias);

using BasicMatmulTLAOp = MatmulLike<CatlassKernel::BasicMatmulTLA>;
static auto& basic_matmul_tla = BasicMatmulTLAOp::Run;
REGISTER_TORCH_FUNC(basic_matmul_tla);

using MatmulReluOp = MatmulLike<CatlassKernel::MatmulRelu>;
static auto& matmul_relu = MatmulReluOp::Run;
REGISTER_TORCH_FUNC(matmul_relu);

using MatmulGeluOp = MatmulLike<CatlassKernel::MatmulGelu>;
static auto& matmul_gelu = MatmulGeluOp::Run;
REGISTER_TORCH_FUNC(matmul_gelu);

using MatmulSiluOp = MatmulLike<CatlassKernel::MatmulSilu>;
static auto& matmul_silu = MatmulSiluOp::Run;
REGISTER_TORCH_FUNC(matmul_silu);

using OptimizedMatmulTLAOp = MatmulLike<CatlassKernel::OptimizedMatmulTLA>;
static auto& optimized_matmul_tla = OptimizedMatmulTLAOp::Run;
REGISTER_TORCH_FUNC(optimized_matmul_tla);

using BasicMatmulPreloadZNOp = MatmulLike<CatlassKernel::BasicMatmulPreloadZN>;
static auto& basic_matmul_preload_zN = BasicMatmulPreloadZNOp::Run;
REGISTER_TORCH_FUNC(basic_matmul_preload_zN);

using MatmulFullLoadAOp = MatmulLike<CatlassKernel::MatmulFullLoadA>;
static auto& matmul_full_loadA = MatmulFullLoadAOp::Run;
REGISTER_TORCH_FUNC(matmul_full_loadA);

using SmallMatmulOp = MatmulLike<CatlassKernel::SmallMatmul>;
static auto& small_matmul = SmallMatmulOp::Run;
REGISTER_TORCH_FUNC(small_matmul);

using SingleCoreSplitkMatmulOp = MatmulLike<CatlassKernel::SingleCoreSplitkMatmul>;
static auto& single_core_splitk_matmul = SingleCoreSplitkMatmulOp::Run;
REGISTER_TORCH_FUNC(single_core_splitk_matmul);

using StreamkMatmulOp = MatmulLike<CatlassKernel::StreamkMatmul>;
static auto& streamk_matmul = StreamkMatmulOp::Run;
REGISTER_TORCH_FUNC(streamk_matmul);

using BigMatmulTLAOp = MatmulLike<CatlassKernel::BigMatmulTLA>;
static auto& big_matmul_tla = BigMatmulTLAOp::Run;
REGISTER_TORCH_FUNC(big_matmul_tla);

using QuantOptimizedMatmulTLAOp = QuantMatmulLike<CatlassKernel::QuantOptimizedMatmulTLA>;
static auto& quant_optimized_matmul_tla = QuantOptimizedMatmulTLAOp::Run;
REGISTER_TORCH_FUNC(quant_optimized_matmul_tla);

using Ascend950BasicMatmulOp = MatmulLike<CatlassKernel::Ascend950BasicMatmul>;
static auto& ascend950_basic_matmul = Ascend950BasicMatmulOp::Run;
REGISTER_TORCH_FUNC(ascend950_basic_matmul);

using QuantMatmulFullLoadATLAOp = QuantMatmulLike<CatlassKernel::QuantMatmulFullLoadATLA>;
static auto& quant_matmul_full_loadA_tla = QuantMatmulFullLoadATLAOp::Run;
REGISTER_TORCH_FUNC(quant_matmul_full_loadA_tla);

using QuantMultiCoreSplitkMatmulTLAOp = QuantMatmulLike<CatlassKernel::QuantMultiCoreSplitkMatmulTLA>;
static auto& quant_multi_core_splitk_matmul_tla = QuantMultiCoreSplitkMatmulTLAOp::Run;
REGISTER_TORCH_FUNC(quant_multi_core_splitk_matmul_tla);

using Ascend950Fp8MxMatmulAswtOp = MxMatmulLike<CatlassKernel::Ascend950Fp8MxMatmulAswt>;
static auto& ascend950_fp8_mx_matmul_aswt = Ascend950Fp8MxMatmulAswtOp::Run;
REGISTER_TORCH_FUNC(ascend950_fp8_mx_matmul_aswt);

using Ascend950Fp4MxMatmulAswtOp = MxMatmulLike<CatlassKernel::Ascend950Fp4MxMatmulAswt>;
static auto& ascend950_fp4_mx_matmul_aswt = Ascend950Fp4MxMatmulAswtOp::Run;
REGISTER_TORCH_FUNC(ascend950_fp4_mx_matmul_aswt);

static auto& flash_attention_infer = FlashAttentionInferOp::Run;
REGISTER_TORCH_FUNC(flash_attention_infer);

using A2Fp8E4M3MatmulOp = MatmulLike<CatlassKernel::A2Fp8E4M3Matmul>;
static auto& a2_fp8_e4m3_matmul = A2Fp8E4M3MatmulOp::Run;
REGISTER_TORCH_FUNC(a2_fp8_e4m3_matmul);

using W8A16MatmulOp = MatmulLike<CatlassKernel::W8A16Matmul>;
static auto& w8a16_matmul = W8A16MatmulOp::Run;
REGISTER_TORCH_FUNC(w8a16_matmul);

using W4A8MatmulOp = W4A8MatmulLike<CatlassKernel::W4A8Matmul>;
static auto& w4a8_matmul = W4A8MatmulOp::Run;
REGISTER_TORCH_FUNC(w4a8_matmul);

using SparseMatmulTLAOp = SparseMatmulLike<CatlassKernel::SparseMatmulTLA>;
static auto& sparse_matmul_tla = SparseMatmulTLAOp::Run;
REGISTER_TORCH_FUNC(sparse_matmul_tla);

using StridedBatchedMatmulTLAOp = StridedBatchedMatmulLike<CatlassKernel::StridedBatchedMatmulTLA>;
static auto& strided_batched_matmul_tla = StridedBatchedMatmulTLAOp::Run;
REGISTER_TORCH_FUNC(strided_batched_matmul_tla);

} // namespace CatlassKernelWrapper
