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
#include "common/workspace.h"
#include "template/batched_matmul.h"
#include "template/flash_attention.h"
#include "template/grouped_matmul.h"
#include "template/grouped_quant_matmul.h"
#include "template/matmul.h"
#include "template/matmul_evg.h"
#include "template/matmul_extra.h"
#include "template/mla.h"
#include "template/mx_matmul.h"
#include "template/quant_matmul.h"
#include "template/sparse_matmul.h"
#include "template/strided_batched_matmul.h"
#include "template/w4a8_matmul.h"
#include "template/broadcast_matmul_perblock_quant.h"

// ── Workspace allocator bridge ──
// 通过 dlsym 注入到 g_catlassWorkspaceAlloc，使 JIT 模板分配 NPU tensor
// 而非裸 aclrtMalloc。tensor 保存在静态池中，kernel 执行期间有效。
#include <dlfcn.h>
#include <vector>

namespace {
static std::vector<at::Tensor> g_wsPool;

uint8_t* wsAlloc(size_t size) {
    auto opts = at::TensorOptions().dtype(torch::kInt8).device(torch_npu::utils::get_npu_device_type());
    auto t = at::empty({static_cast<int64_t>(size)}, opts);
    auto* p = static_cast<uint8_t*>(const_cast<void*>(t.storage().data()));
    g_wsPool.push_back(std::move(t));
    return p;
}

void wsFree(uint8_t* p, size_t) {
    for (auto it = g_wsPool.begin(); it != g_wsPool.end(); ++it)
        if (it->storage().data() == p) { g_wsPool.erase(it); break; }
}

struct _WsInit {
    _WsInit() {
        auto sa = (void (*)( decltype(wsAlloc)*))dlsym(RTLD_DEFAULT, "CatlassSetWorkspaceAlloc");
        auto sf = (void (*)( decltype(wsFree)*)) dlsym(RTLD_DEFAULT, "CatlassSetWorkspaceFree");
        if (sa) sa(wsAlloc);
        if (sf) sf(wsFree);
    }
} _wsInit;
}

namespace CatlassKernelWrapper {

using BasicMatmulOp = MatmulLike<CatlassKernel::BasicMatmul>;
static auto& basic_matmul = BasicMatmulOp::Run;
REGISTER_TORCH_FUNC(basic_matmul);

using BatchedMatmulOp = BatchedMatmulLike<CatlassKernel::BatchedMatmul>;
static auto& batched_matmul = BatchedMatmulOp::Run;
REGISTER_TORCH_FUNC(batched_matmul);

using GroupedMatmulSliceMOp = GroupedMatmulLike<CatlassKernel::GroupedMatmulSliceM, GmmSliceDir::M>;
static auto& grouped_matmul_slice_m = GroupedMatmulSliceMOp::Run;
REGISTER_TORCH_FUNC(grouped_matmul_slice_m);

using GroupedMatmulSliceKOp = GroupedMatmulLike<CatlassKernel::GroupedMatmulSliceK, GmmSliceDir::K>;
static auto& grouped_matmul_slice_k = GroupedMatmulSliceKOp::Run;
REGISTER_TORCH_FUNC(grouped_matmul_slice_k);

using GroupedMatmulOp = GroupedMatmulLike<CatlassKernel::GroupedMatmul, GmmSliceDir::K>;
static auto& grouped_matmul = GroupedMatmulOp::Run;
REGISTER_TORCH_FUNC(grouped_matmul);

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

using GroupedMatmulSliceMPerTokenDequantMoeOp =
    GroupedQuantMatmulLike<CatlassKernel::GroupedMatmulSliceMPerTokenDequantMoe, GmmSliceDir::M>;
static auto& grouped_matmul_slice_m_per_token_dequant_moe = GroupedMatmulSliceMPerTokenDequantMoeOp::Run;
REGISTER_TORCH_FUNC(grouped_matmul_slice_m_per_token_dequant_moe);

using GroupedMatmulSliceMPerTokenDequantOp =
    GroupedQuantMatmulLike<CatlassKernel::GroupedMatmulSliceMPerTokenDequant, GmmSliceDir::M>;
static auto& grouped_matmul_slice_m_per_token_dequant = GroupedMatmulSliceMPerTokenDequantOp::Run;
REGISTER_TORCH_FUNC(grouped_matmul_slice_m_per_token_dequant);

using GroupedMatmulSliceKPerTokenDequantOp =
    GroupedQuantMatmulLike<CatlassKernel::GroupedMatmulSliceKPerTokenDequant, GmmSliceDir::K>;
static auto& grouped_matmul_slice_k_per_token_dequant = GroupedMatmulSliceKPerTokenDequantOp::Run;
REGISTER_TORCH_FUNC(grouped_matmul_slice_k_per_token_dequant);

using SplitkMatmulOp = MatmulLike<CatlassKernel::SplitkMatmul>;
static auto& splitk_matmul = SplitkMatmulOp::Run;
REGISTER_TORCH_FUNC(splitk_matmul);

using QuantMatmulOp = QuantMatmulLike<CatlassKernel::QuantMatmul>;
static auto& quant_matmul = QuantMatmulOp::Run;
REGISTER_TORCH_FUNC(quant_matmul);

using PaddingSplitkMatmulOp = MatmulLike<CatlassKernel::PaddingSplitkMatmul>;
static auto& padding_splitk_matmul = PaddingSplitkMatmulOp::Run;
REGISTER_TORCH_FUNC(padding_splitk_matmul);

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

using MatmulEvgOp = MatmulEvgLike<CatlassKernel::MatmulEvg>;
static auto& matmul_evg = MatmulEvgOp::Run;
REGISTER_TORCH_FUNC(matmul_evg);

using A2Fp8E4M3MatmulOp = MatmulLike<CatlassKernel::A2Fp8E4M3Matmul>;
static auto& a2_fp8_e4m3_matmul = A2Fp8E4M3MatmulOp::Run;
REGISTER_TORCH_FUNC(a2_fp8_e4m3_matmul);

static auto& mla = MlaOp::Run;
REGISTER_TORCH_FUNC(mla);

static auto& flash_attention_infer = FlashAttentionInferOp::Run;
REGISTER_TORCH_FUNC(flash_attention_infer);

static auto& flash_attention_infer_tla = FlashAttentionInferTLAOp::Run;
REGISTER_TORCH_FUNC(flash_attention_infer_tla);

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

using Ascend950Fp8MxBatchMatmulOp = MxBatchedMatmulLike<CatlassKernel::Ascend950Fp8MxBatchMatmul>;
static auto& ascend950_fp8_mx_batch_matmul = Ascend950Fp8MxBatchMatmulOp::Run;
REGISTER_TORCH_FUNC(ascend950_fp8_mx_batch_matmul);

using BroadcastMatmulPerblockQuantOp = BroadcastMatmulPerblockQuantLike<CatlassKernel::BroadcastMatmulPerblockQuant>;
static auto& broadcast_matmul_perblock_quant = BroadcastMatmulPerblockQuantOp::Run;
REGISTER_TORCH_FUNC(broadcast_matmul_perblock_quant);

using Ascend950DualLevelQuantMxBatchMatmulOp =
    DualLevelQuantMxBatchedMatmulLike<CatlassKernel::Ascend950DualLevelQuantMxBatchMatmul>;
static auto& ascend950_dual_level_quant_mx_batch_matmul = Ascend950DualLevelQuantMxBatchMatmulOp::Run;
REGISTER_TORCH_FUNC(ascend950_dual_level_quant_mx_batch_matmul);

} // namespace CatlassKernelWrapper
