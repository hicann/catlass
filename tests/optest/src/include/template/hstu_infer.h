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

#ifndef OPTEST_HSTU_INFER_H
#define OPTEST_HSTU_INFER_H

#include <algorithm>
#include <stdexcept>
#include <string>

#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_prebuilt.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"

namespace CatlassKernelWrapper {

struct HstuInferHost {
    using OutputType = at::Tensor;

    static void GetKernelInfo(
        const at::Tensor& query,
        const at::Tensor& key,
        const at::Tensor& value,
        const at::Tensor& actual_q_seqlen,
        const at::Tensor& actual_kv_seqlen,
        const at::Tensor& block_table,
        const std::string& input_layout,
        int64_t num_heads,
        int64_t num_kv_heads,
        int64_t paged_block_size,
        double silu_scale,
        int64_t mask_type,
        CatlassKernel::HstuInferParams& params)
    {
        TORCH_CHECK(
            input_layout == "TND" || input_layout == "NTD",
            "input_layout of ascend950_hstu_infer only supports TND or NTD");

        aclDataType queryDtype = TorchDtypeToAclDtype(query.scalar_type());
        aclDataType keyDtype = TorchDtypeToAclDtype(key.scalar_type());
        aclDataType valueDtype = TorchDtypeToAclDtype(value.scalar_type());
        TORCH_CHECK(
            queryDtype == keyDtype && queryDtype == valueDtype,
            "query, key and value must have the same dtype");
        TORCH_CHECK(queryDtype == ACL_FLOAT16, "ascend950_hstu_infer supports float16 only");

        TORCH_CHECK(actual_q_seqlen.scalar_type() == at::kLong, "actual_q_seqlen must be int64");
        TORCH_CHECK(actual_kv_seqlen.scalar_type() == at::kLong, "actual_kv_seqlen must be int64");
        int64_t batch = actual_q_seqlen.numel() - 1;
        TORCH_CHECK(
            batch >= 1, "actual_q_seqlen must be a cumulative array with batch + 1 elements");
        TORCH_CHECK(
            actual_kv_seqlen.numel() == batch + 1,
            "actual_kv_seqlen must have the same size as actual_q_seqlen");

        int64_t embeddingSize = 0;
        int64_t qTokens = 0;
        if (input_layout == "TND") {
            TORCH_CHECK(query.dim() == 3, "query must be a 3-D tensor in TND layout");
            qTokens = query.size(0);
            embeddingSize = query.size(2);
            TORCH_CHECK(query.size(1) == num_heads, "query.size(1) must match num_heads in TND layout");
        } else {
            TORCH_CHECK(query.dim() == 3, "query must be a 3-D tensor in NTD layout");
            qTokens = query.size(1);
            embeddingSize = query.size(2);
            TORCH_CHECK(query.size(0) == num_heads, "query.size(0) must match num_heads in NTD layout");
        }
        TORCH_CHECK(
            embeddingSize >= 1 && embeddingSize <= 256,
            "ascend950_hstu_infer only supports head_dim in range [1, 256], got ",
            embeddingSize);
        TORCH_CHECK(
            num_kv_heads == num_heads, "ascend950_hstu_infer only supports num_kv_heads == num_heads (MHA)");
        TORCH_CHECK(mask_type == 0 || mask_type == 1, "mask_type of ascend950_hstu_infer should be 0 or 1");

        // KV layout: non-paged follows input_layout; paged is always NHD
        // (num_paged_blocks, paged_block_size, kv_heads, head_dim).
        if (paged_block_size > 0) {
            TORCH_CHECK(key.dim() == 4, "key must be a 4-D tensor (NHD) when paged KV cache is enabled");
            TORCH_CHECK(
                key.size(1) == paged_block_size, "key.size(1) must match paged_block_size in NHD layout");
            TORCH_CHECK(key.size(2) == num_kv_heads, "key.size(2) must match num_kv_heads in NHD layout");
            TORCH_CHECK(key.size(3) == embeddingSize, "key.size(3) must match head_dim in NHD layout");
            TORCH_CHECK(
                value.sizes() == key.sizes(), "key and value must have the same shape");
        } else if (input_layout == "TND") {
            TORCH_CHECK(key.dim() == 3, "key must be a 3-D tensor in TND layout");
            TORCH_CHECK(key.size(1) == num_kv_heads, "key.size(1) must match num_kv_heads in TND layout");
            TORCH_CHECK(key.size(2) == embeddingSize, "key.size(2) must match head_dim in TND layout");
            TORCH_CHECK(
                value.sizes() == key.sizes(), "key and value must have the same shape");
        } else {
            TORCH_CHECK(key.dim() == 3, "key must be a 3-D tensor in NTD layout");
            TORCH_CHECK(key.size(0) == num_kv_heads, "key.size(0) must match num_kv_heads in NTD layout");
            TORCH_CHECK(key.size(2) == embeddingSize, "key.size(2) must match head_dim in NTD layout");
            TORCH_CHECK(
                value.sizes() == key.sizes(), "key and value must have the same shape");
        }

        // Paged KV cache metadata needs per-batch KV lengths on host.
        auto kvSeqHost = actual_kv_seqlen.cpu().contiguous();
        const int64_t* kvSeqData = kvSeqHost.data_ptr<int64_t>();
        int64_t maxKvSeqlen = 0;
        int64_t numPagedBlocks = 0;
        for (int64_t i = 0; i < batch; i++) {
            int64_t kvLen = kvSeqData[i + 1] - kvSeqData[i];
            maxKvSeqlen = std::max(maxKvSeqlen, kvLen);
            if (paged_block_size > 0) {
                numPagedBlocks += (kvLen + paged_block_size - 1) / paged_block_size;
            }
        }

        params.inputAddr.resize(6);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(actual_q_seqlen.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(actual_kv_seqlen.storage().data()));
        params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(query.storage().data()));
        params.inputAddr[3] = static_cast<uint8_t*>(const_cast<void*>(key.storage().data()));
        params.inputAddr[4] = static_cast<uint8_t*>(const_cast<void*>(value.storage().data()));
        params.inputAddr[5] = static_cast<uint8_t*>(const_cast<void*>(block_table.storage().data()));

        params.batch = static_cast<uint32_t>(batch);
        params.numHeads = static_cast<uint32_t>(num_heads);
        params.kvHeads = static_cast<uint32_t>(num_kv_heads);
        params.embeddingSize = static_cast<uint32_t>(embeddingSize);
        params.maxKvSeqlen = static_cast<uint32_t>(maxKvSeqlen);
        params.numPagedBlocks = static_cast<uint32_t>(numPagedBlocks);
        params.pagedBlockSize = static_cast<uint32_t>(paged_block_size);
        params.maskType = static_cast<uint32_t>(mask_type);
        params.siluScale = static_cast<float>(silu_scale);
        params.layout = input_layout;
        params.dataType = queryDtype;
    }

    static OutputType AllocOutput(CatlassKernel::HstuInferParams& params)
    {
        OutputType output;
        if (params.layout == "TND") {
            // qTokens is derived from cumulative Q lengths; use the total token count.
            output = GetOutputTensor(
                {params.qNtokens, params.numHeads, params.embeddingSize},
                AclDtypeToTorchDtype(params.dataType));
        } else {
            output = GetOutputTensor(
                {params.numHeads, params.qNtokens, params.embeddingSize},
                AclDtypeToTorchDtype(params.dataType));
        }
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));
        return output;
    }
};

struct Ascend950HstuInferOp : HstuInferHost {
    using HstuInferHost::OutputType;

    static OutputType Run(
        const at::Tensor& query,
        const at::Tensor& key,
        const at::Tensor& value,
        const at::Tensor& actual_q_seqlen,
        const at::Tensor& actual_kv_seqlen,
        const at::Tensor& block_table,
        const std::string& input_layout,
        int64_t num_heads,
        int64_t num_kv_heads,
        int64_t paged_block_size,
        double silu_scale,
        int64_t mask_type)
    {
        CatlassKernel::HstuInferParams params;
        GetKernelInfo(
            query, key, value, actual_q_seqlen, actual_kv_seqlen, block_table, input_layout, num_heads,
            num_kv_heads, paged_block_size, silu_scale, mask_type, params);

        // Total Q tokens from the cumulative lengths (host side).
        auto qSeqHost = actual_q_seqlen.cpu().contiguous();
        params.qNtokens = static_cast<uint32_t>(qSeqHost.data_ptr<int64_t>()[params.batch]);

        OutputType output = AllocOutput(params);
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(CatlassKernel::Ascend950HstuInfer, aicCoreNum, stream, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif