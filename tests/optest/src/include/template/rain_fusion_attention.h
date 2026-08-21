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

#ifndef OPTEST_RAIN_FUSION_ATTENTION_H
#define OPTEST_RAIN_FUSION_ATTENTION_H

#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_prebuilt.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"

namespace CatlassKernelWrapper {

struct Ascend950RainFusionAttentionHost {
    using OutputType = at::Tensor;

    static void GetKernelInfo(
        const at::Tensor& query,
        const at::Tensor& key,
        const at::Tensor& value,
        const at::Tensor& select_idx,
        const at::Tensor& select_num_idx,
        const at::Tensor& block_shape,
        const at::Tensor& actual_seq_lengths,
        const at::Tensor& actual_seq_lengths_kv,
        const std::string& input_layout,
        int64_t num_heads,
        int64_t num_key_value_heads,
        int64_t is_varied_len,
        CatlassKernel::RainFusionAttentionParams& params)
    {
        TORCH_CHECK(
            input_layout == "TND" || input_layout == "BNSD",
            "input_layout of ascend950_rain_fusion_attention must be TND or BNSD");

        aclDataType queryDtype = TorchDtypeToAclDtype(query.scalar_type());
        aclDataType keyDtype = TorchDtypeToAclDtype(key.scalar_type());
        aclDataType valueDtype = TorchDtypeToAclDtype(value.scalar_type());
        TORCH_CHECK(
            queryDtype == keyDtype && queryDtype == valueDtype,
            "query, key and value must have the same dtype");
        TORCH_CHECK(
            queryDtype == ACL_FLOAT16 || queryDtype == ACL_BF16,
            "rain_fusion_attention supports float16 and bfloat16 only");

        int64_t batch = actual_seq_lengths.numel();
        int64_t embeddingSize = query.size(-1);

        // Infer num_heads / num_key_value_heads from tensors when callers pass 0.
        if (num_heads <= 0) {
            num_heads = query.size(1);
        }
        if (num_key_value_heads <= 0) {
            num_key_value_heads = key.size(1);
        }
        TORCH_CHECK(num_heads > 0 && num_key_value_heads > 0, "num_heads/num_key_value_heads must be positive");
        TORCH_CHECK(num_heads % num_key_value_heads == 0, "num_heads must be divisible by num_key_value_heads");

        int64_t blockShapeX = block_shape[0].item<int64_t>();
        int64_t blockShapeY = block_shape[1].item<int64_t>();

        int64_t maxKvBlockNum = select_idx.size(2);

        uint32_t qInputLayout = (input_layout == "BNSD") ? 1 : 0;
        uint32_t kvInputLayout = (input_layout == "BNSD") ? 1 : 0;

        float scaleValue = static_cast<float>(1.0 / std::sqrt(1.0 * embeddingSize));

        // Per-batch sequence lengths (host side, needed for tiling)
        params.qSeqHost.resize(batch);
        params.kvSeqHost.resize(batch);
        std::memcpy(params.qSeqHost.data(), actual_seq_lengths.cpu().data_ptr(), batch * sizeof(int64_t));
        std::memcpy(params.kvSeqHost.data(), actual_seq_lengths_kv.cpu().data_ptr(), batch * sizeof(int64_t));

        uint32_t maxQSeqlen = 0;
        uint32_t maxKvSeqlen = 0;
        for (int64_t b = 0; b < batch; b++) {
            maxQSeqlen = std::max(maxQSeqlen, static_cast<uint32_t>(params.qSeqHost[b]));
            maxKvSeqlen = std::max(maxKvSeqlen, static_cast<uint32_t>(params.kvSeqHost[b]));
        }

        // Device buffers: qSeq, kvSeq, q, k, v, selectIdx, selectNumIdx
        params.inputAddr.resize(7);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(actual_seq_lengths.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(actual_seq_lengths_kv.storage().data()));
        params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(query.storage().data()));
        params.inputAddr[3] = static_cast<uint8_t*>(const_cast<void*>(key.storage().data()));
        params.inputAddr[4] = static_cast<uint8_t*>(const_cast<void*>(value.storage().data()));
        params.inputAddr[5] = static_cast<uint8_t*>(const_cast<void*>(select_idx.storage().data()));
        params.inputAddr[6] = static_cast<uint8_t*>(const_cast<void*>(select_num_idx.storage().data()));

        params.batch = static_cast<uint32_t>(batch);
        params.numHeads = static_cast<uint32_t>(num_heads);
        params.kvHeads = static_cast<uint32_t>(num_key_value_heads);
        params.embeddingSize = static_cast<uint32_t>(embeddingSize);
        params.blockShapeX = static_cast<uint32_t>(blockShapeX);
        params.blockShapeY = static_cast<uint32_t>(blockShapeY);
        params.maxKvBlockNum = static_cast<uint32_t>(maxKvBlockNum);
        params.maxQSeqlen = maxQSeqlen;
        params.maxKvSeqlen = maxKvSeqlen;
        params.maskType = 0;
        params.scaleValue = scaleValue;
        params.qInputLayout = qInputLayout;
        params.kvInputLayout = kvInputLayout;
        params.isVariedLen = static_cast<uint32_t>(is_varied_len);
        params.dataType = queryDtype;
    }

    static OutputType AllocOutput(CatlassKernel::RainFusionAttentionParams& params)
    {
        // Output has the same shape as query
        std::vector<int64_t> outShape;
        if (params.qInputLayout == 1) {
            // BNSD: [batch, numHeads, maxQSeqlen, embeddingSize]
            outShape = {static_cast<int64_t>(params.batch), static_cast<int64_t>(params.numHeads),
                        static_cast<int64_t>(params.maxQSeqlen), static_cast<int64_t>(params.embeddingSize)};
        } else {
            // TND: [total_q_tokens, numHeads, embeddingSize]
            uint64_t totalQTokens = 0;
            for (uint32_t b = 0; b < params.batch; b++) {
                totalQTokens += params.qSeqHost[b];
            }
            outShape = {static_cast<int64_t>(totalQTokens), static_cast<int64_t>(params.numHeads),
                        static_cast<int64_t>(params.embeddingSize)};
        }
        OutputType output = GetOutputTensor(outShape, AclDtypeToTorchDtype(params.dataType));
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));
        return output;
    }
};

struct Ascend950RainFusionAttentionOp : Ascend950RainFusionAttentionHost {
    static OutputType Run(
        const at::Tensor& query,
        const at::Tensor& key,
        const at::Tensor& value,
        const at::Tensor& select_idx,
        const at::Tensor& select_num_idx,
        const at::Tensor& block_shape,
        const at::Tensor& actual_seq_lengths,
        const at::Tensor& actual_seq_lengths_kv,
        const std::string& input_layout,
        int64_t num_heads,
        int64_t num_key_value_heads,
        int64_t is_varied_len)
    {
        CatlassKernel::RainFusionAttentionParams params;
        GetKernelInfo(
            query, key, value, select_idx, select_num_idx, block_shape,
            actual_seq_lengths, actual_seq_lengths_kv, input_layout,
            num_heads, num_key_value_heads, is_varied_len, params);
        OutputType output = AllocOutput(params);
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(CatlassKernel::RainFusionAttention, aicCoreNum, stream, params);
        return output;
    }
};
} // namespace CatlassKernelWrapper

#endif
