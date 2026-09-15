/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTEST_X_ATTENTION_H
#define OPTEST_X_ATTENTION_H

#include <array>
#include <string>
#include <utility>

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_prebuilt.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"

namespace CatlassKernelWrapper {

struct XAttentionOp {
    using OutputType = at::Tensor;

    static void CheckTensor(const at::Tensor& tensor, const char* name)
    {
        TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, " must be an NPU tensor");
        TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    }

    static uint8_t* TensorAddress(const at::Tensor& tensor)
    {
        return static_cast<uint8_t*>(const_cast<void*>(tensor.storage().data()));
    }

    static OutputType Run(
        const at::Tensor& query,
        const at::Tensor& shared_key_block,
        const at::Tensor& shared_value_block,
        const at::Tensor& unshared_key_block,
        const at::Tensor& unshared_value_block,
        const at::Tensor& unshared_block_table,
        const at::Tensor& shared_kv_lens,
        const at::Tensor& decode_step,
        const at::Tensor& shared_block_table,
        double scale_value)
    {
        const std::array<std::pair<const at::Tensor*, const char*>, 9> tensors{{
            {&query, "query"},
            {&shared_key_block, "shared_key_block"},
            {&shared_value_block, "shared_value_block"},
            {&unshared_key_block, "unshared_key_block"},
            {&unshared_value_block, "unshared_value_block"},
            {&unshared_block_table, "unshared_block_table"},
            {&shared_kv_lens, "shared_kv_lens"},
            {&decode_step, "decode_step"},
            {&shared_block_table, "shared_block_table"},
        }};
        for (const auto& item : tensors) {
            CheckTensor(*item.first, item.second);
        }

        TORCH_CHECK(query.dim() == 3, "query must have shape (batch * beam_size, num_heads, 128)");
        TORCH_CHECK(query.size(2) == 128, "x_attention currently requires embedding_size == 128");
        TORCH_CHECK(scale_value >= 0.0, "scale_value must be non-negative");

        aclDataType dataType = TorchDtypeToAclDtype(query.scalar_type());
        TORCH_CHECK(
            dataType == ACL_FLOAT16 || dataType == ACL_BF16,
            "x_attention supports float16 and bfloat16 only");
        for (const auto* tensor : {&shared_key_block, &shared_value_block, &unshared_key_block, &unshared_value_block}) {
            TORCH_CHECK(tensor->scalar_type() == query.scalar_type(), "all Q/K/V tensors must have the same dtype");
        }
        TORCH_CHECK(
            shared_kv_lens.scalar_type() == at::kInt && decode_step.scalar_type() == at::kInt,
            "shared_kv_lens and decode_step must use int32");
        TORCH_CHECK(
            shared_block_table.scalar_type() == at::kInt && unshared_block_table.scalar_type() == at::kInt,
            "block tables must use int32");
        TORCH_CHECK(decode_step.numel() == 1, "decode_step must contain one int32 value");

        bool sharedPaged = shared_block_table.numel() != 0;
        bool unsharedPaged = unshared_block_table.numel() != 0;
        TORCH_CHECK(
            sharedPaged != unsharedPaged,
            "exactly one of shared_block_table and unshared_block_table must be non-empty");

        int64_t batch = sharedPaged ? shared_block_table.size(0) : unshared_block_table.numel();
        TORCH_CHECK(batch > 0 && query.size(0) % batch == 0, "query token count must be divisible by batch");
        int64_t beamSize = query.size(0) / batch;
        TORCH_CHECK(beamSize > 0, "beam_size must be positive");
        int64_t numHeads = query.size(1);
        int64_t embeddingSize = query.size(2);
        TORCH_CHECK(shared_kv_lens.numel() == batch, "shared_kv_lens must contain one value per batch");

        int64_t kvHeads = 0;
        int64_t sharedKvSeqLen = 0;
        int64_t maxDecodeStep = 0;
        if (sharedPaged) {
            TORCH_CHECK(shared_block_table.dim() == 2, "shared_block_table must be 2D");
            TORCH_CHECK(
                shared_key_block.dim() == 4 && shared_value_block.sizes() == shared_key_block.sizes(),
                "paged shared key/value must have identical 4D shapes");
            TORCH_CHECK(shared_key_block.size(1) == 128, "shared paged cache block size must be 128");
            kvHeads = shared_key_block.size(2);
            TORCH_CHECK(shared_key_block.size(3) == embeddingSize, "shared cache embedding size mismatch");
            TORCH_CHECK(
                shared_key_block.size(0) == batch * shared_block_table.size(1),
                "shared paged cache block count must match the block table");
            sharedKvSeqLen = shared_block_table.size(1) * 128;

            TORCH_CHECK(
                unshared_key_block.dim() == 4 && unshared_value_block.sizes() == unshared_key_block.sizes(),
                "continuous unshared key/value must have identical 4D shapes");
            TORCH_CHECK(
                unshared_key_block.size(0) == query.size(0) && unshared_key_block.size(1) == kvHeads &&
                    unshared_key_block.size(3) == embeddingSize,
                "continuous unshared cache shape must be (batch * beam_size, kv_heads, max_decode_step, 128)");
            maxDecodeStep = unshared_key_block.size(2);
        } else {
            TORCH_CHECK(unshared_block_table.dim() == 1, "unshared_block_table must be 1D");
            TORCH_CHECK(
                shared_key_block.dim() == 3 && shared_value_block.sizes() == shared_key_block.sizes(),
                "continuous shared key/value must have identical 3D shapes");
            TORCH_CHECK(shared_key_block.size(0) % batch == 0, "shared cache token count must be divisible by batch");
            kvHeads = shared_key_block.size(1);
            TORCH_CHECK(shared_key_block.size(2) == embeddingSize, "shared cache embedding size mismatch");
            sharedKvSeqLen = shared_key_block.size(0) / batch;

            TORCH_CHECK(
                unshared_key_block.dim() == 5 && unshared_value_block.sizes() == unshared_key_block.sizes(),
                "paged unshared key/value must have identical 5D shapes");
            TORCH_CHECK(
                unshared_key_block.size(1) == beamSize && unshared_key_block.size(2) == kvHeads &&
                    unshared_key_block.size(4) == embeddingSize,
                "paged unshared cache shape must be (requests, beam_size, kv_heads, max_decode_step, 128)");
            maxDecodeStep = unshared_key_block.size(3);
        }

        TORCH_CHECK(kvHeads > 0 && numHeads % kvHeads == 0, "num_heads must be divisible by kv_heads");
        TORCH_CHECK(numHeads / kvHeads <= 128, "GQA group size must not exceed 128");
        TORCH_CHECK(maxDecodeStep > 0 && maxDecodeStep <= 256, "max_decode_step must be in [1, 256]");
        TORCH_CHECK(sharedKvSeqLen > 0, "shared KV sequence length must be positive");

        CatlassKernel::XAttentionParams params;
        params.inputAddr = {
            TensorAddress(query),
            TensorAddress(shared_key_block),
            TensorAddress(shared_value_block),
            TensorAddress(unshared_key_block),
            TensorAddress(unshared_value_block),
            TensorAddress(unshared_block_table),
            TensorAddress(shared_kv_lens),
            TensorAddress(decode_step),
            TensorAddress(shared_block_table),
        };
        params.numTokens = static_cast<uint32_t>(query.size(0));
        params.batch = static_cast<uint32_t>(batch);
        params.beamSize = static_cast<uint32_t>(beamSize);
        params.sharedKvSeqLen = static_cast<uint32_t>(sharedKvSeqLen);
        params.numHeads = static_cast<uint32_t>(numHeads);
        params.kvHeads = static_cast<uint32_t>(kvHeads);
        params.embeddingSize = static_cast<uint32_t>(embeddingSize);
        params.maxDecodeStep = static_cast<uint32_t>(maxDecodeStep);
        params.sharedPaged = sharedPaged;
        params.scaleValue = static_cast<float>(scale_value);
        params.dataType = dataType;

        OutputType output = GetOutputTensor(
            {query.size(0), query.size(1), query.size(2)}, query.scalar_type());
        params.outputAddr = {TensorAddress(output)};

        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        // params contains raw addresses. Keep their tensors alive until the queued
        // launcher has submitted the kernel to the NPU stream.
        const std::array<at::Tensor, 9> inputs{
            query, shared_key_block, shared_value_block, unshared_key_block,
            unshared_value_block, unshared_block_table, shared_kv_lens,
            decode_step, shared_block_table};
        TORCH_CHECK(CatlassKernel::XAttention != nullptr, "x_attention kernel is not available");
        at_npu::native::OpCommand::RunOpApiV2(
            "CatlassKernel::XAttention", [inputs, output, aicCoreNum, stream, params]() -> aclError {
                try {
                    CatlassKernel::XAttention(aicCoreNum, stream, params);
                } catch (...) {
                    return ACL_ERROR_INTERNAL_ERROR;
                }
                return ACL_SUCCESS;
            });
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif // OPTEST_X_ATTENTION_H
