/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/torch.h>

#include "catlass_kernel.h"
#include "wrapper/catlass_kernel_wrapper.h"
#include "wrapper/common.h"

// Macro function for unwinding acl errors.
#define ACL_CHECK(status)                                                                    \
    do {                                                                                     \
        aclError error = status;                                                             \
        if (error != ACL_ERROR_NONE) {                                                       \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << error << std::endl;  \
        }                                                                                    \
    } while (0)
namespace CatlassKernelWrapper::FAILike {
using namespace CatlassKernel;
using OutputType = at::Tensor;

FAKernelInfo GetKernelInfo(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                           const std::vector<int64_t> &actual_seq_lengths, const std::vector<int64_t> &actual_seq_lengths_kv,
                           const at::Tensor &atten_mask, const at::Tensor &block_table, const std::string &input_layout,
                           const int64_t &num_heads, const int64_t &num_key_value_heads, const int64_t &sparse_mode)
{
    if (input_layout != "TND") {
        throw std::runtime_error("input_layout of fai only support TND");
    }
    aclDataType query_dtype = TorchDtypeToAclDtype(query.scalar_type());
    aclDataType key_dtype = TorchDtypeToAclDtype(key.scalar_type());
    aclDataType value_dtype = TorchDtypeToAclDtype(value.scalar_type());
    if (query_dtype != key_dtype || query_dtype != value_dtype) {
        throw std::runtime_error("query, key and value must have the same dataType");
    }
    int64_t *qNtokens = nullptr;
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void**>(&qNtokens), 1 * sizeof(int64_t)));
    qNtokens[0] = query.sizes().at(0);
    int64_t embedding_size = query.sizes().at(2);

    int64_t batch = actual_seq_lengths.size();
    int64_t q_seqlen = *std::max_element(actual_seq_lengths.begin(), actual_seq_lengths.end());
    int64_t kv_seqlen = *std::max_element(actual_seq_lengths_kv.begin(), actual_seq_lengths_kv.end());

    int64_t mask_type;
    switch (sparse_mode)
    {
    case 0:
        mask_type = 0;
        break;
    case 1:
        mask_type = 3;
        break;
    default:
        throw std::runtime_error("sparse_mode of fai should be 0 or 1");
    }

    int64_t kv_dtype;
    if (block_table.numel() != 0) {
        kv_dtype = 1;
    } else {
        kv_dtype = 0;
    }

    aclDataType str_dtype = query_dtype;
    if ((str_dtype != ACL_FLOAT16) || (str_dtype != ACL_BF16)) {
        throw std::runtime_error("str_dtype of fai should be ACL_FLOAT16 or ACL_BF16");
    }

    uint64_t seqArraySize = batch * sizeof(int64_t);
    uint8_t *qSeqDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(qSeqDevice), seqArraySize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(qSeqDevice, seqArraySize, actual_seq_lengths.data(), seqArraySize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *kvSeqDevice;
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(kvSeqDevice), seqArraySize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(kvSeqDevice, seqArraySize, actual_seq_lengths.data(), seqArraySize, ACL_MEMCPY_HOST_TO_DEVICE));

    FAKernelInfo kernelInfo;
    kernelInfo.inputAddr.resize(7);
    kernelInfo.inputAddr[0] = static_cast<uint8_t *>(qSeqDevice);
    kernelInfo.inputAddr[1] = static_cast<uint8_t *>(kvSeqDevice);
    kernelInfo.inputAddr[2] = static_cast<uint8_t *>(const_cast<void *>(query.storage().data()));
    kernelInfo.inputAddr[3] = static_cast<uint8_t *>(const_cast<void *>(key.storage().data()));
    kernelInfo.inputAddr[4] = static_cast<uint8_t *>(const_cast<void *>(value.storage().data()));
    kernelInfo.inputAddr[5] = static_cast<uint8_t *>(const_cast<void *>(atten_mask.storage().data()));
    kernelInfo.inputAddr[6] = static_cast<uint8_t *>(const_cast<void *>(block_table.storage().data()));
    
    kernelInfo.qNtokens = static_cast<uint32_t>(qNtokens[0]);
    kernelInfo.batch = static_cast<uint32_t>(batch);
    kernelInfo.qSeqlen = static_cast<uint32_t>(q_seqlen);
    kernelInfo.kvSeqlen = static_cast<uint32_t>(kv_seqlen);
    kernelInfo.numHeads = static_cast<uint32_t>(num_heads);
    kernelInfo.kvHeads = static_cast<uint32_t>(num_key_value_heads);
    kernelInfo.embeddingSize = static_cast<uint32_t>(embedding_size);
    kernelInfo.blockSize = 128;
    kernelInfo.maskType = static_cast<uint32_t>(mask_type);
    kernelInfo.dataType = str_dtype;

    return kernelInfo;
}

OutputType AllocOutput(FAKernelInfo &kernelInfo)
{
    uint32_t numTokens = kernelInfo.qNtokens;

    uint32_t numHeads = kernelInfo.numHeads;

    uint32_t embeddingSize = kernelInfo.embeddingSize;

    OutputType output = GetOutputTensor({numTokens, numHeads, embeddingSize}, AclDtypeToTorchDtype(kernelInfo.dataType));
    kernelInfo.outputAddr.resize(1);
    kernelInfo.outputAddr[0] = static_cast<uint8_t *>(const_cast<void *>(output.storage().data()));
    return output;
}
} // namespace CatlassKernelWrapper::MatmulLike