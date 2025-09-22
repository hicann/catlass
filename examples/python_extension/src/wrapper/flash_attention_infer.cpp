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

namespace CatlassKernelWrapper::FAILike {
using namespace CatlassKernel;
using OutputType = at::Tensor;

FAKernelInfo GetKernelInfo(const at::Tensor &qNtokens, const at::Tensor &qSeqDevice, const at::Tensor &kvSeqDevice,
                           const at::Tensor &qDevice, const at::Tensor &kDevice, const at::Tensor &vDevice, const at::Tensor &maskDevice,
                           const at::Tensor &blockTableDevice, const int32_t &batch, const int32_t &q_seqlen, const int32_t &kv_seqlen,
                           const int32_t &num_head, const int32_t &kv_heads, const int32_t &embedding_size, const int32_t &is_varied_len,
                           const int32_t &mask_type, const std::string &str_dtype, const int32_t &kv_dtype)  // 暴露变量整改
{
    FAKernelInfo kernelInfo;

    kernelInfo.inputAddr.resize(8);
    kernelInfo.inputAddr[0] = static_cast<uint8_t *>(const_cast<void *>(qNtokens.storage().data()));
    kernelInfo.inputAddr[1] = static_cast<uint8_t *>(const_cast<void *>(qSeqDevice.storage().data()));
    kernelInfo.inputAddr[2] = static_cast<uint8_t *>(const_cast<void *>(kvSeqDevice.storage().data()));
    kernelInfo.inputAddr[3] = static_cast<uint8_t *>(const_cast<void *>(qDevice.storage().data()));
    kernelInfo.inputAddr[4] = static_cast<uint8_t *>(const_cast<void *>(kDevice.storage().data()));
    kernelInfo.inputAddr[5] = static_cast<uint8_t *>(const_cast<void *>(vDevice.storage().data()));
    kernelInfo.inputAddr[6] = static_cast<uint8_t *>(const_cast<void *>(maskDevice.storage().data()));
    kernelInfo.inputAddr[7] = static_cast<uint8_t *>(const_cast<void *>(blockTableDevice.storage().data()));

    kernelInfo.batch = batch;
    kernelInfo.qSeqlen = q_seqlen;
    kernelInfo.kvSeqlen = kv_seqlen;
    kernelInfo.numHeads = num_head;
    kernelInfo.kvHeads = kv_heads;
    kernelInfo.embeddingSize = embedding_size;
    kernelInfo.blockSize = 128;  // 此处存在问题需要整改
    kernelInfo.isVariedLen = is_varied_len;
    kernelInfo.maskType = mask_type;

    if ((str_dtype != "float16") && (str_dtype != "bf16")) {
        throw std::runtime_error("str_dtype of fai should be float16 or bf16.");
    }
    kernelInfo.dataType = TypeStrToAclDtype(str_dtype);

    return kernelInfo;
}

OutputType AllocOutput(FAKernelInfo &kernelInfo)
{
    void *qNtokens = kernelInfo.inputAddr.at(0);
    int32_t numTokens = static_cast<int32_t *>(qNtokens)[0];

    int32_t numHeads = kernelInfo.numHeads;

    int32_t embeddingSize = kernelInfo.embeddingSize;

    OutputType output = GetOutputTensor({numTokens, numHeads, embeddingSize}, AclDtypeToTorchDtype(kernelInfo.dataType));
    kernelInfo.outputAddr.resize(1);
    kernelInfo.outputAddr[0] = static_cast<uint8_t *>(const_cast<void *>(output.storage().data()));
    return output;
}
} // namespace CatlassKernelWrapper::MatmulLike