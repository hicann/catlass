
/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <acl/acl.h>
#include <tiling/platform/platform_ascendc.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "catlass_kernel.h"
#include "wrapper/catlass_kernel_wrapper.h"

#define ACL_CHECK(status)                                                                    \
    do {                                                                                     \
        aclError error = status;                                                             \
        if (error != ACL_ERROR_NONE) {                                                       \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << error << std::endl;  \
        }                                                                                    \
    } while (0)

namespace py = pybind11;
using namespace CatlassKernel;

namespace CatlassKernelWrapper {

at::Tensor RunMLA(
    const at::Tensor &q,
    const at::Tensor &q_rope,
    const at::Tensor &k,
    const at::Tensor &k_rope,
    const int32_t kv_seqlen,
    const std::vector<int32_t> &kv_seqlens,
    const at::Tensor &block_table,
    const at::Tensor &s,
    const at::Tensor &p,
    const at::Tensor &result_temp,
    const at::Tensor &global_o,
    const at::Tensor &l,
    const at::Tensor &o_core_tmp,
    const std::string &dtype_str,
    const float softmax_scale
)
{
    torch::Dtype outputDataType;
    if (dtype_str == "float16") {
        outputDataType = torch::kFloat16;
    } else if(dtype_str == "bf16") {
        outputDataType = torch::kBFloat16;
    } else {
        throw std::runtime_error("unsupported dtype");
    }

    at::TensorOptions options = at::TensorOptions();
    options = options.dtype(outputDataType).layout(at::kStrided).requires_grad(false).device(
        torch_npu::utils::get_npu_device_type());

    // NOTE: MLA output has same shape as q_nope
    torch::Tensor result =  at_npu::native::empty_with_format(
        q.sizes(), options, ACL_FORMAT_ND);

    int32_t dTypeKey = (dtype_str == "float16") ? 0 : 1;

    // prepare inputs for kernel launch
    MLAKernelInfo mlaKernelInfo;
    mlaKernelInfo.dTypeKey = dTypeKey;
    mlaKernelInfo.softmaxScale = softmax_scale;
    mlaKernelInfo.inputAddr.resize(7);
    mlaKernelInfo.inputAddr[0] = static_cast<uint8_t *>(const_cast<void *>(q.storage().data()));
    mlaKernelInfo.inputAddr[1] = static_cast<uint8_t *>(const_cast<void *>(q_rope.storage().data()));
    mlaKernelInfo.inputAddr[2] = static_cast<uint8_t *>(const_cast<void *>(k.storage().data()));
    mlaKernelInfo.inputAddr[3] = static_cast<uint8_t *>(const_cast<void *>(k_rope.storage().data()));
    mlaKernelInfo.inputAddr[4] = static_cast<uint8_t *>(const_cast<void *>(block_table.storage().data()));
    mlaKernelInfo.inputAddr[5] = static_cast<uint8_t *>(const_cast<void *>(s.storage().data()));
    mlaKernelInfo.inputAddr[6] = static_cast<uint8_t *>(const_cast<void *>(p.storage().data()));


    mlaKernelInfo.outputAddr.resize(5);
    mlaKernelInfo.outputAddr[0] = static_cast<uint8_t *>(const_cast<void *>(result.storage().data()));
    mlaKernelInfo.outputAddr[1] = static_cast<uint8_t *>(const_cast<void *>(result_temp.storage().data()));
    mlaKernelInfo.outputAddr[2] = static_cast<uint8_t *>(const_cast<void *>(global_o.storage().data()));
    mlaKernelInfo.outputAddr[3] = static_cast<uint8_t *>(const_cast<void *>(l.storage().data()));
    mlaKernelInfo.outputAddr[4] = static_cast<uint8_t *>(const_cast<void *>(o_core_tmp.storage().data()));

    int32_t batch = q.sizes().at(0); // Q shape: [batch, kv_heads, head_size]
    int32_t qSeqLen = 1;  // NOTE: for now assume no spec-dec & MTP
    int32_t numBlocks = k.sizes().at(0);  // K shape: [num_block, block_size, kv_heads, head_size]
    int32_t blockSize = k.sizes().at(1);
    int32_t kvSeqLen = kv_seqlen;

    mlaKernelInfo.batch = batch;
    mlaKernelInfo.numHeads = q.sizes().at(1);
    mlaKernelInfo.embeddingSize = q.sizes().at(2);
    mlaKernelInfo.embeddingSizeRope = q_rope.sizes().at(2);
    mlaKernelInfo.numTokens = batch * qSeqLen;
    mlaKernelInfo.kvHeads = 1;
    mlaKernelInfo.numBlocks = numBlocks;
    mlaKernelInfo.blockSize = blockSize;
    mlaKernelInfo.maxKvSeqlen = kvSeqLen;  // ignore padding for now

    // NOTE: assume equal length for qSeqLen and kvSeqLen
    void *qSeq = nullptr;
    ACL_CHECK(aclrtMallocHost(&qSeq, batch * sizeof(int32_t)));
    int32_t *qSeq_int = static_cast<int32_t *>(qSeq);
    for (int i=0; i<batch; i++){
        qSeq_int[i] = qSeqLen;
    }

    void *kvSeq = nullptr;
    ACL_CHECK(aclrtMallocHost(&kvSeq, batch * sizeof(int32_t)));
    int32_t *kvSeq_int = static_cast<int32_t *>(kvSeq);
    for (int i = 0; i < batch; i++) {
        kvSeq_int[i] = kv_seqlens[i];
    }

    mlaKernelInfo.qSeqLen = qSeq_int;
    mlaKernelInfo.kvSeqLen = kvSeq_int;

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    LaunchMLA(aicCoreNum, stream, mlaKernelInfo);

    // clean memory
    aclrtFreeHost(qSeq);
    aclrtFreeHost(kvSeq);

    return result;
}

std::vector<uint64_t> PrepareMLA(
    const at::Tensor &q,
    const at::Tensor &q_rope,
    const at::Tensor &k,
    const at::Tensor &k_rope,
    const int32_t kv_seqlen,
    const std::vector<int32_t> &kv_seqlens,
    const at::Tensor &block_table,
    const at::Tensor &s,
    const at::Tensor &p,
    const at::Tensor &result_temp,
    const at::Tensor &global_o,
    const at::Tensor &l,
    const at::Tensor &o_core_tmp,
    const std::string &dtype_str
)
{
    torch::Dtype outputDataType;
    if (dtype_str == "float16") {
        outputDataType = torch::kFloat16;
    } else if(dtype_str == "bf16") {
        outputDataType = torch::kBFloat16;
    } else {
        throw std::runtime_error("unsupported dtype");
    }

    at::TensorOptions options = at::TensorOptions();
    options = options.dtype(outputDataType).layout(at::kStrided).requires_grad(false).device(
        torch_npu::utils::get_npu_device_type());

    // NOTE: MLA output has same shape as q_nope
    torch::Tensor result =  at_npu::native::empty_with_format(
        q.sizes(), options, ACL_FORMAT_ND);

    int32_t dTypeKey = (dtype_str == "float16") ? 0 : 1;

    // prepare inputs for kernel launch
    MLAKernelInfo mlaKernelInfo;
    mlaKernelInfo.dTypeKey = dTypeKey;
    mlaKernelInfo.inputAddr.resize(7);
    mlaKernelInfo.inputAddr[0] = static_cast<uint8_t *>(const_cast<void *>(q.storage().data()));
    mlaKernelInfo.inputAddr[1] = static_cast<uint8_t *>(const_cast<void *>(q_rope.storage().data()));
    mlaKernelInfo.inputAddr[2] = static_cast<uint8_t *>(const_cast<void *>(k.storage().data()));
    mlaKernelInfo.inputAddr[3] = static_cast<uint8_t *>(const_cast<void *>(k_rope.storage().data()));
    mlaKernelInfo.inputAddr[4] = static_cast<uint8_t *>(const_cast<void *>(block_table.storage().data()));
    mlaKernelInfo.inputAddr[5] = static_cast<uint8_t *>(const_cast<void *>(s.storage().data()));
    mlaKernelInfo.inputAddr[6] = static_cast<uint8_t *>(const_cast<void *>(p.storage().data()));


    mlaKernelInfo.outputAddr.resize(5);
    mlaKernelInfo.outputAddr[0] = static_cast<uint8_t *>(const_cast<void *>(result.storage().data()));
    mlaKernelInfo.outputAddr[1] = static_cast<uint8_t *>(const_cast<void *>(result_temp.storage().data()));
    mlaKernelInfo.outputAddr[2] = static_cast<uint8_t *>(const_cast<void *>(global_o.storage().data()));
    mlaKernelInfo.outputAddr[3] = static_cast<uint8_t *>(const_cast<void *>(l.storage().data()));
    mlaKernelInfo.outputAddr[4] = static_cast<uint8_t *>(const_cast<void *>(o_core_tmp.storage().data()));

    int32_t batch = q.sizes().at(0); // Q shape: [batch, kv_heads, head_size]
    int32_t qSeqLen = 1;  // NOTE: for now assume no spec-dec & MTP
    int32_t numBlocks = k.sizes().at(0);  // K shape: [num_block, block_size, kv_heads, head_size]
    int32_t blockSize = k.sizes().at(1);
    int32_t kvSeqLen = kv_seqlen;

    mlaKernelInfo.batch = batch;
    mlaKernelInfo.numHeads = q.sizes().at(1);
    mlaKernelInfo.embeddingSize = q.sizes().at(2);
    mlaKernelInfo.embeddingSizeRope = q_rope.sizes().at(2);
    mlaKernelInfo.numTokens = batch * qSeqLen;
    mlaKernelInfo.kvHeads = 1;
    mlaKernelInfo.numBlocks = numBlocks;
    mlaKernelInfo.blockSize = blockSize;
    mlaKernelInfo.maxKvSeqlen = kvSeqLen;  // ignore padding for now

    // NOTE: assume equal length for qSeqLen and kvSeqLen
    void *qSeq = nullptr;
    ACL_CHECK(aclrtMallocHost(&qSeq, batch * sizeof(int32_t)));
    int32_t *qSeq_int = static_cast<int32_t *>(qSeq);
    for (int i=0; i<batch; i++){
        qSeq_int[i] = qSeqLen;
    }

    void *kvSeq = nullptr;
    ACL_CHECK(aclrtMallocHost(&kvSeq, batch * sizeof(int32_t)));
    int32_t *kvSeq_int = static_cast<int32_t *>(kvSeq);
    for (int i = 0; i < batch; i++) {
        kvSeq_int[i] = kv_seqlens[i];
    }

    mlaKernelInfo.qSeqLen = qSeq_int;
    mlaKernelInfo.kvSeqLen = kvSeq_int;

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    std::vector<uint64_t> mla_prep_params = PrepareMLAParams(aicCoreNum, stream, mlaKernelInfo);

    // clean memory
    aclrtFreeHost(qSeq);
    aclrtFreeHost(kvSeq);

    return mla_prep_params;
}

} // namespace Catlass
