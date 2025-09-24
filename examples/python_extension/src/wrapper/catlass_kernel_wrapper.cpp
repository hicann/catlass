
/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include "wrapper/catlass_kernel_wrapper.h"

#include <tiling/platform/platform_ascendc.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "catlass_kernel.h"
#include "wrapper/grouped_matmul.h"
#include "wrapper/matmul.h"
#include "wrapper/conv.h"
#include "wrapper/flash_attention_infer.h"

namespace py = pybind11;
using namespace CatlassKernel;

namespace CatlassKernelWrapper {

at::Tensor RunBasicMatmul(const at::Tensor &mat1, const at::Tensor &mat2, const std::string &outDType)
{
    KernelInfo kernelInfo = MatmulLike::GetKernelInfo(mat1, mat2, outDType);
    at::Tensor output = MatmulLike::AllocOutput(kernelInfo);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    BasicMatmul(aicCoreNum, stream, kernelInfo);
    return output;
}

at::Tensor RunGroupedMatmul(const at::Tensor &mat1,
                            const at::Tensor &mat2,
                            const at::Tensor &groupList,
                            const std::string &outDType,
                            const bool transA,
                            const bool transB,
                            const bool splitK)
{
    KernelInfo kernelInfo = GroupedMatmulLike::GetKernelInfo(mat1, mat2, groupList, outDType, transA, transB, splitK);
    at::Tensor output = GroupedMatmulLike::AllocOutput(kernelInfo);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    GroupedMatmul(aicCoreNum, stream, kernelInfo);
    return output;
}

at::Tensor RunOptimizedMatmul(const at::Tensor &mat1, const at::Tensor &mat2, const std::string &outDType)
{
    KernelInfo kernelInfo = MatmulLike::GetKernelInfo(mat1, mat2, outDType);
    at::Tensor output = MatmulLike::AllocOutput(kernelInfo);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    OptimizedMatmul(aicCoreNum, stream, kernelInfo);
    return output;
}

at::Tensor RunConvBias(const at::Tensor &fmap, const at::Tensor &filter, const at::Tensor &bias,
                       const std::vector<int64_t> &strideList, const std::vector<int64_t> &padList,
                       const std::vector<int64_t> &dilationList, const std::string &outDType)
{
    ConvKernelInfo kernelInfo = ConvLike::GetKernelInfo(fmap, filter, bias,
                                                        strideList, padList, dilationList,
                                                        outDType);
    at::Tensor output = ConvLike::AllocOutput(kernelInfo);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    ConvBias(aicCoreNum, stream, kernelInfo);
    return output;
}

at::Tensor RunFlashAttentionInfer(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                           const std::vector<int64_t> &actual_seq_lengths, const std::vector<int64_t> &actual_seq_lengths_kv,
                           const at::Tensor &atten_mask, const at::Tensor &block_table, const std::string &input_layout,
                           const int64_t &num_heads, const int64_t &num_key_value_heads, const int64_t &sparse_mode)
{
    FAKernelInfo kernelInfo = FAILike::GetKernelInfo(query, key, value,
                                                    actual_seq_lengths, actual_seq_lengths_kv,
                                                    atten_mask, block_table, input_layout,
                                                    num_heads, num_key_value_heads, sparse_mode);
    at::Tensor output = FAILike::AllocOutput(kernelInfo);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    FlashAttentionInfer(aicCoreNum, stream, kernelInfo);
    return output;
}
} // namespace CatlassKernelWrapper