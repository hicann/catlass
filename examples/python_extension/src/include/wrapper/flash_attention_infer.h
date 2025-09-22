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

#ifndef PY_EXT_FAI
#define PY_EXT_FAI

#include <torch/torch.h>

#include "catlass_kernel.h"
#include "wrapper/catlass_kernel_wrapper.h"
#include "wrapper/common.h"
namespace CatlassKernelWrapper::FAILike {
using namespace CatlassKernel;
using OutputType = at::Tensor;

OutputType AllocOutput(FAKernelInfo &kernelInfo);
FAKernelInfo GetKernelInfo(const at::Tensor &qNtokens, const at::Tensor &qSeqDevice, const at::Tensor &kvSeqDevice,
                           const at::Tensor &qDevice, const at::Tensor &kDevice, const at::Tensor &vDevice, const at::Tensor &maskDevice,
                           const at::Tensor &blockTableDevice, const int32_t &batch, const int32_t &q_seqlen, const int32_t &kv_seqlen,
                           const int32_t &num_head, const int32_t &kv_heads, const int32_t &embedding_size, const int32_t &is_varied_len,
                           const int32_t &mask_type, const std::string &str_dtype, const int32_t &kv_dtype);
} // namespace CatlassKernelWrapper::FAILike
#endif