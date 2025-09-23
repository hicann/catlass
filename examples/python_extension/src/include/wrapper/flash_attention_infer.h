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
FAKernelInfo GetKernelInfo(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                           const std::vector<int64_t> &actual_seq_lengths, const std::vector<int64_t> &actual_seq_lengths_kv,
                           const at::Tensor &atten_mask, const at::Tensor &block_table, const std::string &input_layout,
                           const int32_t &num_heads, const int32_t &num_key_value_heads, const int32_t &sparse_mode);
} // namespace CatlassKernelWrapper::FAILike
#endif