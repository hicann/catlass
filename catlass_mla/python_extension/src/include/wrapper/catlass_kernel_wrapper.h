/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PY_EXT_CATLASS_KERNEL_WRAPPER_H
#define PY_EXT_CATLASS_KERNEL_WRAPPER_H

#include <pybind11/stl.h>
#include <torch/extension.h>

#include "catlass_kernel.h"

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
);

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
);

} // namespace CatlassKernelWrapper

#endif // PY_EXT_CATLASS_KERNEL_WRAPPER_H
