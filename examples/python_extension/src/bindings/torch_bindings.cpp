/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>

#include "wrapper/catlass_kernel_wrapper.h"

#define NPU PrivateUse1

using namespace CatlassKernelWrapper;
TORCH_LIBRARY(CatlassTorch, m)
{
    m.def("basic_matmul(Tensor mat1, Tensor mat2, str c) -> Tensor")
        .def("grouped_matmul(Tensor mat1, Tensor mat2, Tensor groupList, str c, bool trans_a, bool trans_b, bool "
             "split) -> Tensor")
        .def("optimized_matmul(Tensor mat1, Tensor mat2, str c) -> Tensor")
        .def("flash_attention_infer(Tensor qNtokens, Tensor qSeqDevice, Tensor kvSeqDevice, Tensor qDevice, "
              "Tensor kDevice, Tensor vDevice, Tensor maskDevice, Tensor blockTableDevice, int batch, "
              "int q_seqlen, int kv_seqlen, num_head, int kv_heads, int embedding_size, int is_varied_len, "
              "int mask_type, str str_dtype, int kv_dtype) -> Tensor");
}

TORCH_LIBRARY_IMPL(CatlassTorch, NPU, m)
{
    m.impl("basic_matmul", &RunBasicMatmul)
        .impl("grouped_matmul", &RunGroupedMatmul)
        .impl("optimized_matmul", &RunOptimizedMatmul)
        .impl("flash_attention_infer", &RunFlashAttentionInfer);
}