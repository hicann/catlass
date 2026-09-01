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

#ifndef OPTEST_BASIC_SYRK_H
#define OPTEST_BASIC_SYRK_H

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_jit.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"
#include "type_utils.hpp"

namespace CatlassKernelWrapper {

using BasicSyrkKernelFn =
    void (*)(const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::MatmulParams&);

/**
 * @brief Torch adapter for Y = X * X^T (example 82_ascend950_basic_syrk).
 *
 * Reuses MatmulParams with m == n; only inputAddr[0] (X) and outputAddr[0] (Y) are used.
 */
template <BasicSyrkKernelFn KernelFunc>
struct BasicSyrkLike {
    using OutputType = at::Tensor;

    static OutputType Run(const at::Tensor& matX, const c10::ScalarType& outDType)
    {
        TORCH_CHECK(matX.dim() == 2, "ascend950_basic_syrk expects a 2-D input X of shape (M, K)");
        TORCH_CHECK(
            matX.scalar_type() == at::kBFloat16 || matX.scalar_type() == at::kHalf,
            "ascend950_basic_syrk currently supports bfloat16 / float16 inputs");
        TORCH_CHECK(
            outDType == at::kBFloat16 || outDType == at::kHalf,
            "ascend950_basic_syrk currently supports bfloat16 / float16 output");

        CatlassKernel::TParams tParams;
        CatlassKernel::MatmulParams params;

        // ELEMENT_A → X, ELEMENT_C → Y (layouts are fixed inside BlockMmadSyrkTla).
        tParams.element["A"] = TorchDtypeToAclDtype(matX.scalar_type());
        tParams.element["C"] = TorchDtypeToAclDtype(outDType);
        tParams.transpose["A"] = false;
        tParams.transpose["C"] = false;
        tParams.useNz["A"] = false;
        tParams.useNz["C"] = false;

        params.m = static_cast<uint32_t>(matX.size(0));
        params.k = static_cast<uint32_t>(matX.size(1));
        params.n = params.m; // Y is [M, M]

        params.inputAddr.resize(1);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(matX.storage().data()));

        OutputType output = GetOutputTensor({params.m, params.n}, AclDtypeToTorchDtype(tParams.elem("C")));
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));

        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, tParams, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif // OPTEST_BASIC_SYRK_H
