/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTEST_MATMUL_H
#define OPTEST_MATMUL_H

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel.h"
#include "common/run_npu_func.h"
#include "common/utils.h"

namespace CatlassKernelWrapper {

template <typename KernelInfoType, void (*KernelFunc)(uint32_t, aclrtStream, const KernelInfoType&)>
struct MatmulLike {
    using OutputType = at::Tensor;
    using KernelInfo = KernelInfoType;

    static OutputType AllocOutput(KernelInfoType& kernelInfo)
    {
        OutputType output =
            GetOutputTensor({kernelInfo.m, kernelInfo.n}, AclDtypeToTorchDtype(kernelInfo.outputDataType));
        kernelInfo.outputAddr.resize(1);
        kernelInfo.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));
        return output;
    }

    static KernelInfoType GetKernelInfo(
        const at::Tensor& mat1, const at::Tensor& mat2, const std::string& outDType, bool transA, bool transB,
        bool formatA, bool formatB)
    {
        KernelInfoType kernelInfo{};
        kernelInfo.inputAddr.resize(2);
        kernelInfo.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
        kernelInfo.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));

        int64_t m, k1, k2, n;
        if (transA) {
            m = mat1.size(1);
            k1 = mat1.size(0);
        } else {
            m = mat1.size(0);
            k1 = mat1.size(1);
        }

        if (transB) {
            k2 = mat2.size(1);
            n = mat2.size(0);
        } else {
            k2 = mat2.size(0);
            n = mat2.size(1);
        }

        TORCH_CHECK(k1 == k2, "mat1 and mat2 shapes cannot be multiplied (", m, "x", k1, " and ", k2, "x", n, ")");

        kernelInfo.m = static_cast<uint32_t>(m);
        kernelInfo.k = static_cast<uint32_t>(k1);
        kernelInfo.n = static_cast<uint32_t>(n);
        kernelInfo.inputDataType = TorchDtypeToAclDtype(mat1.scalar_type());
        kernelInfo.outputDataType = TypeStrToAclDtype(outDType);
        kernelInfo.transA = transA;
        kernelInfo.transB = transB;
        kernelInfo.formatA = formatA;
        kernelInfo.formatB = formatB;
        return kernelInfo;
    }

    static OutputType Run(
        const at::Tensor& mat1, const at::Tensor& mat2, const std::string& outDType, bool transA, bool transB,
        bool formatA, bool formatB)
    {
        KernelInfoType kernelInfo = GetKernelInfo(mat1, mat2, outDType, transA, transB, formatA, formatB);
        OutputType output = AllocOutput(kernelInfo);
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, kernelInfo);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif
