/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTEST_GROUPED_MATMUL_SLICE_M_FIXPIPE_DEQUANT_H
#define OPTEST_GROUPED_MATMUL_SLICE_M_FIXPIPE_DEQUANT_H

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_jit.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"
#include "type_utils.hpp"

namespace CatlassKernelWrapper {

using FixpipeKernelFn = void (*)(
    const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::MatmulFixPipeParams&);

template <FixpipeKernelFn KernelFunc>
struct GroupedMatmulSliceMFixpipeDequantLike {
    using OutputType = at::Tensor;

    static void GetKernelInfo(
        const at::Tensor& mat1, const at::Tensor& mat2,
        const at::Tensor& groupList, const at::Tensor& perTensorScale,
        const at::Tensor& perChannelScale, bool perChannelMode,
        CatlassKernel::TParams& tParams,
        CatlassKernel::MatmulFixPipeParams& params)
    {
        TORCH_CHECK(mat1.dim() >= 2, "mat1 must be at least 2D");
        TORCH_CHECK(mat2.dim() >= 2, "mat2 must be at least 2D");
        TORCH_CHECK(groupList.dim() == 1, "group_list must be 1D");
        TORCH_CHECK(mat1.device().type() == c10::DeviceType::PrivateUse1, "mat1 must be on NPU");
        TORCH_CHECK(mat2.device().type() == c10::DeviceType::PrivateUse1, "mat2 must be on NPU");
        TORCH_CHECK(groupList.device().type() == c10::DeviceType::PrivateUse1, "group_list must be on NPU");
        TORCH_CHECK(
            perChannelScale.device().type() == c10::DeviceType::PrivateUse1,
            "per_channel_scale must be on NPU");
        TORCH_CHECK(perTensorScale.numel() >= 1, "per_tensor_scale must contain at least one value");
        TORCH_CHECK(
            perTensorScale.scalar_type() == at::kFloat,
            "per_tensor_scale must be float32");
        TORCH_CHECK(
            perChannelScale.scalar_type() == torch::kUInt64,
            "per_channel_scale must contain uint64 packed fp32 values");

        tParams.element["A"] = TorchDtypeToAclDtype(mat1.scalar_type());
        tParams.element["B"] = TorchDtypeToAclDtype(mat2.scalar_type());
        tParams.element["C"] = TorchDtypeToAclDtype(torch::kFloat16);
        tParams.element["SCALE"] = TorchDtypeToAclDtype(perChannelScale.scalar_type());
        tParams.element["PER_TENSOR_SCALE"] = TorchDtypeToAclDtype(perTensorScale.scalar_type());

        int64_t m, k1, k2, n;
        int64_t dimA = mat1.dim() - 1;
        int64_t dimB = mat2.dim() - 1;
        m = mat1.size(dimA - 1);
        k1 = mat1.size(dimA);
        k2 = mat2.size(dimB - 1);
        n = mat2.size(dimB);
        TORCH_CHECK(k1 == k2, "mat1 K dimension must match mat2 K dimension");
        if (perChannelMode) {
            TORCH_CHECK(
                perChannelScale.numel() == n,
                "per_channel_scale length must match mat2 N dimension in per-channel mode");
        }

        auto g = static_cast<uint32_t>(groupList.numel());

        auto perTensorScaleCpu = perTensorScale.contiguous().cpu().to(at::kFloat);

        params.inputAddr.resize(4);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));
        params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(groupList.storage().data()));
        params.inputAddr[3] = static_cast<uint8_t*>(const_cast<void*>(perChannelScale.storage().data()));
        params.m = static_cast<uint32_t>(m);
        params.n = static_cast<uint32_t>(n);
        params.k = static_cast<uint32_t>(k1);
        params.batch = g;
        params.fixPipeQuantMode = perChannelMode
            ? CatlassKernel::MatmulFixPipeParams::FixPipeQuantMode::PerChannel
            : CatlassKernel::MatmulFixPipeParams::FixPipeQuantMode::PerTensor;
        params.perTensorScale = perTensorScaleCpu.data_ptr<float>()[0];
    }

    static OutputType AllocOutput(
        const CatlassKernel::TParams& tParams, CatlassKernel::MatmulFixPipeParams& params)
    {
        auto output = GetOutputTensor({params.m, params.n}, AclDtypeToTorchDtype(tParams.elem("C")));
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));
        return output;
    }

    static OutputType Run(
        const at::Tensor& mat1, const at::Tensor& mat2,
        const at::Tensor& groupList, const at::Tensor& perTensorScale,
        const at::Tensor& perChannelScale, bool perChannelMode)
    {
        CatlassKernel::TParams tParams;
        CatlassKernel::MatmulFixPipeParams params;
        GetKernelInfo(mat1, mat2, groupList, perTensorScale, perChannelScale, perChannelMode, tParams, params);
        OutputType output = AllocOutput(tParams, params);
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, tParams, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif
