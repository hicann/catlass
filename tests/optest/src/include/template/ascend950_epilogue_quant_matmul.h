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

#ifndef OPTEST_ASCEND950_EPILOGUE_QUANT_MATMUL_H
#define OPTEST_ASCEND950_EPILOGUE_QUANT_MATMUL_H

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_jit.h"
#include "common/run_npu_func.h"
#include "mx_matmul.h"
#include "torch_utils.h"
#include "type_utils.hpp"

namespace CatlassKernelWrapper {

using EpilogueQuantKernelFn = void (*)(
    const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::MatmulParams&);

inline void CheckFp8E4M3Tensor(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(
        tensor.scalar_type() == torch::kFloat8_e4m3fn,
        name, " must have dtype torch.float8_e4m3fn, got ", tensor.scalar_type());
}

inline void CheckPackedFp4Tensor(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(
        tensor.scalar_type() == torch::kFloat4_e2m1fn_x2,
        name, " must have dtype torch.float4_e2m1fn_x2, got ", tensor.scalar_type());
}

template <EpilogueQuantKernelFn KernelFunc>
struct Fp4MxMatmulPerTokenPerChannelLike {
    using OutputType = at::Tensor;

    static void GetKernelInfo(
        const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& mxScaleA,
        const at::Tensor& mxScaleB, const at::Tensor& perTokenScale, const at::Tensor& perChannelScale,
        CatlassKernel::TParams& tParams, CatlassKernel::MatmulParams& params)
    {
        CheckNpuTensor(mat1, "mat1");
        CheckNpuTensor(mat2, "mat2");
        CheckNpuTensor(mxScaleA, "mx_scale_a");
        CheckNpuTensor(mxScaleB, "mx_scale_b");
        CheckNpuTensor(perTokenScale, "per_token_scale");
        CheckNpuTensor(perChannelScale, "per_channel_scale");
        CheckSameDevice(mat1, "mat1", mat2, "mat2");
        CheckSameDevice(mat1, "mat1", mxScaleA, "mx_scale_a");
        CheckSameDevice(mat1, "mat1", mxScaleB, "mx_scale_b");
        CheckSameDevice(mat1, "mat1", perTokenScale, "per_token_scale");
        CheckSameDevice(mat1, "mat1", perChannelScale, "per_channel_scale");
        CheckPackedFp4Tensor(mat1, "mat1");
        CheckPackedFp4Tensor(mat2, "mat2");
        CheckMxScaleDType(mxScaleA, "mx_scale_a");
        CheckMxScaleDType(mxScaleB, "mx_scale_b");
        CheckFp8E4M3Tensor(perTokenScale, "per_token_scale");
        CheckFp8E4M3Tensor(perChannelScale, "per_channel_scale");

        TORCH_CHECK(mat1.dim() == 2, "mat1 must be 2-D with shape (M, K)");
        TORCH_CHECK(mat2.dim() == 2, "mat2 must be 2-D with shape (K, N)");
        TORCH_CHECK(mat1.is_contiguous(), "mat1 must be contiguous");
        TORCH_CHECK(mat2.is_contiguous(), "mat2 must be contiguous");
        TORCH_CHECK(mxScaleA.is_contiguous(), "mx_scale_a must be contiguous");
        TORCH_CHECK(mxScaleB.is_contiguous(), "mx_scale_b must be contiguous");
        TORCH_CHECK(perTokenScale.is_contiguous(), "per_token_scale must be contiguous");
        TORCH_CHECK(perChannelScale.is_contiguous(), "per_channel_scale must be contiguous");

        const int64_t m = mat1.size(0);
        const int64_t k = mat1.size(1);
        TORCH_CHECK(mat2.size(0) == k, "mat2 first dimension must equal K");
        const int64_t n = mat2.size(1);
        TORCH_CHECK(k % 2 == 0, "K must be even for packed FP4, got ", k);
        TORCH_CHECK(n % 2 == 0, "N must be even for packed FP4, got ", n);
        TORCH_CHECK(perTokenScale.numel() == m, "per_token_scale must have ", m, " elements");
        TORCH_CHECK(perChannelScale.numel() == n, "per_channel_scale must have ", n, " elements");

        tParams.element["A"] = test_utils::TypeCast<std::string, aclDataType>("float4_e2m1fn_x2");
        tParams.element["B"] = test_utils::TypeCast<std::string, aclDataType>("float4_e2m1fn_x2");
        tParams.element["C"] = ACL_FLOAT;
        tParams.element["MX_SCALE"] = test_utils::TypeCast<std::string, aclDataType>("float8_e8m0fnu");
        tParams.element["SCALE"] = test_utils::TypeCast<std::string, aclDataType>("float8_e4m3fn");
        tParams.element["PER_TOKEN_SCALE"] = tParams.element["SCALE"];
        tParams.element["D"] = ACL_FLOAT;
        tParams.transpose["A"] = false;
        tParams.transpose["B"] = false;
        tParams.transpose["C"] = false;
        tParams.useNz["A"] = false;
        tParams.useNz["B"] = false;
        tParams.useNz["C"] = false;

        params.m = static_cast<uint32_t>(m);
        params.k = static_cast<uint32_t>(k);
        params.n = static_cast<uint32_t>(n);

        uint32_t mxScaleAlignedK = 0;
        int64_t scaleANumel = 0;
        int64_t scaleBNumel = 0;
        ComputeMxScaleShapes(params.m, params.n, params.k, mxScaleAlignedK, scaleANumel, scaleBNumel);
        TORCH_CHECK(
            mxScaleA.numel() == scaleANumel,
            "mx_scale_a must have ", scaleANumel, " elements (m=", m, ", mxScaleAlignedK=", mxScaleAlignedK,
            "), got ", mxScaleA.numel());
        TORCH_CHECK(
            mxScaleB.numel() == scaleBNumel,
            "mx_scale_b must have ", scaleBNumel, " elements (n=", n, ", mxScaleAlignedK=", mxScaleAlignedK,
            "), got ", mxScaleB.numel());

        params.inputAddr.resize(6);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));
        params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(mxScaleA.storage().data()));
        params.inputAddr[3] = static_cast<uint8_t*>(const_cast<void*>(mxScaleB.storage().data()));
        params.inputAddr[4] = static_cast<uint8_t*>(const_cast<void*>(perChannelScale.storage().data()));
        params.inputAddr[5] = static_cast<uint8_t*>(const_cast<void*>(perTokenScale.storage().data()));
    }

    static OutputType AllocOutput(CatlassKernel::MatmulParams& params)
    {
        OutputType output = GetOutputTensor({params.m, params.n}, torch::kFloat32);
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));
        return output;
    }

    static OutputType Run(
        const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& mxScaleA,
        const at::Tensor& mxScaleB, const at::Tensor& perTokenScale, const at::Tensor& perChannelScale)
    {
        CatlassKernel::TParams tParams;
        CatlassKernel::MatmulParams params;
        GetKernelInfo(mat1, mat2, mxScaleA, mxScaleB, perTokenScale, perChannelScale, tParams, params);
        OutputType output = AllocOutput(params);
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, tParams, params);
        return output;
    }
};

template <EpilogueQuantKernelFn KernelFunc>
struct Fp8EpilogueQuantMatmulLike {
    using OutputType = at::Tensor;

    static void GetKernelInfo(
        const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& perTokenScale,
        const at::Tensor& perChannelScale, CatlassKernel::TParams& tParams, CatlassKernel::MatmulParams& params)
    {
        CheckNpuTensor(mat1, "mat1");
        CheckNpuTensor(mat2, "mat2");
        CheckNpuTensor(perTokenScale, "per_token_scale");
        CheckNpuTensor(perChannelScale, "per_channel_scale");
        CheckSameDevice(mat1, "mat1", mat2, "mat2");
        CheckSameDevice(mat1, "mat1", perTokenScale, "per_token_scale");
        CheckSameDevice(mat1, "mat1", perChannelScale, "per_channel_scale");
        CheckFp8E4M3Tensor(mat1, "mat1");
        CheckFp8E4M3Tensor(mat2, "mat2");
        CheckFp8E4M3Tensor(perTokenScale, "per_token_scale");
        CheckFp8E4M3Tensor(perChannelScale, "per_channel_scale");

        TORCH_CHECK(mat1.dim() == 2, "mat1 must be 2-D with shape (M, K)");
        TORCH_CHECK(mat2.dim() == 2, "mat2 must be 2-D with shape (K, N)");
        TORCH_CHECK(mat1.is_contiguous(), "mat1 must be contiguous");
        TORCH_CHECK(mat2.is_contiguous(), "mat2 must be contiguous");
        TORCH_CHECK(perTokenScale.is_contiguous(), "per_token_scale must be contiguous");
        TORCH_CHECK(perChannelScale.is_contiguous(), "per_channel_scale must be contiguous");

        const int64_t m = mat1.size(0);
        const int64_t k = mat1.size(1);
        TORCH_CHECK(mat2.size(0) == k, "mat2 first dimension must equal K");
        const int64_t n = mat2.size(1);
        TORCH_CHECK(perTokenScale.numel() == m, "per_token_scale must have ", m, " elements");
        TORCH_CHECK(perChannelScale.numel() == n, "per_channel_scale must have ", n, " elements");

        tParams.element["A"] = TorchDtypeToAclDtype(mat1.scalar_type());
        tParams.element["B"] = TorchDtypeToAclDtype(mat2.scalar_type());
        tParams.element["C"] = ACL_FLOAT;
        tParams.element["SCALE"] = TorchDtypeToAclDtype(perChannelScale.scalar_type());
        tParams.element["PER_TOKEN_SCALE"] = TorchDtypeToAclDtype(perTokenScale.scalar_type());
        tParams.element["D"] = ACL_FLOAT;
        tParams.transpose["A"] = false;
        tParams.transpose["B"] = false;
        tParams.transpose["C"] = false;
        tParams.useNz["A"] = false;
        tParams.useNz["B"] = false;
        tParams.useNz["C"] = false;

        params.m = static_cast<uint32_t>(m);
        params.k = static_cast<uint32_t>(k);
        params.n = static_cast<uint32_t>(n);
        params.inputAddr.resize(4);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));
        params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(perTokenScale.storage().data()));
        params.inputAddr[3] = static_cast<uint8_t*>(const_cast<void*>(perChannelScale.storage().data()));
    }

    static OutputType AllocOutput(CatlassKernel::MatmulParams& params)
    {
        OutputType output = GetOutputTensor({params.m, params.n}, torch::kFloat32);
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));
        return output;
    }

    static OutputType Run(
        const at::Tensor& mat1, const at::Tensor& mat2,
        const at::Tensor& perTokenScale, const at::Tensor& perChannelScale)
    {
        CatlassKernel::TParams tParams;
        CatlassKernel::MatmulParams params;
        GetKernelInfo(mat1, mat2, perTokenScale, perChannelScale, tParams, params);
        OutputType output = AllocOutput(params);
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, tParams, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif // OPTEST_ASCEND950_EPILOGUE_QUANT_MATMUL_H
