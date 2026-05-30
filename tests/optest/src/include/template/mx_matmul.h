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

#ifndef OPTEST_MX_MATMUL_H
#define OPTEST_MX_MATMUL_H

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"
#include "type_utils.hpp"

namespace CatlassKernelWrapper {

constexpr uint32_t kMxScaleGroupNum = 32;

inline uint32_t CeilDivUint32(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

inline uint32_t RoundUp2Uint32(uint32_t v)
{
    return (v + 1U) / 2U * 2U;
}

/**
 * @brief Compute MX scale tensor shapes for the default layout mapping.
 *
 * When A is RowMajor and B is ColumnMajor (trans_a=0, trans_b=1):
 *   mx_scale_a: (m, mxScaleAlignedK / 2, 2)  → numel = m * mxScaleAlignedK
 *   mx_scale_b: (n, mxScaleAlignedK / 2, 2)  → numel = n * mxScaleAlignedK
 */
inline void ComputeMxScaleShapes(
    uint32_t m, uint32_t n, uint32_t k, uint32_t& mxScaleAlignedK, int64_t& scaleANumel, int64_t& scaleBNumel)
{
    const uint32_t mxScaleK = CeilDivUint32(k, kMxScaleGroupNum);
    mxScaleAlignedK = RoundUp2Uint32(mxScaleK);
    scaleANumel = static_cast<int64_t>(m) * static_cast<int64_t>(mxScaleAlignedK);
    scaleBNumel = static_cast<int64_t>(n) * static_cast<int64_t>(mxScaleAlignedK);
}

template <auto KernelFunc>
struct MxMatmulLike {
    using OutputType = at::Tensor;

    static void GetKernelInfo(
        const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& mx_scale_a,
        const at::Tensor& mx_scale_b, bool transA, bool transB, CatlassKernel::MatmulTParams& tParams,
        CatlassKernel::MatmulParams& params)
    {
        tParams.element["A"] = TorchDtypeToAclDtype(mat1.scalar_type());
        tParams.element["B"] = TorchDtypeToAclDtype(mat2.scalar_type());
        tParams.element["C"] = ACL_FLOAT;
        tParams.element["MX_SCALE"] = test_utils::TypeCast<std::string, aclDataType>("float8_e8m0fnu");
        tParams.transpose["A"] = transA;
        tParams.transpose["B"] = transB;
        tParams.transpose["C"] = false;
        tParams.useNz["A"] = false;
        tParams.useNz["B"] = false;
        tParams.useNz["C"] = false;

        params.inputAddr.resize(4);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));
        params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(mx_scale_a.storage().data()));
        params.inputAddr[3] = static_cast<uint8_t*>(const_cast<void*>(mx_scale_b.storage().data()));

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
        TORCH_CHECK(mat1.is_contiguous(), "mat1 must be contiguous");
        TORCH_CHECK(mat2.is_contiguous(), "mat2 must be contiguous");
        TORCH_CHECK(mx_scale_a.is_contiguous(), "mx_scale_a must be contiguous");
        TORCH_CHECK(mx_scale_b.is_contiguous(), "mx_scale_b must be contiguous");
        params.m = static_cast<uint32_t>(m);
        params.k = static_cast<uint32_t>(k1);
        params.n = static_cast<uint32_t>(n);

        uint32_t mxScaleAlignedK = 0;
        int64_t scaleANumel = 0;
        int64_t scaleBNumel = 0;
        ComputeMxScaleShapes(params.m, params.n, params.k, mxScaleAlignedK, scaleANumel, scaleBNumel);
        TORCH_CHECK(
            mx_scale_a.numel() == scaleANumel,
            "mx_scale_a must have ", scaleANumel, " elements (m=", m, ", mxScaleAlignedK=", mxScaleAlignedK,
            "), got ", mx_scale_a.numel());
        TORCH_CHECK(
            mx_scale_b.numel() == scaleBNumel,
            "mx_scale_b must have ", scaleBNumel, " elements (n=", n, ", mxScaleAlignedK=", mxScaleAlignedK,
            "), got ", mx_scale_b.numel());
    }

    static OutputType AllocOutput(CatlassKernel::MatmulParams& params)
    {
        OutputType output = GetOutputTensor({params.m, params.n}, torch::kFloat32);
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));
        return output;
    }

    static OutputType Run(
        const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& mx_scale_a,
        const at::Tensor& mx_scale_b, bool transA, bool transB)
    {
        CatlassKernel::MatmulTParams tParams;
        CatlassKernel::MatmulParams params;
        GetKernelInfo(mat1, mat2, mx_scale_a, mx_scale_b, transA, transB, tParams, params);
        OutputType output = AllocOutput(params);
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, tParams, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif
