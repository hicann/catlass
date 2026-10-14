/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

#ifndef OPTEST_BATCHED_MATMUL_SOFTMAX_H
#define OPTEST_BATCHED_MATMUL_SOFTMAX_H

#include <algorithm>
#include <cstdint>
#include <limits>

#include <acl/acl.h>
#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass/gemm/kernel/batched_matmul_softmax_tla_tuning.hpp"

#include "catlass_kernel_jit.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"
#include "type_utils.hpp"

namespace CatlassKernelWrapper {

using BatchedMatmulSoftmaxKernelFn =
    void (*)(const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::MatmulParams&);

template <BatchedMatmulSoftmaxKernelFn KernelFunc>
struct BatchedMatmulSoftmaxLike {
    using OutputType = at::Tensor;

    static OutputType Run(const at::Tensor& mat1, const at::Tensor& mat2, const c10::ScalarType& outDType)
    {
        TORCH_CHECK(mat1.dim() == 3 && mat2.dim() == 3, "BatchedMatmulSoftmax expects 3D inputs");
        TORCH_CHECK(mat1.device().type() == c10::DeviceType::PrivateUse1, "A must be on NPU");
        TORCH_CHECK(mat2.device().type() == c10::DeviceType::PrivateUse1, "B must be on NPU");
        TORCH_CHECK(mat1.device() == mat2.device(), "A and B must be on the same NPU device");
        TORCH_CHECK(mat1.size(0) == mat2.size(0), "batch dimensions must match");
        TORCH_CHECK(mat1.size(2) == mat2.size(1), "K dimensions must match");
        TORCH_CHECK(mat1.scalar_type() == c10::ScalarType::Half, "A must be float16");
        TORCH_CHECK(mat2.scalar_type() == c10::ScalarType::Half, "B must be float16");
        TORCH_CHECK(outDType == c10::ScalarType::Half, "S must be float16");
        TORCH_CHECK(mat1.is_contiguous(), "A must be contiguous RowMajor");
        TORCH_CHECK(mat2.stride(0) == mat2.size(1) * mat2.size(2) &&
                        mat2.stride(1) == 1 && mat2.stride(2) == mat2.size(1),
                    "B must be ColumnMajor with strides (K*N, 1, K)");

        int64_t batch = mat1.size(0);
        int64_t m = mat1.size(1);
        int64_t k = mat1.size(2);
        int64_t n = mat2.size(2);
        TORCH_CHECK(
            batch <= std::numeric_limits<uint32_t>::max() && m <= std::numeric_limits<uint32_t>::max() &&
                k <= std::numeric_limits<uint32_t>::max() && n <= std::numeric_limits<uint32_t>::max(),
            "BatchedMatmulSoftmax dimensions must fit the kernel's 32-bit shape fields");
        if (batch == 0 || m == 0 || n == 0) {
            return GetOutputTensor({batch, m, n}, outDType);
        }
        if (k == 0) {
            OutputType output = GetOutputTensor({batch, m, n}, outDType);
            output.fill_(1.0 / static_cast<double>(n));
            return output;
        }
        TORCH_CHECK(
            Catlass::Gemm::Kernel::BatchedMatmulSoftmaxTuning::IsSoftmaxNSupported(static_cast<uint64_t>(n)),
            "BatchedMatmulSoftmax N is too large for the available UB; supported N must fit the Softmax workspace");
        const uint64_t outputRows = static_cast<uint64_t>(batch) * static_cast<uint64_t>(m);
        TORCH_CHECK(
            outputRows <= std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(n) / sizeof(at::Half),
            "BatchedMatmulSoftmax output is too large");

        CatlassKernel::TParams tParams;
        tParams.element["A"] = TorchDtypeToAclDtype(mat1.scalar_type());
        tParams.element["B"] = TorchDtypeToAclDtype(mat2.scalar_type());
        tParams.element["C"] = TorchDtypeToAclDtype(outDType);
        tParams.transpose["A"] = false;
        tParams.transpose["B"] = true;
        tParams.transpose["C"] = false;
        tParams.useNz["A"] = false;
        tParams.useNz["B"] = false;
        tParams.useNz["C"] = false;

        CatlassKernel::MatmulParams params;
        auto aStorageOffset = mat1.storage_offset() * mat1.element_size();
        auto bStorageOffset = mat2.storage_offset() * mat2.element_size();
        params.batch = static_cast<uint32_t>(mat1.size(0));
        params.m = static_cast<uint32_t>(mat1.size(1));
        params.k = static_cast<uint32_t>(mat1.size(2));
        params.n = static_cast<uint32_t>(mat2.size(2));
        params.inputAddr.resize(2);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data())) + aStorageOffset;
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data())) + bStorageOffset;

        OutputType output = GetOutputTensor(
            {params.batch, params.m, params.n}, AclDtypeToTorchDtype(tParams.elem("C")));
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));

        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);

        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        uint32_t blockNum =
            Catlass::Gemm::Kernel::SelectBatchedMatmulSoftmaxBlockNum(params.batch, params.m, params.n, aicCoreNum);
        RUN_NPU_FUNC(KernelFunc, blockNum, stream, tParams, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif
