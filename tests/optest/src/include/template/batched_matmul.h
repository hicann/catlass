#ifndef OPTEST_BATCHED_MATMUL_H
#define OPTEST_BATCHED_MATMUL_H

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_jit.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"
#include "type_utils.hpp"

namespace CatlassKernelWrapper {

using KernelFn = void (*)(const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::MatmulParams&);

template <KernelFn KernelFunc>
struct BatchedMatmulLike {
    using OutputType = at::Tensor;

    static void GetKernelInfo(
        const at::Tensor& mat1, const at::Tensor& mat2,
        const c10::ScalarType& outDType, bool transA, bool transB,
        bool useNzA, bool useNzB,
        CatlassKernel::TParams& tParams,
        CatlassKernel::MatmulParams& params)
    {
        auto aclType = TorchDtypeToAclDtype(mat1.scalar_type());
        tParams.element["A"] = aclType;
        tParams.element["B"] = aclType;
        tParams.element["C"] = TorchDtypeToAclDtype(outDType);
        tParams.transpose["A"] = transA;
        tParams.transpose["B"] = transB;
        tParams.transpose["C"] = false;
        tParams.useNz["A"] = useNzA;
        tParams.useNz["B"] = useNzB;
        tParams.useNz["C"] = false;

        TORCH_CHECK(mat1.dim() == 3, "mat1 must be 3-D (B, M, K)");
        TORCH_CHECK(mat2.dim() == 3, "mat2 must be 3-D (B, K, N)");
        auto B = mat1.size(0);
        TORCH_CHECK(mat2.size(0) == B, "batch dim mismatch");

        params.inputAddr.resize(2);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));

        int64_t m, k1, k2, n;
        if (transA) {
            m = mat1.size(2); k1 = mat1.size(1);
        } else {
            m = mat1.size(1); k1 = mat1.size(2);
        }
        if (transB) {
            k2 = mat2.size(2); n = mat2.size(1);
        } else {
            k2 = mat2.size(1); n = mat2.size(2);
        }
        TORCH_CHECK(k1 == k2, "mat1 and mat2 shapes cannot be multiplied");
        params.m = static_cast<uint32_t>(m);
        params.k = static_cast<uint32_t>(k1);
        params.n = static_cast<uint32_t>(n);
        params.batch = static_cast<uint32_t>(B);
    }

    static OutputType AllocOutput(
        const CatlassKernel::TParams& tParams, CatlassKernel::MatmulParams& params)
    {
        OutputType output = GetOutputTensor(
            {static_cast<int64_t>(params.batch), params.m, params.n},
            AclDtypeToTorchDtype(tParams.elem("C")));
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));
        return output;
    }

    static OutputType Run(
        const at::Tensor& mat1, const at::Tensor& mat2,
        const c10::ScalarType& outDType, bool transA, bool transB,
        bool useNzA, bool useNzB)
    {
        CatlassKernel::TParams tParams;
        CatlassKernel::MatmulParams  params;
        GetKernelInfo(mat1, mat2, outDType, transA, transB, useNzA, useNzB, tParams, params);
        OutputType output = AllocOutput(tParams, params);
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, aicCoreNum, stream, tParams, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper

#endif
