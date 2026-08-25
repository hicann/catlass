#ifndef OPTEST_GEMM_H
#define OPTEST_GEMM_H

#include <torch/torch.h>
#include <tiling/platform/platform_ascendc.h>

#include "catlass_kernel_jit.h"
#include "common/run_npu_func.h"
#include "torch_utils.h"
#include "type_utils.hpp"

namespace CatlassKernelWrapper {

using GemmKernelFn = void (*)(const uint32_t, aclrtStream, const CatlassKernel::TParams&, const CatlassKernel::GemmParams&);

template <GemmKernelFn KernelFunc, bool IsGemv = false>
struct GemmLike {
    using OutputType = at::Tensor;
    static OutputType Run(const at::Tensor& mat1, const at::Tensor& mat2,
        const at::Tensor& matC,
        const c10::ScalarType& outDType, double alpha, double beta,
        bool transA = false, bool transB = false, bool formatA = false, bool formatB = false)
    {
        CatlassKernel::TParams tParams;
        CatlassKernel::GemmParams params;
        tParams.element["A"] = TorchDtypeToAclDtype(mat1.scalar_type());
        tParams.element["B"] = TorchDtypeToAclDtype(mat2.scalar_type());
        tParams.element["C"] = TorchDtypeToAclDtype(outDType);
        tParams.transpose["A"] = transA;
        tParams.transpose["B"] = transB;
        tParams.useNz["A"] = formatA;
        tParams.useNz["B"] = formatB;
        params.alpha = static_cast<float>(alpha);
        params.beta  = static_cast<float>(beta);

        if constexpr (IsGemv) {
            TORCH_CHECK(mat2.dim() == 1, "gemv: vecX must be 1-D");
            params.n = static_cast<uint32_t>(mat1.size(1));
            params.m = static_cast<uint32_t>(mat1.size(0));
            params.k = params.n;
            TORCH_CHECK(static_cast<uint32_t>(mat2.size(0)) == params.k, "gemv: size mismatch");
        } else {
            int64_t m, k1, k2, n;
            if (transA) { m = mat1.size(1); k1 = mat1.size(0); }
            else        { m = mat1.size(0); k1 = mat1.size(1); }
            if (transB) { k2 = mat2.size(1); n = mat2.size(0); }
            else        { k2 = mat2.size(0); n = mat2.size(1); }
            TORCH_CHECK(k1 == k2, "shape mismatch");
            params.m = static_cast<uint32_t>(m);
            params.k = static_cast<uint32_t>(k1);
            params.n = static_cast<uint32_t>(n);
        }

        params.inputAddr.resize(3);
        params.inputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(mat1.storage().data()));
        params.inputAddr[1] = static_cast<uint8_t*>(const_cast<void*>(mat2.storage().data()));

        // Matrix C for epilogue: D = alpha * (A*B) + beta * C
        // inputAddr[2] holds the residual matrix C; the kernel copies it into outputAddr[0]
        // internally before using outputAddr[0] as the epilogue input/output.
        auto torchDtype = AclDtypeToTorchDtype(tParams.elem("C"));
        auto outShape = IsGemv ? std::vector<int64_t>{params.m} : std::vector<int64_t>{params.m, params.n};
        params.inputAddr[2] = static_cast<uint8_t*>(const_cast<void*>(matC.storage().data()));

        // Output tensor: the kernel copies inputAddr[2] → outputAddr[0] then writes D there
        OutputType output = GetOutputTensor(outShape, torchDtype);
        params.outputAddr.resize(1);
        params.outputAddr[0] = static_cast<uint8_t*>(const_cast<void*>(output.storage().data()));

        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        uint32_t coreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
        RUN_NPU_FUNC(KernelFunc, coreNum, stream, tParams, params);
        return output;
    }
};

} // namespace CatlassKernelWrapper
#endif
