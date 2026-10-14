/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

#include "catlass_kernel.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"

namespace CatlassKernel {

extern "C" void Ascend950BatchedMatmulSoftmax(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params)
{
    auto macros = JitMacroGenerator<TParams>::generate("ascend950_batched_matmul_softmax", tParams);
    auto* entry = JitCompiler::instance().getKernel(
        "ascend950_batched_matmul_softmax_impl.cpp", macros, JitKernelType::MIX);
    if (entry) {
        entry(blockNum, stream, &params);
    }
    aclError status = aclrtSynchronizeStream(stream);
    if (status != ACL_SUCCESS) {
        JIT_LOGE("Ascend950BatchedMatmulSoftmax stream synchronization failed: %d", status);
        JIT_THROW("Ascend950BatchedMatmulSoftmax stream synchronization failed: " + std::to_string(status));
    }
}

} // namespace CatlassKernel
