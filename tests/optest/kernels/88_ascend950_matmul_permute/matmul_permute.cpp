/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "catlass_kernel.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"
#include "jit_macros.h"

namespace CatlassKernel {

extern "C" void Ascend950MatmulPermute(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params)
{
    auto macros = JitMacroGenerator<TParams>::generate("ascend950_matmul_permute", tParams);
    auto* entry = JitCompiler::instance().getKernel("matmul_permute_impl.cpp", macros, JitKernelType::MIX);
    JIT_CHECK(entry != nullptr, "JIT load failed for matmul_permute_impl.cpp");
    entry(blockNum, stream, &params);
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
