#include "catlass_kernel.h"
#include "jit_compiler.h"
#include "jit_macro_generator.h"

namespace CatlassKernel {

extern "C" void BigMatmulTLA(
    const uint32_t blockNum, aclrtStream stream, const TParams& tParams, const MatmulParams& params)
{
    auto* entry = JitCompiler::instance().getKernel(
        "big_matmul_tla_impl.cpp", JitMacroGenerator<TParams>::generate("big_matmul_tla", tParams), JitKernelType::AIC);
    if (entry) {
        entry(blockNum, stream, &params);
    }
    aclrtSynchronizeStream(stream);
}

} // namespace CatlassKernel
