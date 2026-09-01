#include "catlass/catlass.hpp"
#include "../common.h"

namespace Catlass::Gemm {

template <class ElementA, class ElementB, class ElementC>
__aicore__ __attribute__((always_inline)) void Mmad(
    __ca__ ElementA* a, __cb__ ElementB* b, __cc__ ElementC* c, uint32_t m, uint32_t n, uint32_t k, bool initC = true,
    uint8_t unitFlag = 0)
{
    AscendC::LocalTensor<ElementA> l0a{
        AscendC::TPosition::A2, (uint32_t) reinterpret_cast<int64_t>(a), (uint32_t)(m * k)};
    AscendC::LocalTensor<ElementB> l0b{
        AscendC::TPosition::B2, (uint32_t) reinterpret_cast<int64_t>(b), (uint32_t)(n * k)};
    AscendC::LocalTensor<ElementC> l0c{
        AscendC::TPosition::CO1, (uint32_t) reinterpret_cast<int64_t>(c), (uint32_t)(m * n)};

    AscendC::MmadParams mmadParams;
    mmadParams.m = m;
    mmadParams.n = n;
    mmadParams.k = k;
    mmadParams.unitFlag = unitFlag;
    mmadParams.cmatrixInitVal = initC;
    mmadParams.disableGemv = true;

    AscendC::Mmad(l0c, l0a, l0b, mmadParams);

    const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
    if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
        AscendC::PipeBarrier<PIPE_M>();
    }
}

} // namespace Catlass::Gemm

extern "C" {
__aicore__
    __attribute__((always_inline))
    // MLIR lowering passes memref<?xT, strided<[?], offset:?>, #ca/cb/cc> (rank 1); only GM uses rank 2.
    // Cube MMAD on dav-c310 uses fp32 L0C (__cc__); f16/bf16 outputs use copy_l0c_to_gm_RowMajor_* stubs.
    void
    _mlir_ciface_mmad_float_float_float(
        memref_t<__ca__ float, 1>* a, memref_t<__cb__ float, 1>* b, memref_t<__cc__ float, 1>* c, int64_t m, int64_t n,
        int64_t k, bool initC = true, uint8_t unitFlag = 0)
{
    Catlass::Gemm::Mmad<float, float, float>(
        a->aligned + a->offset, b->aligned + b->offset, c->aligned + c->offset, m, n, k, initC, unitFlag);
}

__aicore__ __attribute__((always_inline)) void _mlir_ciface_mmad_half_half_float(
    memref_t<__ca__ half, 1>* a, memref_t<__cb__ half, 1>* b, memref_t<__cc__ float, 1>* c, int64_t m, int64_t n,
    int64_t k, bool initC = true, uint8_t unitFlag = 0)
{
    Catlass::Gemm::Mmad<half, half, float>(
        a->aligned + a->offset, b->aligned + b->offset, c->aligned + c->offset, m, n, k, initC, unitFlag);
}

__aicore__ __attribute__((always_inline)) void _mlir_ciface_mmad_bf16_bf16_float(
    memref_t<__ca__ bfloat16_t, 1>* a, memref_t<__cb__ bfloat16_t, 1>* b, memref_t<__cc__ float, 1>* c, int64_t m,
    int64_t n, int64_t k, bool initC = true, uint8_t unitFlag = 0)
{
    Catlass::Gemm::Mmad<bfloat16_t, bfloat16_t, float>(
        a->aligned + a->offset, b->aligned + b->offset, c->aligned + c->offset, m, n, k, initC, unitFlag);
}

// Integer route: int8 operands accumulate into an int32 L0C (not fp32). This is
// the non-MX `mad` path, same intrinsic as the float routes -- only the L0C
// element type differs.
__aicore__ __attribute__((always_inline)) void _mlir_ciface_mmad_int8_int8_int32(
    memref_t<__ca__ int8_t, 1>* a, memref_t<__cb__ int8_t, 1>* b, memref_t<__cc__ int32_t, 1>* c, int64_t m, int64_t n,
    int64_t k, bool initC = true, uint8_t unitFlag = 0)
{
    Catlass::Gemm::Mmad<int8_t, int8_t, int32_t>(
        a->aligned + a->offset, b->aligned + b->offset, c->aligned + c->offset, m, n, k, initC, unitFlag);
}

#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))
// The PLAIN fp8 routes, not the mx_fp8_* ones -- MmadCal only takes the mad_mx
// path for mx_fp8_*/fp4 operands, so these land on the same `mad` intrinsic as
// the f16/bf16/f32 routes and need no MX scale setup. A and B formats are
// independent, hence four symbols.
//
// The MX fp8 operand types need re-declaring at global scope under the same
// spelling: AscendC::mx_fp8_e4m3_t contains a `::`, which cannot be pasted into
// a symbol name, and these macros paste the type token into both the symbol and
// the signature. The fp4 types are global already and are used as-is.
using mx_fp8_e4m3_t = AscendC::mx_fp8_e4m3_t;
using mx_fp8_e5m2_t = AscendC::mx_fp8_e5m2_t;

#define REGISTER_MMAD_FP8(TypeA, TypeB)                                                                        \
    __aicore__ __attribute__((always_inline)) void _mlir_ciface_mmad_##TypeA##_##TypeB##_float(                \
        memref_t<__ca__ TypeA, 1>* a, memref_t<__cb__ TypeB, 1>* b, memref_t<__cc__ float, 1>* c, int64_t m,   \
        int64_t n, int64_t k, bool initC = true, uint8_t unitFlag = 0)                                         \
    {                                                                                                          \
        Catlass::Gemm::Mmad<TypeA, TypeB, float>(                                                              \
            a->aligned + a->offset, b->aligned + b->offset, c->aligned + c->offset, m, n, k, initC, unitFlag); \
    }

REGISTER_MMAD_FP8(fp8_e4m3fn_t, fp8_e4m3fn_t)
REGISTER_MMAD_FP8(fp8_e5m2_t, fp8_e5m2_t)
REGISTER_MMAD_FP8(fp8_e4m3fn_t, fp8_e5m2_t)
REGISTER_MMAD_FP8(fp8_e5m2_t, fp8_e4m3fn_t)

// MX routes. The L0 tiles were loaded as mx_fp8_*, which is what makes MmadCal
// pick mad_mx; the scale itself was consumed by that load, so this call has no
// scale operand and is otherwise identical to the plain fp8 one.
#define REGISTER_MMAD_MXFP8(TypeA, TypeB)                                                                      \
    __aicore__ __attribute__((always_inline)) void _mlir_ciface_mmad_##TypeA##_##TypeB##_float(                \
        memref_t<__ca__ TypeA, 1>* a, memref_t<__cb__ TypeB, 1>* b, memref_t<__cc__ float, 1>* c, int64_t m,   \
        int64_t n, int64_t k, bool initC = true, uint8_t unitFlag = 0)                                         \
    {                                                                                                          \
        Catlass::Gemm::Mmad<TypeA, TypeB, float>(                                                              \
            a->aligned + a->offset, b->aligned + b->offset, c->aligned + c->offset, m, n, k, initC, unitFlag); \
    }

REGISTER_MMAD_MXFP8(mx_fp8_e4m3_t, mx_fp8_e4m3_t)
REGISTER_MMAD_MXFP8(mx_fp8_e5m2_t, mx_fp8_e5m2_t)
REGISTER_MMAD_MXFP8(mx_fp8_e4m3_t, mx_fp8_e5m2_t)
REGISTER_MMAD_MXFP8(mx_fp8_e5m2_t, mx_fp8_e4m3_t)

// MX fp4. Unlike fp8, the L0 element type is the fp4 type itself -- MmadCal's
// isMx list matches fp4x2_e2m1_t/fp4x2_e1m2_t directly, with no mx_* variant.
// The tiles arrive as int8_t storage (two fp4 per byte) and are reinterpreted.
#define REGISTER_MMAD_MXFP4(TypeA, TypeB)                                                                      \
    __aicore__ __attribute__((always_inline)) void _mlir_ciface_mmad_##TypeA##_##TypeB##_float(                \
        memref_t<__ca__ int8_t, 1>* a, memref_t<__cb__ int8_t, 1>* b, memref_t<__cc__ float, 1>* c, int64_t m, \
        int64_t n, int64_t k, bool initC = true, uint8_t unitFlag = 0)                                         \
    {                                                                                                          \
        Catlass::Gemm::Mmad<TypeA, TypeB, float>(                                                              \
            reinterpret_cast<__ca__ TypeA*>(a->aligned + a->offset),                                           \
            reinterpret_cast<__cb__ TypeB*>(b->aligned + b->offset), c->aligned + c->offset, m, n, k, initC,   \
            unitFlag);                                                                                         \
    }

REGISTER_MMAD_MXFP4(float4_e2m1x2_t, float4_e2m1x2_t)
REGISTER_MMAD_MXFP4(float4_e1m2x2_t, float4_e1m2x2_t)
REGISTER_MMAD_MXFP4(float4_e2m1x2_t, float4_e1m2x2_t)
REGISTER_MMAD_MXFP4(float4_e1m2x2_t, float4_e2m1x2_t)
#endif
}
