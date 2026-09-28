#include "../common.h"
#include "catlass/catlass.hpp"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510

#include "vector_reg_utils.h"

// MaskLoadDist US/DS → CCE plds dist=1 / 2.
// Bypasses AscendNPU-IR i1→plds hardcoding dist=0 (and HIVMAVE US/DS
// enum values 24/25 which are not CCE MaskDist 1/2).
//
// One stub per UB element *byte width* (_b8/_b16/_b32/_b64). The memref
// element type only affects address arithmetic (aligned + offset);
// plds's intrinsic ABI is always __ubuf__ uint32_t*.

extern "C" {

#define MASK_LOAD_US_DS_STUBS(Suffix, ElemTy)                                          \
    __aiv__ __attribute__((always_inline)) ave_preg _mlir_ciface_mask_load_us##Suffix( \
        memref_t<__ubuf__ ElemTy, 1>* maskUb)                                          \
    {                                                                                  \
        __ubuf__ ElemTy* addr = maskUb->aligned + maskUb->offset;                      \
        __ubuf__ uint32_t* ptr = reinterpret_cast<__ubuf__ uint32_t*>(addr);           \
        vector_bool pred;                                                              \
        plds(pred, ptr, 0, US);                                                        \
        return *reinterpret_cast<ave_preg*>(&pred);                                    \
    }                                                                                  \
    __aiv__ __attribute__((always_inline)) ave_preg _mlir_ciface_mask_load_ds##Suffix( \
        memref_t<__ubuf__ ElemTy, 1>* maskUb)                                          \
    {                                                                                  \
        __ubuf__ ElemTy* addr = maskUb->aligned + maskUb->offset;                      \
        __ubuf__ uint32_t* ptr = reinterpret_cast<__ubuf__ uint32_t*>(addr);           \
        vector_bool pred;                                                              \
        plds(pred, ptr, 0, DS);                                                        \
        return *reinterpret_cast<ave_preg*>(&pred);                                    \
    }

MASK_LOAD_US_DS_STUBS(_b8, uint8_t)
MASK_LOAD_US_DS_STUBS(_b16, uint16_t)
MASK_LOAD_US_DS_STUBS(_b32, uint32_t)
MASK_LOAD_US_DS_STUBS(_b64, uint64_t)

#undef MASK_LOAD_US_DS_STUBS

} // extern "C"

#endif
