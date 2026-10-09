#include "catlass/catlass.hpp"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510

#include "vector_reg_utils.h"

// Reg→Reg Gather via CCE vselr (AscendC Reg::Gather 1:1).
// Index is unsigned and same byte-width as the element type.
// float has no direct vselr overload → bitcast through uint32_t (AscendC style).

namespace {

__aiv__ __attribute__((always_inline)) VectorReg<float> gatherRegFloat(VectorReg<float> src, VectorReg<uint32_t> index)
{
    VectorReg<float> dst;
    vselr(reinterpret_cast<VectorReg<uint32_t>&>(dst), reinterpret_cast<VectorReg<uint32_t>&>(src), index);
    return dst;
}

template <typename T, typename Idx>
__aiv__ __attribute__((always_inline)) VectorReg<T> gatherRegImpl(VectorReg<T> src, VectorReg<Idx> index)
{
    VectorReg<T> dst;
    vselr(dst, src, index);
    return dst;
}

#define DECLARE_GATHER_REG(dtype, idx_dtype)                                    \
    __aiv__ __attribute__((always_inline)) VectorReg<dtype> gather_reg_##dtype( \
        VectorReg<dtype> src, VectorReg<idx_dtype> index)

#define REGISTER_GATHER_REG(dtype, idx_dtype)               \
    DECLARE_GATHER_REG(dtype, idx_dtype)                    \
    {                                                       \
        return gatherRegImpl<dtype, idx_dtype>(src, index); \
    }

} // namespace

extern "C" {
REGISTER_GATHER_REG(int8_t, uint8_t);
REGISTER_GATHER_REG(uint8_t, uint8_t);
REGISTER_GATHER_REG(half, uint16_t);
REGISTER_GATHER_REG(bfloat16_t, uint16_t);
REGISTER_GATHER_REG(int16_t, uint16_t);
REGISTER_GATHER_REG(uint16_t, uint16_t);
DECLARE_GATHER_REG(float, uint32_t)
{
    return gatherRegFloat(src, index);
}
REGISTER_GATHER_REG(int32_t, uint32_t);
REGISTER_GATHER_REG(uint32_t, uint32_t);
}

#endif
