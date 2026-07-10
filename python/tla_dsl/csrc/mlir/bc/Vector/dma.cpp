#include "../dma_common.h"

#include "catlass/epilogue/tile/tile_copy.hpp"

template <class ArchTag, typename T>
CATLASS_DEVICE void copyUbufRowMajorToCbufzN(
    memref_t<__ubuf__ T, 1> *src, memref_t<__cbuf__ T, 1> *dst,
    const TensorDesc2D &srcDesc, const TensorDesc4D &dstDesc) {
    auto srcTensor = makeUBTensor(src, srcDesc);
    auto dstTensor = makeL1Tensor(dst, dstDesc);
    Catlass::Epilogue::Tile::CopyUb2L1Tla<ArchTag, decltype(srcTensor),
                                     decltype(dstTensor)>{}(dstTensor,
                                                            srcTensor);
}


extern "C" {
#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) ||                    \
     (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))

#define REGISTER_UBUF_ROW_TO_CBUF_ZN(DType)                                                                          \
[aicore] __attribute__((always_inline))                                                                              \
void _mlir_ciface_copy_ubuf_row_major_to_cbuf_zN_##DType(                                                            \
    memref_t<__ubuf__ DType, 1> *src, memref_t<__cbuf__ DType, 1> *dst,                                              \
    int64_t srcShape0, int64_t srcShape1, int64_t srcStride0,                                                        \
    int64_t srcStride1, int64_t srcCoord0, int64_t srcCoord1,                                                        \
    int64_t srcOrgShape0, int64_t srcOrgShape1, int64_t dstShape0,                                                   \
    int64_t dstShape1, int64_t dstShape2, int64_t dstShape3,                                                         \
    int64_t dstStride0, int64_t dstStride1, int64_t dstStride2,                                                      \
    int64_t dstStride3, int64_t dstCoord0, int64_t dstCoord1,                                                        \
    int64_t dstOrgShape0, int64_t dstOrgShape1) {                                                                    \
    copyUbufRowMajorToCbufzN<Catlass::Arch::Ascend950, DType>(                                                       \
        src, dst,                                                                                                    \
        TensorDesc2D{(uint32_t)srcShape0, (uint32_t)srcShape1, srcStride0, srcStride1,                               \
                         (uint32_t)srcCoord0, (uint32_t)srcCoord1, (uint32_t)srcOrgShape0, (uint32_t)srcOrgShape1},  \
        TensorDesc4D{(uint32_t)dstShape0, (uint32_t)dstShape1, (uint32_t)dstShape2, (uint32_t)dstShape3, dstStride0, \
                       dstStride1, dstStride2, dstStride3, (uint32_t)dstCoord0, (uint32_t)dstCoord1,                 \
                       (uint32_t)dstOrgShape0, (uint32_t)dstOrgShape1});                                             \
}

REGISTER_UBUF_ROW_TO_CBUF_ZN(float)
REGISTER_UBUF_ROW_TO_CBUF_ZN(half)
REGISTER_UBUF_ROW_TO_CBUF_ZN(bf16)

#endif
}
