#include "../dma_common.h"

#include "catlass/epilogue/tile/tile_copy.hpp"

template <class ArchTag, typename T>
CATLASS_DEVICE void copyUbRowMajorToL1zN(
    memref_t<__ubuf__ T, 1> *src, memref_t<__cbuf__ T, 1> *dst,
    const TensorDesc2D &srcDesc, const TensorDesc4D &dstDesc) {
    auto srcTensor = makeUBTensor(src, srcDesc);
    auto dstTensor = makeL1Tensor(dst, dstDesc);
    Catlass::Epilogue::Tile::CopyUb2L1Tla<ArchTag, decltype(srcTensor),
                                     decltype(dstTensor)>{}(dstTensor,
                                                            srcTensor);
}

template <class ArchTag, typename T>
CATLASS_DEVICE void copyGmRowMajorToUbRowMajor(
    memref_t<__gm__ T, 2> *src, memref_t<__ubuf__ T, 1> *dst,
    const TensorDesc2D &srcDesc, const TensorDesc2D &dstDesc) {
    auto srcTensor = makeGMTensor(src, srcDesc);
    auto dstTensor = makeUBTensor(dst, dstDesc);
    Catlass::Epilogue::Tile::CopyGm2UbTla<ArchTag, decltype(srcTensor),
                                     decltype(dstTensor)>{}(dstTensor,
                                                            srcTensor);
}

template <class ArchTag, typename T>
CATLASS_DEVICE void copyUbRowMajorToGmRowMajor(
    memref_t<__ubuf__ T, 1> *src, memref_t<__gm__ T, 2> *dst,
    const TensorDesc2D &srcDesc, const TensorDesc2D &dstDesc) {
    auto srcTensor = makeUBTensor(src, srcDesc);
    auto dstTensor = makeGMTensor(dst, dstDesc);
    Catlass::Epilogue::Tile::CopyUb2GmTla<ArchTag, decltype(srcTensor),
                                     decltype(dstTensor)>{}(dstTensor,
                                                            srcTensor);
}


extern "C" {
#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) ||                    \
     (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))

#define REGISTER_UB_ROW_TO_L1_ZN(DType)                                                                          \
[aicore] __attribute__((always_inline))                                                                              \
void _mlir_ciface_copy_ub_row_major_to_l1_zN_##DType(                                                            \
    memref_t<__ubuf__ DType, 1> *src, memref_t<__cbuf__ DType, 1> *dst,                                              \
    int64_t srcShape0, int64_t srcShape1, int64_t srcStride0,                                                        \
    int64_t srcStride1, int64_t srcCoord0, int64_t srcCoord1,                                                        \
    int64_t srcOrgShape0, int64_t srcOrgShape1, int64_t dstShape0,                                                   \
    int64_t dstShape1, int64_t dstShape2, int64_t dstShape3,                                                         \
    int64_t dstStride0, int64_t dstStride1, int64_t dstStride2,                                                      \
    int64_t dstStride3, int64_t dstCoord0, int64_t dstCoord1,                                                        \
    int64_t dstOrgShape0, int64_t dstOrgShape1) {                                                                    \
    copyUbRowMajorToL1zN<Catlass::Arch::Ascend950, DType>(                                                       \
        src, dst,                                                                                                    \
        TensorDesc2D{(uint32_t)srcShape0, (uint32_t)srcShape1, srcStride0, srcStride1,                               \
                         (uint32_t)srcCoord0, (uint32_t)srcCoord1, (uint32_t)srcOrgShape0, (uint32_t)srcOrgShape1},  \
        TensorDesc4D{(uint32_t)dstShape0, (uint32_t)dstShape1, (uint32_t)dstShape2, (uint32_t)dstShape3, dstStride0, \
                       dstStride1, dstStride2, dstStride3, (uint32_t)dstCoord0, (uint32_t)dstCoord1,                 \
                       (uint32_t)dstOrgShape0, (uint32_t)dstOrgShape1});                                             \
}

REGISTER_UB_ROW_TO_L1_ZN(float)
REGISTER_UB_ROW_TO_L1_ZN(half)
REGISTER_UB_ROW_TO_L1_ZN(bf16)

#define REGISTER_GM_ROW_TO_UB_ROW(DType)                                                                      \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_gm_row_major_to_ub_row_major_##DType(      \
        memref_t<__gm__ DType, 2>* src, memref_t<__ubuf__ DType, 1>* dst, int64_t srcShape0, int64_t srcShape1, \
        int64_t srcStride0, int64_t srcStride1, int64_t srcCoord0, int64_t srcCoord1, int64_t srcOrgShape0,     \
        int64_t srcOrgShape1, int64_t dstShape0, int64_t dstShape1, int64_t dstStride0, int64_t dstStride1,     \
        int64_t dstCoord0, int64_t dstCoord1, int64_t dstOrgShape0, int64_t dstOrgShape1)                       \
    {                                                                                                           \
        copyGmRowMajorToUbRowMajor<Catlass::Arch::Ascend950, DType>(                                          \
            src, dst,                                                                                           \
            TensorDesc2D{                                                                                       \
                (uint32_t)srcShape0, (uint32_t)srcShape1, srcStride0, srcStride1, (uint32_t)srcCoord0,          \
                (uint32_t)srcCoord1, (uint32_t)srcOrgShape0, (uint32_t)srcOrgShape1},                           \
            TensorDesc2D{                                                                                       \
                (uint32_t)dstShape0, (uint32_t)dstShape1, dstStride0, dstStride1, (uint32_t)dstCoord0,          \
                (uint32_t)dstCoord1, (uint32_t)dstOrgShape0, (uint32_t)dstOrgShape1});                          \
    }

REGISTER_GM_ROW_TO_UB_ROW(float)
REGISTER_GM_ROW_TO_UB_ROW(half)
REGISTER_GM_ROW_TO_UB_ROW(bf16)
REGISTER_GM_ROW_TO_UB_ROW(int32_t)
REGISTER_GM_ROW_TO_UB_ROW(int16_t)
REGISTER_GM_ROW_TO_UB_ROW(int8_t)

#define REGISTER_UB_ROW_TO_GM_ROW(DType)                                                                      \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_ub_row_major_to_gm_row_major_##DType(      \
        memref_t<__ubuf__ DType, 1>* src, memref_t<__gm__ DType, 2>* dst, int64_t srcShape0, int64_t srcShape1, \
        int64_t srcStride0, int64_t srcStride1, int64_t srcCoord0, int64_t srcCoord1, int64_t srcOrgShape0,     \
        int64_t srcOrgShape1, int64_t dstShape0, int64_t dstShape1, int64_t dstStride0, int64_t dstStride1,     \
        int64_t dstCoord0, int64_t dstCoord1, int64_t dstOrgShape0, int64_t dstOrgShape1)                       \
    {                                                                                                           \
        copyUbRowMajorToGmRowMajor<Catlass::Arch::Ascend950, DType>(                                          \
            src, dst,                                                                                           \
            TensorDesc2D{                                                                                       \
                (uint32_t)srcShape0, (uint32_t)srcShape1, srcStride0, srcStride1, (uint32_t)srcCoord0,          \
                (uint32_t)srcCoord1, (uint32_t)srcOrgShape0, (uint32_t)srcOrgShape1},                           \
            TensorDesc2D{                                                                                       \
                (uint32_t)dstShape0, (uint32_t)dstShape1, dstStride0, dstStride1, (uint32_t)dstCoord0,          \
                (uint32_t)dstCoord1, (uint32_t)dstOrgShape0, (uint32_t)dstOrgShape1});                          \
    }

REGISTER_UB_ROW_TO_GM_ROW(float)
REGISTER_UB_ROW_TO_GM_ROW(half)
REGISTER_UB_ROW_TO_GM_ROW(bf16)
REGISTER_UB_ROW_TO_GM_ROW(int32_t)
REGISTER_UB_ROW_TO_GM_ROW(int16_t)
REGISTER_UB_ROW_TO_GM_ROW(int8_t)

#endif
}
