#include "../dma_common.h"

#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"
#include "catlass/gemm/tile/copy_l0c_to_ub.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

template <class ArchTag, typename T>
CATLASS_DEVICE void copyGmRowMajorToCbufzN(
    memref_t<__gm__ T, 2> *src, memref_t<__cbuf__ T, 1> *dst,
    const TensorDesc2D &srcDesc, const TensorDesc4D &dstDesc) {
    auto srcTensor = makeGMTensor(src, srcDesc);
    auto dstTensor = makeL1Tensor(dst, dstDesc);
    Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(srcTensor),
                                     decltype(dstTensor)>{}(dstTensor,
                                                            srcTensor);
}

template <class ArchTag, typename T>
CATLASS_DEVICE void copyGmColumnMajorToCbufnZ(
    memref_t<__gm__ T, 2> *src, memref_t<__cbuf__ T, 1> *dst,
    const TensorDesc2D &srcDesc, const TensorDesc4D &dstDesc) {
    auto srcTensor = makeGMTensorColumnMajor(src, srcDesc);
    auto dstTensor = makeL1nZTensor(dst, dstDesc);
    Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(srcTensor),
                                     decltype(dstTensor)>{}(dstTensor,
                                                            srcTensor);
}

template <class ArchTag, typename T>
CATLASS_DEVICE void copyCbufzNToCazN(
    memref_t<__cbuf__ T, 1> *src, memref_t<__ca__ T, 1> *dst,
    const TensorDesc4D &srcDesc, const TensorDesc4D &dstDesc) {
    auto srcTensor = makeL1Tensor(src, srcDesc);
    auto dstTensor = makeL0ATensor(dst, dstDesc);
    Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(srcTensor),
                                     decltype(dstTensor)>{}(dstTensor,
                                                            srcTensor);
}

template <class ArchTag, typename T>
CATLASS_DEVICE void copyCbufnZToCazN(
    memref_t<__cbuf__ T, 1> *src, memref_t<__ca__ T, 1> *dst,
    const TensorDesc4D &srcDesc, const TensorDesc4D &dstDesc) {
    auto srcTensor = makeL1nZTensor(src, srcDesc);
    auto dstTensor = makeL0ATensor(dst, dstDesc);
    Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(srcTensor),
                                     decltype(dstTensor)>{}(dstTensor,
                                                            srcTensor);
}

template <class ArchTag, typename T>
CATLASS_DEVICE void copyCbufzNToCbnZ(
    memref_t<__cbuf__ T, 1> *src, memref_t<__cb__ T, 1> *dst,
    const TensorDesc4D &srcDesc, const TensorDesc4D &dstDesc) {
    auto srcTensor = makeL1Tensor(src, srcDesc);
    auto dstTensor = makeL0BTensor(dst, dstDesc);
    Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(srcTensor),
                                     decltype(dstTensor)>{}(dstTensor,
                                                            srcTensor);
}

template <class ArchTag, typename T>
CATLASS_DEVICE void copyCbufnZToCbnZ(
    memref_t<__cbuf__ T, 1> *src, memref_t<__cb__ T, 1> *dst,
    const TensorDesc4D &srcDesc, const TensorDesc4D &dstDesc) {
    auto srcTensor = makeL1nZTensor(src, srcDesc);
    auto dstTensor = makeL0BTensor(dst, dstDesc);
    Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(srcTensor),
                                     decltype(dstTensor)>{}(dstTensor,
                                                            srcTensor);
}

template <class ArchTag, typename T>
CATLASS_DEVICE void copyCcToGmRowMajor(
    memref_t<__cc__ T, 1> *src, memref_t<__gm__ T, 2> *dst,
    const TensorDesc4D &srcDesc, const TensorDesc2D &dstDesc, uint8_t unitFlag) {
    auto srcTensor = makeL0CTensor(src, srcDesc);
    auto dstTensor = makeGMTensor(dst, dstDesc);
    Catlass::Gemm::Tile::CopyL0CToGmTla<ArchTag, decltype(srcTensor),
                                        decltype(dstTensor)>{}(dstTensor,
                                                               srcTensor, unitFlag);
}

// L0C holds fp32 MMAD accumulator; cast to fp16/bf16 when writing GM row-major (Ascend950 fixpipe).
template <class ArchTag, typename ElementDst>
CATLASS_DEVICE void copyCcFloatToGmRowMajorCast(
    memref_t<__cc__ float, 1> *src, memref_t<__gm__ ElementDst, 2> *dst,
    const TensorDesc4D &srcDesc, const TensorDesc2D &dstDesc, uint8_t unitFlag) {
    auto srcTensor = makeL0CTensor<float>(src, srcDesc);
    auto dstTensor = makeGMTensor(dst, dstDesc);
    Catlass::Gemm::Tile::CopyL0CToGmTla<ArchTag, decltype(srcTensor),
                                        decltype(dstTensor)>{}(dstTensor,
                                                               srcTensor, unitFlag);
}

template <class ArchTag, typename ElementSrc, typename ElementDst,
    Catlass::Gemm::Tile::CopyL0CToUBMode l0c2ubMode>
CATLASS_DEVICE void copyCcToUbufRowMajor(
    memref_t<__cc__ ElementSrc, 1> *src, memref_t<__ubuf__ ElementDst, 1> *dst,
    const TensorDesc4D &srcDesc, const TensorDesc2D &dstDesc, uint8_t unitFlag, uint8_t subBlockId) {
    auto srcTensor = makeL0CTensor(src, srcDesc);
    auto dstTensor = makeUBTensor(dst, dstDesc);
    if constexpr (l0c2ubMode == Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT) {
        Catlass::Gemm::Tile::CopyL0CToUBTla<ArchTag, decltype(srcTensor), decltype(dstTensor), l0c2ubMode>{}(
                                            dstTensor, srcTensor, (bool)subBlockId, unitFlag);
    } else if constexpr (l0c2ubMode == Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M ||
        l0c2ubMode == Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_N) {
        Catlass::Gemm::Tile::CopyL0CToUBTla<ArchTag, decltype(srcTensor), decltype(dstTensor), l0c2ubMode>{}(
                                            dstTensor, srcTensor, unitFlag);
    }
}

template <class ArchTag, typename T>
CATLASS_DEVICE void copyCcToGmzN(memref_t<__cc__ T, 1> *src, memref_t<__gm__ T, 2> *dst,
                                 const TensorDesc4D &srcDesc, const TensorDesc4D &dstDesc) {
    auto srcTensor = makeL0CTensor(src, srcDesc);
    auto dstTensor = makeGMzNTensor(dst, dstDesc);
    Catlass::Gemm::Tile::CopyL0CToGmTla<ArchTag, decltype(srcTensor),
                                        decltype(dstTensor)>{}(dstTensor,
                                                               srcTensor);
}


extern "C" {
#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) ||                    \
     (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))
[aicore] __attribute__((always_inline))
void _mlir_ciface_copy_gm_row_major_to_cbuf_zN_float(
    memref_t<__gm__ float, 2> *src, memref_t<__cbuf__ float, 1> *dst,
    int64_t srcShape0, int64_t srcShape1, int64_t srcStride0,
    int64_t srcStride1, int64_t srcCoord0, int64_t srcCoord1,
    int64_t srcOrgShape0, int64_t srcOrgShape1, int64_t dstShape0,
    int64_t dstShape1, int64_t dstShape2, int64_t dstShape3,
    int64_t dstStride0, int64_t dstStride1, int64_t dstStride2,
    int64_t dstStride3, int64_t dstCoord0, int64_t dstCoord1,
    int64_t dstOrgShape0, int64_t dstOrgShape1) {
    copyGmRowMajorToCbufzN<Catlass::Arch::Ascend950>(
        src, dst,
        TensorDesc2D{(uint32_t)srcShape0, (uint32_t)srcShape1, srcStride0, srcStride1,
                         (uint32_t)srcCoord0, (uint32_t)srcCoord1, (uint32_t)srcOrgShape0, (uint32_t)srcOrgShape1},
        TensorDesc4D{(uint32_t)dstShape0, (uint32_t)dstShape1, (uint32_t)dstShape2, (uint32_t)dstShape3, dstStride0,
                       dstStride1, dstStride2, dstStride3, (uint32_t)dstCoord0, (uint32_t)dstCoord1,
                       (uint32_t)dstOrgShape0, (uint32_t)dstOrgShape1});
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_copy_gm_column_major_to_cbuf_nZ_float(
    memref_t<__gm__ float, 2> *src, memref_t<__cbuf__ float, 1> *dst,
    int64_t srcShape0, int64_t srcShape1, int64_t srcStride0,
    int64_t srcStride1, int64_t srcCoord0, int64_t srcCoord1,
    int64_t srcOrgShape0, int64_t srcOrgShape1, int64_t dstShape0,
    int64_t dstShape1, int64_t dstShape2, int64_t dstShape3,
    int64_t dstStride0, int64_t dstStride1, int64_t dstStride2,
    int64_t dstStride3, int64_t dstCoord0, int64_t dstCoord1,
    int64_t dstOrgShape0, int64_t dstOrgShape1) {
    copyGmColumnMajorToCbufnZ<Catlass::Arch::Ascend950>(
        src, dst,
        TensorDesc2D{(uint32_t)srcShape0, (uint32_t)srcShape1, srcStride0, srcStride1,
                         (uint32_t)srcCoord0, (uint32_t)srcCoord1, (uint32_t)srcOrgShape0, (uint32_t)srcOrgShape1},
        TensorDesc4D{(uint32_t)dstShape0, (uint32_t)dstShape1, (uint32_t)dstShape2, (uint32_t)dstShape3, dstStride0,
                       dstStride1, dstStride2, dstStride3, (uint32_t)dstCoord0, (uint32_t)dstCoord1,
                       (uint32_t)dstOrgShape0, (uint32_t)dstOrgShape1});
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_copy_cbuf_zN_to_ca_zN_float(
    memref_t<__cbuf__ float, 1> *src, memref_t<__ca__ float, 1> *dst,
    int64_t srcShape0, int64_t srcShape1, int64_t srcShape2,
    int64_t srcShape3, int64_t srcStride0, int64_t srcStride1,
    int64_t srcStride2, int64_t srcStride3, int64_t srcCoord0,
    int64_t srcCoord1, int64_t srcOrgShape0, int64_t srcOrgShape1,
    int64_t dstShape0, int64_t dstShape1, int64_t dstShape2,
    int64_t dstShape3, int64_t dstStride0, int64_t dstStride1,
    int64_t dstStride2, int64_t dstStride3, int64_t dstCoord0,
    int64_t dstCoord1, int64_t dstOrgShape0, int64_t dstOrgShape1) {
    copyCbufzNToCazN<Catlass::Arch::Ascend950>(
        src, dst,
        TensorDesc4D{(uint32_t)srcShape0, (uint32_t)srcShape1, (uint32_t)srcShape2, (uint32_t)srcShape3, srcStride0,
                       srcStride1, srcStride2, srcStride3, (uint32_t)srcCoord0, (uint32_t)srcCoord1,
                       (uint32_t)srcOrgShape0, (uint32_t)srcOrgShape1},
        TensorDesc4D{(uint32_t)dstShape0, (uint32_t)dstShape1, (uint32_t)dstShape2, (uint32_t)dstShape3, dstStride0,
                       dstStride1, dstStride2, dstStride3, (uint32_t)dstCoord0, (uint32_t)dstCoord1,
                       (uint32_t)dstOrgShape0, (uint32_t)dstOrgShape1});
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_copy_cbuf_nZ_to_ca_zN_float(
    memref_t<__cbuf__ float, 1> *src, memref_t<__ca__ float, 1> *dst,
    int64_t srcShape0, int64_t srcShape1, int64_t srcShape2,
    int64_t srcShape3, int64_t srcStride0, int64_t srcStride1,
    int64_t srcStride2, int64_t srcStride3, int64_t srcCoord0,
    int64_t srcCoord1, int64_t srcOrgShape0, int64_t srcOrgShape1,
    int64_t dstShape0, int64_t dstShape1, int64_t dstShape2,
    int64_t dstShape3, int64_t dstStride0, int64_t dstStride1,
    int64_t dstStride2, int64_t dstStride3, int64_t dstCoord0,
    int64_t dstCoord1, int64_t dstOrgShape0, int64_t dstOrgShape1) {
    copyCbufnZToCazN<Catlass::Arch::Ascend950>(
        src, dst,
        TensorDesc4D{(uint32_t)srcShape0, (uint32_t)srcShape1, (uint32_t)srcShape2, (uint32_t)srcShape3, srcStride0,
                       srcStride1, srcStride2, srcStride3, (uint32_t)srcCoord0, (uint32_t)srcCoord1,
                       (uint32_t)srcOrgShape0, (uint32_t)srcOrgShape1},
        TensorDesc4D{(uint32_t)dstShape0, (uint32_t)dstShape1, (uint32_t)dstShape2, (uint32_t)dstShape3, dstStride0,
                       dstStride1, dstStride2, dstStride3, (uint32_t)dstCoord0, (uint32_t)dstCoord1,
                       (uint32_t)dstOrgShape0, (uint32_t)dstOrgShape1});
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_copy_cbuf_zN_to_cb_nZ_float(
    memref_t<__cbuf__ float, 1> *src, memref_t<__cb__ float, 1> *dst,
    int64_t srcShape0, int64_t srcShape1, int64_t srcShape2,
    int64_t srcShape3, int64_t srcStride0, int64_t srcStride1,
    int64_t srcStride2, int64_t srcStride3, int64_t srcCoord0,
    int64_t srcCoord1, int64_t srcOrgShape0, int64_t srcOrgShape1,
    int64_t dstShape0, int64_t dstShape1, int64_t dstShape2,
    int64_t dstShape3, int64_t dstStride0, int64_t dstStride1,
    int64_t dstStride2, int64_t dstStride3, int64_t dstCoord0,
    int64_t dstCoord1, int64_t dstOrgShape0, int64_t dstOrgShape1) {
    copyCbufzNToCbnZ<Catlass::Arch::Ascend950>(
        src, dst,
        TensorDesc4D{(uint32_t)srcShape0, (uint32_t)srcShape1, (uint32_t)srcShape2, (uint32_t)srcShape3, srcStride0,
                       srcStride1, srcStride2, srcStride3, (uint32_t)srcCoord0, (uint32_t)srcCoord1,
                       (uint32_t)srcOrgShape0, (uint32_t)srcOrgShape1},
        TensorDesc4D{(uint32_t)dstShape0, (uint32_t)dstShape1, (uint32_t)dstShape2, (uint32_t)dstShape3, dstStride0,
                       dstStride1, dstStride2, dstStride3, (uint32_t)dstCoord0, (uint32_t)dstCoord1,
                       (uint32_t)dstOrgShape0, (uint32_t)dstOrgShape1});
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_copy_cbuf_nZ_to_cb_nZ_float(
    memref_t<__cbuf__ float, 1> *src, memref_t<__cb__ float, 1> *dst,
    int64_t srcShape0, int64_t srcShape1, int64_t srcShape2,
    int64_t srcShape3, int64_t srcStride0, int64_t srcStride1,
    int64_t srcStride2, int64_t srcStride3, int64_t srcCoord0,
    int64_t srcCoord1, int64_t srcOrgShape0, int64_t srcOrgShape1,
    int64_t dstShape0, int64_t dstShape1, int64_t dstShape2,
    int64_t dstShape3, int64_t dstStride0, int64_t dstStride1,
    int64_t dstStride2, int64_t dstStride3, int64_t dstCoord0,
    int64_t dstCoord1, int64_t dstOrgShape0, int64_t dstOrgShape1) {
    copyCbufnZToCbnZ<Catlass::Arch::Ascend950>(
        src, dst,
        TensorDesc4D{(uint32_t)srcShape0, (uint32_t)srcShape1, (uint32_t)srcShape2, (uint32_t)srcShape3, srcStride0,
                       srcStride1, srcStride2, srcStride3, (uint32_t)srcCoord0, (uint32_t)srcCoord1,
                       (uint32_t)srcOrgShape0, (uint32_t)srcOrgShape1},
        TensorDesc4D{(uint32_t)dstShape0, (uint32_t)dstShape1, (uint32_t)dstShape2, (uint32_t)dstShape3, dstStride0,
                       dstStride1, dstStride2, dstStride3, (uint32_t)dstCoord0, (uint32_t)dstCoord1,
                       (uint32_t)dstOrgShape0, (uint32_t)dstOrgShape1});
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_copy_cc_to_gm_row_major_float(
    memref_t<__cc__ float, 1> *src, memref_t<__gm__ float, 2> *dst,
    int64_t srcShape0, int64_t srcShape1, int64_t srcShape2,
    int64_t srcShape3, int64_t srcStride0, int64_t srcStride1,
    int64_t srcStride2, int64_t srcStride3, int64_t srcCoord0,
    int64_t srcCoord1, int64_t srcOrgShape0, int64_t srcOrgShape1,
    int64_t dstShape0, int64_t dstShape1, int64_t dstStride0,
    int64_t dstStride1, int64_t dstCoord0, int64_t dstCoord1,
    int64_t dstOrgShape0, int64_t dstOrgShape1, uint8_t unitFlag) {
    copyCcToGmRowMajor<Catlass::Arch::Ascend950>(
        src, dst,
        TensorDesc4D{(uint32_t)srcShape0, (uint32_t)srcShape1, (uint32_t)srcShape2, (uint32_t)srcShape3, srcStride0,
                       srcStride1, srcStride2, srcStride3, (uint32_t)srcCoord0, (uint32_t)srcCoord1,
                       (uint32_t)srcOrgShape0, (uint32_t)srcOrgShape1},
        TensorDesc2D{(uint32_t)dstShape0, (uint32_t)dstShape1, dstStride0, dstStride1,
                         (uint32_t)dstCoord0, (uint32_t)dstCoord1, (uint32_t)dstOrgShape0, (uint32_t)dstOrgShape1},
        unitFlag);
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_copy_cc_to_gm_row_major_half(
    memref_t<__cc__ float, 1> *src, memref_t<__gm__ half, 2> *dst,
    int64_t srcShape0, int64_t srcShape1, int64_t srcShape2,
    int64_t srcShape3, int64_t srcStride0, int64_t srcStride1,
    int64_t srcStride2, int64_t srcStride3, int64_t srcCoord0,
    int64_t srcCoord1, int64_t srcOrgShape0, int64_t srcOrgShape1,
    int64_t dstShape0, int64_t dstShape1, int64_t dstStride0,
    int64_t dstStride1, int64_t dstCoord0, int64_t dstCoord1,
    int64_t dstOrgShape0, int64_t dstOrgShape1, uint8_t unitFlag) {
    copyCcFloatToGmRowMajorCast<Catlass::Arch::Ascend950, half>(
        src, dst,
        TensorDesc4D{(uint32_t)srcShape0, (uint32_t)srcShape1, (uint32_t)srcShape2, (uint32_t)srcShape3, srcStride0,
                       srcStride1, srcStride2, srcStride3, (uint32_t)srcCoord0, (uint32_t)srcCoord1,
                       (uint32_t)srcOrgShape0, (uint32_t)srcOrgShape1},
        TensorDesc2D{(uint32_t)dstShape0, (uint32_t)dstShape1, dstStride0, dstStride1,
                         (uint32_t)dstCoord0, (uint32_t)dstCoord1, (uint32_t)dstOrgShape0, (uint32_t)dstOrgShape1},
        unitFlag);
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_copy_cc_to_gm_row_major_bf16(
    memref_t<__cc__ float, 1> *src, memref_t<__gm__ bfloat16_t, 2> *dst,
    int64_t srcShape0, int64_t srcShape1, int64_t srcShape2,
    int64_t srcShape3, int64_t srcStride0, int64_t srcStride1,
    int64_t srcStride2, int64_t srcStride3, int64_t srcCoord0,
    int64_t srcCoord1, int64_t srcOrgShape0, int64_t srcOrgShape1,
    int64_t dstShape0, int64_t dstShape1, int64_t dstStride0,
    int64_t dstStride1, int64_t dstCoord0, int64_t dstCoord1,
    int64_t dstOrgShape0, int64_t dstOrgShape1, uint8_t unitFlag) {
    copyCcFloatToGmRowMajorCast<Catlass::Arch::Ascend950, bfloat16_t>(
        src, dst,
        TensorDesc4D{(uint32_t)srcShape0, (uint32_t)srcShape1, (uint32_t)srcShape2, (uint32_t)srcShape3, srcStride0,
                       srcStride1, srcStride2, srcStride3, (uint32_t)srcCoord0, (uint32_t)srcCoord1,
                       (uint32_t)srcOrgShape0, (uint32_t)srcOrgShape1},
        TensorDesc2D{(uint32_t)dstShape0, (uint32_t)dstShape1, dstStride0, dstStride1,
                         (uint32_t)dstCoord0, (uint32_t)dstCoord1, (uint32_t)dstOrgShape0, (uint32_t)dstOrgShape1},
        unitFlag);
}

#define REGISTER_CC_TO_UBUF_ROW(mode, MODE, DTypeSrc, DTypeDst)                                                      \
[aicore] __attribute__((always_inline))                                                                              \
void _mlir_ciface_copy_cc_to_ubuf_row_major_##mode##_##DTypeDst(                                                     \
    memref_t<__cc__ DTypeSrc, 1> *src, memref_t<__ubuf__ DTypeDst, 1> *dst,                                          \
    int64_t srcShape0, int64_t srcShape1, int64_t srcShape2,                                                         \
    int64_t srcShape3, int64_t srcStride0, int64_t srcStride1,                                                       \
    int64_t srcStride2, int64_t srcStride3, int64_t srcCoord0,                                                       \
    int64_t srcCoord1, int64_t srcOrgShape0, int64_t srcOrgShape1,                                                   \
    int64_t dstShape0, int64_t dstShape1, int64_t dstStride0,                                                        \
    int64_t dstStride1, int64_t dstCoord0, int64_t dstCoord1,                                                        \
    int64_t dstOrgShape0, int64_t dstOrgShape1, uint8_t unitFlag,                                                    \
    uint8_t subBlockId) {                                                                                            \
    copyCcToUbufRowMajor<Catlass::Arch::Ascend950, DTypeSrc, DTypeDst, Catlass::Gemm::Tile::CopyL0CToUBMode::MODE>(  \
        src, dst,                                                                                                    \
        TensorDesc4D{(uint32_t)srcShape0, (uint32_t)srcShape1, (uint32_t)srcShape2, (uint32_t)srcShape3, srcStride0, \
                       srcStride1, srcStride2, srcStride3, (uint32_t)srcCoord0, (uint32_t)srcCoord1,                 \
                       (uint32_t)srcOrgShape0, (uint32_t)srcOrgShape1},                                              \
        TensorDesc2D{(uint32_t)dstShape0, (uint32_t)dstShape1, dstStride0, dstStride1,                               \
                         (uint32_t)dstCoord0, (uint32_t)dstCoord1, (uint32_t)dstOrgShape0, (uint32_t)dstOrgShape1},  \
        unitFlag, subBlockId);                                                                                       \
}

REGISTER_CC_TO_UBUF_ROW(nosplit, NO_SPLIT, float, float)
REGISTER_CC_TO_UBUF_ROW(nosplit, NO_SPLIT, float, half)
REGISTER_CC_TO_UBUF_ROW(nosplit, NO_SPLIT, float, bf16)
// split mode src=dst
REGISTER_CC_TO_UBUF_ROW(splitm, SPLIT_M, float, float)
REGISTER_CC_TO_UBUF_ROW(splitn, SPLIT_N, float, float)

[aicore] __attribute__((always_inline))
void _mlir_ciface_copy_cc_to_gm_zN_float(
    memref_t<__cc__ float, 1> *src, memref_t<__gm__ float, 2> *dst,
    int64_t srcShape0, int64_t srcShape1, int64_t srcShape2,
    int64_t srcShape3, int64_t srcStride0, int64_t srcStride1,
    int64_t srcStride2, int64_t srcStride3, int64_t srcCoord0,
    int64_t srcCoord1, int64_t srcOrgShape0, int64_t srcOrgShape1,
    int64_t dstShape0, int64_t dstShape1, int64_t dstShape2, int64_t dstShape3,
    int64_t dstStride0, int64_t dstStride1, int64_t dstStride2, int64_t dstStride3,
    int64_t dstCoord0, int64_t dstCoord1, int64_t dstOrgShape0, int64_t dstOrgShape1) {
    copyCcToGmzN<Catlass::Arch::Ascend950>(
        src, dst,
        TensorDesc4D{(uint32_t)srcShape0, (uint32_t)srcShape1, (uint32_t)srcShape2, (uint32_t)srcShape3, srcStride0,
                       srcStride1, srcStride2, srcStride3, (uint32_t)srcCoord0, (uint32_t)srcCoord1,
                       (uint32_t)srcOrgShape0, (uint32_t)srcOrgShape1},
        TensorDesc4D{(uint32_t)dstShape0, (uint32_t)dstShape1, (uint32_t)dstShape2, (uint32_t)dstShape3, dstStride0,
                       dstStride1, dstStride2, dstStride3, (uint32_t)dstCoord0, (uint32_t)dstCoord1,
                       (uint32_t)dstOrgShape0, (uint32_t)dstOrgShape1});
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_copy_gm_row_major_to_cbuf_zN_half(
    memref_t<__gm__ half, 2> *src, memref_t<__cbuf__ half, 1> *dst,
    int64_t srcShape0, int64_t srcShape1, int64_t srcStride0,
    int64_t srcStride1, int64_t srcCoord0, int64_t srcCoord1,
    int64_t srcOrgShape0, int64_t srcOrgShape1, int64_t dstShape0,
    int64_t dstShape1, int64_t dstShape2, int64_t dstShape3,
    int64_t dstStride0, int64_t dstStride1, int64_t dstStride2,
    int64_t dstStride3, int64_t dstCoord0, int64_t dstCoord1,
    int64_t dstOrgShape0, int64_t dstOrgShape1) {
    copyGmRowMajorToCbufzN<Catlass::Arch::Ascend950>(
        src, dst,
        TensorDesc2D{(uint32_t)srcShape0, (uint32_t)srcShape1, srcStride0, srcStride1,
                         (uint32_t)srcCoord0, (uint32_t)srcCoord1, (uint32_t)srcOrgShape0, (uint32_t)srcOrgShape1},
        TensorDesc4D{(uint32_t)dstShape0, (uint32_t)dstShape1, (uint32_t)dstShape2, (uint32_t)dstShape3, dstStride0,
                       dstStride1, dstStride2, dstStride3, (uint32_t)dstCoord0, (uint32_t)dstCoord1,
                       (uint32_t)dstOrgShape0, (uint32_t)dstOrgShape1});
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_copy_gm_column_major_to_cbuf_nZ_half(
    memref_t<__gm__ half, 2> *src, memref_t<__cbuf__ half, 1> *dst,
    int64_t srcShape0, int64_t srcShape1, int64_t srcStride0,
    int64_t srcStride1, int64_t srcCoord0, int64_t srcCoord1,
    int64_t srcOrgShape0, int64_t srcOrgShape1, int64_t dstShape0,
    int64_t dstShape1, int64_t dstShape2, int64_t dstShape3,
    int64_t dstStride0, int64_t dstStride1, int64_t dstStride2,
    int64_t dstStride3, int64_t dstCoord0, int64_t dstCoord1,
    int64_t dstOrgShape0, int64_t dstOrgShape1) {
    copyGmColumnMajorToCbufnZ<Catlass::Arch::Ascend950>(
        src, dst,
        TensorDesc2D{(uint32_t)srcShape0, (uint32_t)srcShape1, srcStride0, srcStride1,
                         (uint32_t)srcCoord0, (uint32_t)srcCoord1, (uint32_t)srcOrgShape0, (uint32_t)srcOrgShape1},
        TensorDesc4D{(uint32_t)dstShape0, (uint32_t)dstShape1, (uint32_t)dstShape2, (uint32_t)dstShape3, dstStride0,
                       dstStride1, dstStride2, dstStride3, (uint32_t)dstCoord0, (uint32_t)dstCoord1,
                       (uint32_t)dstOrgShape0, (uint32_t)dstOrgShape1});
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_copy_cbuf_zN_to_ca_zN_half(
    memref_t<__cbuf__ half, 1> *src, memref_t<__ca__ half, 1> *dst,
    int64_t srcShape0, int64_t srcShape1, int64_t srcShape2,
    int64_t srcShape3, int64_t srcStride0, int64_t srcStride1,
    int64_t srcStride2, int64_t srcStride3, int64_t srcCoord0,
    int64_t srcCoord1, int64_t srcOrgShape0, int64_t srcOrgShape1,
    int64_t dstShape0, int64_t dstShape1, int64_t dstShape2,
    int64_t dstShape3, int64_t dstStride0, int64_t dstStride1,
    int64_t dstStride2, int64_t dstStride3, int64_t dstCoord0,
    int64_t dstCoord1, int64_t dstOrgShape0, int64_t dstOrgShape1) {
    copyCbufzNToCazN<Catlass::Arch::Ascend950>(
        src, dst,
        TensorDesc4D{(uint32_t)srcShape0, (uint32_t)srcShape1, (uint32_t)srcShape2, (uint32_t)srcShape3, srcStride0,
                       srcStride1, srcStride2, srcStride3, (uint32_t)srcCoord0, (uint32_t)srcCoord1,
                       (uint32_t)srcOrgShape0, (uint32_t)srcOrgShape1},
        TensorDesc4D{(uint32_t)dstShape0, (uint32_t)dstShape1, (uint32_t)dstShape2, (uint32_t)dstShape3, dstStride0,
                       dstStride1, dstStride2, dstStride3, (uint32_t)dstCoord0, (uint32_t)dstCoord1,
                       (uint32_t)dstOrgShape0, (uint32_t)dstOrgShape1});
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_copy_cbuf_nZ_to_ca_zN_half(
    memref_t<__cbuf__ half, 1> *src, memref_t<__ca__ half, 1> *dst,
    int64_t srcShape0, int64_t srcShape1, int64_t srcShape2,
    int64_t srcShape3, int64_t srcStride0, int64_t srcStride1,
    int64_t srcStride2, int64_t srcStride3, int64_t srcCoord0,
    int64_t srcCoord1, int64_t srcOrgShape0, int64_t srcOrgShape1,
    int64_t dstShape0, int64_t dstShape1, int64_t dstShape2,
    int64_t dstShape3, int64_t dstStride0, int64_t dstStride1,
    int64_t dstStride2, int64_t dstStride3, int64_t dstCoord0,
    int64_t dstCoord1, int64_t dstOrgShape0, int64_t dstOrgShape1) {
    copyCbufnZToCazN<Catlass::Arch::Ascend950>(
        src, dst,
        TensorDesc4D{(uint32_t)srcShape0, (uint32_t)srcShape1, (uint32_t)srcShape2, (uint32_t)srcShape3, srcStride0,
                       srcStride1, srcStride2, srcStride3, (uint32_t)srcCoord0, (uint32_t)srcCoord1,
                       (uint32_t)srcOrgShape0, (uint32_t)srcOrgShape1},
        TensorDesc4D{(uint32_t)dstShape0, (uint32_t)dstShape1, (uint32_t)dstShape2, (uint32_t)dstShape3, dstStride0,
                       dstStride1, dstStride2, dstStride3, (uint32_t)dstCoord0, (uint32_t)dstCoord1,
                       (uint32_t)dstOrgShape0, (uint32_t)dstOrgShape1});
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_copy_cbuf_zN_to_cb_nZ_half(
    memref_t<__cbuf__ half, 1> *src, memref_t<__cb__ half, 1> *dst,
    int64_t srcShape0, int64_t srcShape1, int64_t srcShape2,
    int64_t srcShape3, int64_t srcStride0, int64_t srcStride1,
    int64_t srcStride2, int64_t srcStride3, int64_t srcCoord0,
    int64_t srcCoord1, int64_t srcOrgShape0, int64_t srcOrgShape1,
    int64_t dstShape0, int64_t dstShape1, int64_t dstShape2,
    int64_t dstShape3, int64_t dstStride0, int64_t dstStride1,
    int64_t dstStride2, int64_t dstStride3, int64_t dstCoord0,
    int64_t dstCoord1, int64_t dstOrgShape0, int64_t dstOrgShape1) {
    copyCbufzNToCbnZ<Catlass::Arch::Ascend950>(
        src, dst,
        TensorDesc4D{(uint32_t)srcShape0, (uint32_t)srcShape1, (uint32_t)srcShape2, (uint32_t)srcShape3, srcStride0,
                       srcStride1, srcStride2, srcStride3, (uint32_t)srcCoord0, (uint32_t)srcCoord1,
                       (uint32_t)srcOrgShape0, (uint32_t)srcOrgShape1},
        TensorDesc4D{(uint32_t)dstShape0, (uint32_t)dstShape1, (uint32_t)dstShape2, (uint32_t)dstShape3, dstStride0,
                       dstStride1, dstStride2, dstStride3, (uint32_t)dstCoord0, (uint32_t)dstCoord1,
                       (uint32_t)dstOrgShape0, (uint32_t)dstOrgShape1});
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_copy_cbuf_nZ_to_cb_nZ_half(
    memref_t<__cbuf__ half, 1> *src, memref_t<__cb__ half, 1> *dst,
    int64_t srcShape0, int64_t srcShape1, int64_t srcShape2,
    int64_t srcShape3, int64_t srcStride0, int64_t srcStride1,
    int64_t srcStride2, int64_t srcStride3, int64_t srcCoord0,
    int64_t srcCoord1, int64_t srcOrgShape0, int64_t srcOrgShape1,
    int64_t dstShape0, int64_t dstShape1, int64_t dstShape2,
    int64_t dstShape3, int64_t dstStride0, int64_t dstStride1,
    int64_t dstStride2, int64_t dstStride3, int64_t dstCoord0,
    int64_t dstCoord1, int64_t dstOrgShape0, int64_t dstOrgShape1) {
    copyCbufnZToCbnZ<Catlass::Arch::Ascend950>(
        src, dst,
        TensorDesc4D{(uint32_t)srcShape0, (uint32_t)srcShape1, (uint32_t)srcShape2, (uint32_t)srcShape3, srcStride0,
                       srcStride1, srcStride2, srcStride3, (uint32_t)srcCoord0, (uint32_t)srcCoord1,
                       (uint32_t)srcOrgShape0, (uint32_t)srcOrgShape1},
        TensorDesc4D{(uint32_t)dstShape0, (uint32_t)dstShape1, (uint32_t)dstShape2, (uint32_t)dstShape3, dstStride0,
                       dstStride1, dstStride2, dstStride3, (uint32_t)dstCoord0, (uint32_t)dstCoord1,
                       (uint32_t)dstOrgShape0, (uint32_t)dstOrgShape1});
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_copy_gm_row_major_to_cbuf_zN_bf16(
    memref_t<__gm__ bfloat16_t, 2> *src, memref_t<__cbuf__ bfloat16_t, 1> *dst,
    int64_t srcShape0, int64_t srcShape1, int64_t srcStride0,
    int64_t srcStride1, int64_t srcCoord0, int64_t srcCoord1,
    int64_t srcOrgShape0, int64_t srcOrgShape1, int64_t dstShape0,
    int64_t dstShape1, int64_t dstShape2, int64_t dstShape3,
    int64_t dstStride0, int64_t dstStride1, int64_t dstStride2,
    int64_t dstStride3, int64_t dstCoord0, int64_t dstCoord1,
    int64_t dstOrgShape0, int64_t dstOrgShape1) {
    copyGmRowMajorToCbufzN<Catlass::Arch::Ascend950>(
        src, dst,
        TensorDesc2D{(uint32_t)srcShape0, (uint32_t)srcShape1, srcStride0, srcStride1,
                         (uint32_t)srcCoord0, (uint32_t)srcCoord1, (uint32_t)srcOrgShape0, (uint32_t)srcOrgShape1},
        TensorDesc4D{(uint32_t)dstShape0, (uint32_t)dstShape1, (uint32_t)dstShape2, (uint32_t)dstShape3, dstStride0,
                       dstStride1, dstStride2, dstStride3, (uint32_t)dstCoord0, (uint32_t)dstCoord1,
                       (uint32_t)dstOrgShape0, (uint32_t)dstOrgShape1});
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_copy_gm_column_major_to_cbuf_nZ_bf16(
    memref_t<__gm__ bfloat16_t, 2> *src, memref_t<__cbuf__ bfloat16_t, 1> *dst,
    int64_t srcShape0, int64_t srcShape1, int64_t srcStride0,
    int64_t srcStride1, int64_t srcCoord0, int64_t srcCoord1,
    int64_t srcOrgShape0, int64_t srcOrgShape1, int64_t dstShape0,
    int64_t dstShape1, int64_t dstShape2, int64_t dstShape3,
    int64_t dstStride0, int64_t dstStride1, int64_t dstStride2,
    int64_t dstStride3, int64_t dstCoord0, int64_t dstCoord1,
    int64_t dstOrgShape0, int64_t dstOrgShape1) {
    copyGmColumnMajorToCbufnZ<Catlass::Arch::Ascend950>(
        src, dst,
        TensorDesc2D{(uint32_t)srcShape0, (uint32_t)srcShape1, srcStride0, srcStride1,
                         (uint32_t)srcCoord0, (uint32_t)srcCoord1, (uint32_t)srcOrgShape0, (uint32_t)srcOrgShape1},
        TensorDesc4D{(uint32_t)dstShape0, (uint32_t)dstShape1, (uint32_t)dstShape2, (uint32_t)dstShape3, dstStride0,
                       dstStride1, dstStride2, dstStride3, (uint32_t)dstCoord0, (uint32_t)dstCoord1,
                       (uint32_t)dstOrgShape0, (uint32_t)dstOrgShape1});
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_copy_cbuf_zN_to_ca_zN_bf16(
    memref_t<__cbuf__ bfloat16_t, 1> *src, memref_t<__ca__ bfloat16_t, 1> *dst,
    int64_t srcShape0, int64_t srcShape1, int64_t srcShape2,
    int64_t srcShape3, int64_t srcStride0, int64_t srcStride1,
    int64_t srcStride2, int64_t srcStride3, int64_t srcCoord0,
    int64_t srcCoord1, int64_t srcOrgShape0, int64_t srcOrgShape1,
    int64_t dstShape0, int64_t dstShape1, int64_t dstShape2,
    int64_t dstShape3, int64_t dstStride0, int64_t dstStride1,
    int64_t dstStride2, int64_t dstStride3, int64_t dstCoord0,
    int64_t dstCoord1, int64_t dstOrgShape0, int64_t dstOrgShape1) {
    copyCbufzNToCazN<Catlass::Arch::Ascend950>(
        src, dst,
        TensorDesc4D{(uint32_t)srcShape0, (uint32_t)srcShape1, (uint32_t)srcShape2, (uint32_t)srcShape3, srcStride0,
                       srcStride1, srcStride2, srcStride3, (uint32_t)srcCoord0, (uint32_t)srcCoord1,
                       (uint32_t)srcOrgShape0, (uint32_t)srcOrgShape1},
        TensorDesc4D{(uint32_t)dstShape0, (uint32_t)dstShape1, (uint32_t)dstShape2, (uint32_t)dstShape3, dstStride0,
                       dstStride1, dstStride2, dstStride3, (uint32_t)dstCoord0, (uint32_t)dstCoord1,
                       (uint32_t)dstOrgShape0, (uint32_t)dstOrgShape1});
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_copy_cbuf_nZ_to_ca_zN_bf16(
    memref_t<__cbuf__ bfloat16_t, 1> *src, memref_t<__ca__ bfloat16_t, 1> *dst,
    int64_t srcShape0, int64_t srcShape1, int64_t srcShape2,
    int64_t srcShape3, int64_t srcStride0, int64_t srcStride1,
    int64_t srcStride2, int64_t srcStride3, int64_t srcCoord0,
    int64_t srcCoord1, int64_t srcOrgShape0, int64_t srcOrgShape1,
    int64_t dstShape0, int64_t dstShape1, int64_t dstShape2,
    int64_t dstShape3, int64_t dstStride0, int64_t dstStride1,
    int64_t dstStride2, int64_t dstStride3, int64_t dstCoord0,
    int64_t dstCoord1, int64_t dstOrgShape0, int64_t dstOrgShape1) {
    copyCbufnZToCazN<Catlass::Arch::Ascend950>(
        src, dst,
        TensorDesc4D{(uint32_t)srcShape0, (uint32_t)srcShape1, (uint32_t)srcShape2, (uint32_t)srcShape3, srcStride0,
                       srcStride1, srcStride2, srcStride3, (uint32_t)srcCoord0, (uint32_t)srcCoord1,
                       (uint32_t)srcOrgShape0, (uint32_t)srcOrgShape1},
        TensorDesc4D{(uint32_t)dstShape0, (uint32_t)dstShape1, (uint32_t)dstShape2, (uint32_t)dstShape3, dstStride0,
                       dstStride1, dstStride2, dstStride3, (uint32_t)dstCoord0, (uint32_t)dstCoord1,
                       (uint32_t)dstOrgShape0, (uint32_t)dstOrgShape1});
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_copy_cbuf_zN_to_cb_nZ_bf16(
    memref_t<__cbuf__ bfloat16_t, 1> *src, memref_t<__cb__ bfloat16_t, 1> *dst,
    int64_t srcShape0, int64_t srcShape1, int64_t srcShape2,
    int64_t srcShape3, int64_t srcStride0, int64_t srcStride1,
    int64_t srcStride2, int64_t srcStride3, int64_t srcCoord0,
    int64_t srcCoord1, int64_t srcOrgShape0, int64_t srcOrgShape1,
    int64_t dstShape0, int64_t dstShape1, int64_t dstShape2,
    int64_t dstShape3, int64_t dstStride0, int64_t dstStride1,
    int64_t dstStride2, int64_t dstStride3, int64_t dstCoord0,
    int64_t dstCoord1, int64_t dstOrgShape0, int64_t dstOrgShape1) {
    copyCbufzNToCbnZ<Catlass::Arch::Ascend950>(
        src, dst,
        TensorDesc4D{(uint32_t)srcShape0, (uint32_t)srcShape1, (uint32_t)srcShape2, (uint32_t)srcShape3, srcStride0,
                       srcStride1, srcStride2, srcStride3, (uint32_t)srcCoord0, (uint32_t)srcCoord1,
                       (uint32_t)srcOrgShape0, (uint32_t)srcOrgShape1},
        TensorDesc4D{(uint32_t)dstShape0, (uint32_t)dstShape1, (uint32_t)dstShape2, (uint32_t)dstShape3, dstStride0,
                       dstStride1, dstStride2, dstStride3, (uint32_t)dstCoord0, (uint32_t)dstCoord1,
                       (uint32_t)dstOrgShape0, (uint32_t)dstOrgShape1});
}

[aicore] __attribute__((always_inline))
void _mlir_ciface_copy_cbuf_nZ_to_cb_nZ_bf16(
    memref_t<__cbuf__ bfloat16_t, 1> *src, memref_t<__cb__ bfloat16_t, 1> *dst,
    int64_t srcShape0, int64_t srcShape1, int64_t srcShape2,
    int64_t srcShape3, int64_t srcStride0, int64_t srcStride1,
    int64_t srcStride2, int64_t srcStride3, int64_t srcCoord0,
    int64_t srcCoord1, int64_t srcOrgShape0, int64_t srcOrgShape1,
    int64_t dstShape0, int64_t dstShape1, int64_t dstShape2,
    int64_t dstShape3, int64_t dstStride0, int64_t dstStride1,
    int64_t dstStride2, int64_t dstStride3, int64_t dstCoord0,
    int64_t dstCoord1, int64_t dstOrgShape0, int64_t dstOrgShape1) {
    copyCbufnZToCbnZ<Catlass::Arch::Ascend950>(
        src, dst,
        TensorDesc4D{(uint32_t)srcShape0, (uint32_t)srcShape1, (uint32_t)srcShape2, (uint32_t)srcShape3, srcStride0,
                       srcStride1, srcStride2, srcStride3, (uint32_t)srcCoord0, (uint32_t)srcCoord1,
                       (uint32_t)srcOrgShape0, (uint32_t)srcOrgShape1},
        TensorDesc4D{(uint32_t)dstShape0, (uint32_t)dstShape1, (uint32_t)dstShape2, (uint32_t)dstShape3, dstStride0,
                       dstStride1, dstStride2, dstStride3, (uint32_t)dstCoord0, (uint32_t)dstCoord1,
                       (uint32_t)dstOrgShape0, (uint32_t)dstOrgShape1});
}

#endif
}
