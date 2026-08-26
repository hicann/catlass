#include "../dma_common.h"

#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"
#include "catlass/gemm/tile/copy_l0c_to_ub.hpp"
#include "catlass/gemm/tile/copy_l0c_to_l1.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

template <class ArchTag, LayoutTag SrcLayout, LayoutTag DstLayout, typename T>
CATLASS_DEVICE void copyGMToL1(
    memref_t<__gm__ T, 2>* src, memref_t<__cbuf__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc)
{
    auto srcTensor = makeGMTensor<SrcLayout, T>(src, srcDesc);
    auto dstTensor = makeL1Tensor<DstLayout, T>(dst, dstDesc);
    Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

template <class ArchTag, LayoutTag SrcLayout, typename T>
CATLASS_DEVICE void copyL1ToL0A(
    memref_t<__cbuf__ T, 1>* src, memref_t<__ca__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc)
{
    auto srcTensor = makeL1Tensor<SrcLayout, T>(src, srcDesc);
    auto dstTensor = makeL0ATensor<LayoutTag::zN, T>(dst, dstDesc);
    Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

template <class ArchTag, LayoutTag SrcLayout, typename T>
CATLASS_DEVICE void copyL1ToL0B(
    memref_t<__cbuf__ T, 1>* src, memref_t<__cb__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc)
{
    auto srcTensor = makeL1Tensor<SrcLayout, T>(src, srcDesc);
    auto dstTensor = makeL0BTensor<LayoutTag::nZ, T>(dst, dstDesc);
    Catlass::Gemm::Tile::TileCopyTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

// L0C holds fp32 MMAD accumulator; ElementDst may be f32/f16/bf16 (Ascend950 fixpipe).
template <class ArchTag, LayoutTag DstLayout, typename ElementSrc, typename ElementDst>
CATLASS_DEVICE void copyL0CToGM(
    memref_t<__cc__ ElementSrc, 1>* src, memref_t<__gm__ ElementDst, 2>* dst, const TensorDesc& srcDesc,
    const TensorDesc& dstDesc, uint8_t unitFlag)
{
    auto srcTensor = makeL0CTensor<LayoutTag::L0C, ElementSrc>(src, srcDesc);
    auto dstTensor = makeGMTensor<DstLayout, ElementDst>(dst, dstDesc);
    Catlass::Gemm::Tile::CopyL0CToGmTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(
        dstTensor, srcTensor, unitFlag);
}

// L0C holds fp32 MMAD accumulator; ElementDst may be f32/f16/bf16 (Ascend950 fixpipe).
template <class ArchTag, LayoutTag DstLayout, typename ElementSrc, typename ElementDst>
CATLASS_DEVICE void copyL0CToL1(
    memref_t<__cc__ ElementSrc, 1>* src, memref_t<__cbuf__ ElementDst, 1>* dst, const TensorDesc& srcDesc,
    const TensorDesc& dstDesc, uint8_t unitFlag)
{
    auto srcTensor = makeL0CTensor<LayoutTag::L0C, ElementSrc>(src, srcDesc);
    auto dstTensor = makeL1Tensor<DstLayout, ElementDst>(dst, dstDesc);
    Catlass::Gemm::Tile::CopyL0CToL1Tla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(
        dstTensor, srcTensor, unitFlag);
}

template <
    class ArchTag, LayoutTag DstLayout, Catlass::Gemm::Tile::CopyL0CToUBMode Mode, typename ElementSrc,
    typename ElementDst>
CATLASS_DEVICE void copyL0CToUB(
    memref_t<__cc__ ElementSrc, 1>* src, memref_t<__ubuf__ ElementDst, 1>* dst, const TensorDesc& srcDesc,
    const TensorDesc& dstDesc, uint8_t unitFlag, uint8_t subBlockId)
{
    auto srcTensor = makeL0CTensor<LayoutTag::L0C, ElementSrc>(src, srcDesc);
    auto dstTensor = makeUBTensor<DstLayout, ElementDst>(dst, dstDesc);
    if constexpr (Mode == Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT) {
        Catlass::Gemm::Tile::CopyL0CToUBTla<ArchTag, decltype(srcTensor), decltype(dstTensor), Mode>{}(
            dstTensor, srcTensor, (bool)subBlockId, unitFlag);
    } else if constexpr (
        Mode == Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M ||
        Mode == Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_N) {
        Catlass::Gemm::Tile::CopyL0CToUBTla<ArchTag, decltype(srcTensor), decltype(dstTensor), Mode>{}(
            dstTensor, srcTensor, unitFlag);
    }
}

extern "C" {
#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))

#define REGISTER_GM_TO_L1(LayoutSrc, LayoutDst, DType)                                                                \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_gm_##LayoutSrc##_to_l1_##LayoutDst##_##DType(      \
        memref_t<__gm__ DType, 2>* src, memref_t<__cbuf__ DType, 1>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst)) \
    {                                                                                                                 \
        copyGMToL1<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, DType>(                      \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                      \
    }

REGISTER_GM_TO_L1(RowMajor, zN, float)
REGISTER_GM_TO_L1(RowMajor, zN, half)
REGISTER_GM_TO_L1(RowMajor, zN, bf16)
REGISTER_GM_TO_L1(RowMajor, zN, int8_t)
REGISTER_GM_TO_L1(ColumnMajor, nZ, float)
REGISTER_GM_TO_L1(ColumnMajor, nZ, half)
REGISTER_GM_TO_L1(ColumnMajor, nZ, bf16)
REGISTER_GM_TO_L1(ColumnMajor, nZ, int8_t)
REGISTER_GM_TO_L1(RowMajor, zN, fp8_e4m3fn_t)
REGISTER_GM_TO_L1(ColumnMajor, nZ, fp8_e4m3fn_t)
REGISTER_GM_TO_L1(RowMajor, zN, fp8_e5m2_t)
REGISTER_GM_TO_L1(ColumnMajor, nZ, fp8_e5m2_t)

#define REGISTER_L1_TO_L0A(LayoutSrc, DType)                                                                          \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_l1_##LayoutSrc##_to_l0a_zN_##DType(                \
        memref_t<__cbuf__ DType, 1>* src, memref_t<__ca__ DType, 1>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst)) \
    {                                                                                                                 \
        copyL1ToL0A<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, DType>(                                           \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                      \
    }

REGISTER_L1_TO_L0A(zN, float)
REGISTER_L1_TO_L0A(zN, half)
REGISTER_L1_TO_L0A(zN, bf16)
REGISTER_L1_TO_L0A(zN, int8_t)
REGISTER_L1_TO_L0A(nZ, float)
REGISTER_L1_TO_L0A(nZ, half)
REGISTER_L1_TO_L0A(nZ, bf16)
REGISTER_L1_TO_L0A(nZ, int8_t)
REGISTER_L1_TO_L0A(zN, fp8_e4m3fn_t)
REGISTER_L1_TO_L0A(nZ, fp8_e4m3fn_t)
REGISTER_L1_TO_L0A(zN, fp8_e5m2_t)
REGISTER_L1_TO_L0A(nZ, fp8_e5m2_t)

#define REGISTER_L1_TO_L0B(LayoutSrc, DType)                                                                          \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_l1_##LayoutSrc##_to_l0b_nZ_##DType(                \
        memref_t<__cbuf__ DType, 1>* src, memref_t<__cb__ DType, 1>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst)) \
    {                                                                                                                 \
        copyL1ToL0B<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, DType>(                                           \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                      \
    }

REGISTER_L1_TO_L0B(zN, float)
REGISTER_L1_TO_L0B(zN, half)
REGISTER_L1_TO_L0B(zN, bf16)
REGISTER_L1_TO_L0B(zN, int8_t)
REGISTER_L1_TO_L0B(nZ, float)
REGISTER_L1_TO_L0B(nZ, half)
REGISTER_L1_TO_L0B(nZ, bf16)
REGISTER_L1_TO_L0B(nZ, int8_t)
REGISTER_L1_TO_L0B(zN, fp8_e4m3fn_t)
REGISTER_L1_TO_L0B(nZ, fp8_e4m3fn_t)
REGISTER_L1_TO_L0B(zN, fp8_e5m2_t)
REGISTER_L1_TO_L0B(nZ, fp8_e5m2_t)

#define REGISTER_L0C_TO_GM(DTypeSrc, DTypeDst)                                                      \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_l0c_to_gm_RowMajor_##DTypeDst(   \
        memref_t<__cc__ DTypeSrc, 1>* src, memref_t<__gm__ DTypeDst, 2>* dst, DESC_ABI_PARAMS(src), \
        DESC_ABI_PARAMS(dst), uint8_t unitFlag)                                                     \
    {                                                                                               \
        copyL0CToGM<Catlass::Arch::Ascend950, LayoutTag::RowMajor, DTypeSrc, DTypeDst>(             \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), unitFlag);                          \
    }

REGISTER_L0C_TO_GM(float, float)
REGISTER_L0C_TO_GM(float, half)
REGISTER_L0C_TO_GM(float, bf16)
// Integer route: an int8 MMAD leaves an int32 accumulator in L0C.
REGISTER_L0C_TO_GM(int32_t, int32_t)

#define REGISTER_L0C_TO_L1(DTypeSrc, DTypeDst)                                                        \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_l0c_to_l1_zN_##DTypeDst(           \
        memref_t<__cc__ DTypeSrc, 1>* src, memref_t<__cbuf__ DTypeDst, 1>* dst, DESC_ABI_PARAMS(src), \
        DESC_ABI_PARAMS(dst), uint8_t unitFlag)                                                       \
    {                                                                                                 \
        copyL0CToL1<Catlass::Arch::Ascend950, LayoutTag::zN, DTypeSrc, DTypeDst>(                     \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), unitFlag);                            \
    }

REGISTER_L0C_TO_L1(float, float)
REGISTER_L0C_TO_L1(float, half)
REGISTER_L0C_TO_L1(float, bf16)
// Integer route: an i32 accumulator stays i32 on the way to L1, mirroring the
// GM and UB exits.
REGISTER_L0C_TO_L1(int32_t, int32_t)

#define REGISTER_L0C_TO_UB(LayoutDst, mode, MODE, DTypeSrc, DTypeDst)                                             \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_l0c_to_ub_##LayoutDst##_##mode##_##DTypeDst(   \
        memref_t<__cc__ DTypeSrc, 1>* src, memref_t<__ubuf__ DTypeDst, 1>* dst, DESC_ABI_PARAMS(src),             \
        DESC_ABI_PARAMS(dst), uint8_t unitFlag, uint8_t subBlockId)                                               \
    {                                                                                                             \
        copyL0CToUB<                                                                                              \
            Catlass::Arch::Ascend950, LayoutTag::LayoutDst, Catlass::Gemm::Tile::CopyL0CToUBMode::MODE, DTypeSrc, \
            DTypeDst>(src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst), unitFlag, subBlockId);                  \
    }

REGISTER_L0C_TO_UB(RowMajor, nosplit, NO_SPLIT, float, float)
REGISTER_L0C_TO_UB(RowMajor, nosplit, NO_SPLIT, float, half)
REGISTER_L0C_TO_UB(RowMajor, nosplit, NO_SPLIT, float, bf16)
// split mode src=dst
REGISTER_L0C_TO_UB(RowMajor, splitm, SPLIT_M, float, float)
REGISTER_L0C_TO_UB(RowMajor, splitn, SPLIT_N, float, float)
REGISTER_L0C_TO_UB(ColumnMajor, nosplit, NO_SPLIT, float, float)
REGISTER_L0C_TO_UB(ColumnMajor, nosplit, NO_SPLIT, float, half)
REGISTER_L0C_TO_UB(ColumnMajor, nosplit, NO_SPLIT, float, bf16)
// Integer route: fixpipe carries an i32 accumulator to UB unconverted (Catlass
// selects QuantMode NoQuant for int32 -> int32). Mirrors the float set.
REGISTER_L0C_TO_UB(RowMajor, nosplit, NO_SPLIT, int32_t, int32_t)
REGISTER_L0C_TO_UB(ColumnMajor, nosplit, NO_SPLIT, int32_t, int32_t)
REGISTER_L0C_TO_UB(RowMajor, splitm, SPLIT_M, int32_t, int32_t)
REGISTER_L0C_TO_UB(RowMajor, splitn, SPLIT_N, int32_t, int32_t)

#endif
}
