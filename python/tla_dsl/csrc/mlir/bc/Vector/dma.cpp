#include "../dma_common.h"

#include "catlass/epilogue/tile/tile_copy.hpp"

template <class ArchTag, LayoutTag SrcLayout, LayoutTag DstLayout, typename T>
CATLASS_DEVICE void copyGMToUB(
    memref_t<__gm__ T, 2>* src, memref_t<__ubuf__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc)
{
    auto srcTensor = makeGMTensor<SrcLayout, T>(src, srcDesc);
    auto dstTensor = makeUBTensor<DstLayout, T>(dst, dstDesc);
    Catlass::Epilogue::Tile::CopyGm2UbTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

template <class ArchTag, LayoutTag SrcLayout, LayoutTag DstLayout, typename T>
CATLASS_DEVICE void copyUBToGM(
    memref_t<__ubuf__ T, 1>* src, memref_t<__gm__ T, 2>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc)
{
    auto srcTensor = makeUBTensor<SrcLayout, T>(src, srcDesc);
    auto dstTensor = makeGMTensor<DstLayout, T>(dst, dstDesc);
    Catlass::Epilogue::Tile::CopyUb2GmTla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

// UB -> UB. DataCopy with DataCopyParams lowers to MOV.UB.TO.UB (arch 3510), a
// single block move that carries any element width and any row length.
//
// The move is block-based: blockLen and both gaps are counts of 32-byte blocks.
// A row whose byte length is not a whole number of blocks is rounded up, so the
// move carries the row's own 32-byte alignment fill along with the row, and the
// gaps are measured from the rounded row end. tla.copy admits only the
// geometries where that is a copy -- both origins on a block boundary, and a row
// pitch with whole-block slack for the rounded row -- which is what makes the
// block arithmetic below exact. The guard is
// _ubub_require_metaop_representable in core_api.py.
template <class ArchTag, LayoutTag SrcLayout, LayoutTag DstLayout, typename T>
CATLASS_DEVICE void copyUBToUB(
    memref_t<__ubuf__ T, 1>* src, memref_t<__ubuf__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc)
{
    AscendC::LocalTensor<T> srcTensor(AscendC::TPosition::VECCALC, localAddr(src), elementCount(src));
    AscendC::LocalTensor<T> dstTensor(AscendC::TPosition::VECCALC, localAddr(dst), elementCount(dst));

    constexpr uint32_t kBlockBytes = 32;
    // originShape is the tile's valid extent; shape is the packing it is
    // addressed through, which a tail tile pads out beyond that extent. The move
    // carries the valid elements, so it is sized from the origin, as every other
    // route here is.
    const uint32_t rows = static_cast<uint32_t>(srcDesc.originShape0);
    const uint32_t cols = static_cast<uint32_t>(srcDesc.originShape1);
    const uint32_t rowBytes = cols * static_cast<uint32_t>(sizeof(T));
    const uint32_t srcPitchBytes = static_cast<uint32_t>(srcDesc.stride0) * static_cast<uint32_t>(sizeof(T));
    const uint32_t dstPitchBytes = static_cast<uint32_t>(dstDesc.stride0) * static_cast<uint32_t>(sizeof(T));

    // The descriptor's coordinate is the tile's origin inside its parent, in
    // elements. Every other route reaches it through makeTlaTileCoord(), which
    // builds a tla::Tensor that folds the coordinate into the address; this one
    // drives DataCopy from a raw LocalTensor, so it has to apply the coordinate
    // itself. Without this a sub-block view (a padded 64x32 slice of a 64x64
    // tile, say) copies at the parent's origin instead, silently landing the
    // wrong data -- the row pitch below is honoured, but the base is not.
    const uint32_t srcElemOffset =
        static_cast<uint32_t>(static_cast<int64_t>(srcDesc.coord0) * srcDesc.stride0 + srcDesc.coord1);
    const uint32_t dstElemOffset =
        static_cast<uint32_t>(static_cast<int64_t>(dstDesc.coord0) * dstDesc.stride0 + dstDesc.coord1);

    // Rounding the row length up is what makes every row length expressible.
    // The alternative is to round down, which is what AscendC's own Level-2
    // overload does -- its header states that no error is reported in NPU mode
    // and that the value is rounded down by 32B -- and that drops the tail of
    // every row. There is no DataCopyPad for a local-to-local move to pick
    // instead, so rounding up, into the row's own alignment fill, is the only
    // direction that moves all of the bytes.
    const uint32_t rowBlocks = (rowBytes + kBlockBytes - 1) / kBlockBytes;
    const uint32_t rowBlockBytes = rowBlocks * kBlockBytes;
    // A single-row copy has no gap to express, and its pitch is allowed to be
    // narrower than the rounded row, so clamp rather than underflow.
    const uint32_t srcGapBlocks = srcPitchBytes > rowBlockBytes ? (srcPitchBytes - rowBlockBytes) / kBlockBytes : 0;
    const uint32_t dstGapBlocks = dstPitchBytes > rowBlockBytes ? (dstPitchBytes - rowBlockBytes) / kBlockBytes : 0;

    AscendC::DataCopyParams params;
    params.blockCount = static_cast<uint16_t>(rows);
    params.blockLen = static_cast<uint16_t>(rowBlocks);
    params.srcGap = static_cast<uint16_t>(srcGapBlocks);
    params.dstGap = static_cast<uint16_t>(dstGapBlocks);
    AscendC::DataCopy(dstTensor[dstElemOffset], srcTensor[srcElemOffset], params);
}

template <class ArchTag, LayoutTag SrcLayout, LayoutTag DstLayout, typename T>
CATLASS_DEVICE void copyUBToL1(
    memref_t<__ubuf__ T, 1>* src, memref_t<__cbuf__ T, 1>* dst, const TensorDesc& srcDesc, const TensorDesc& dstDesc)
{
    auto srcTensor = makeUBTensor<SrcLayout, T>(src, srcDesc);
    auto dstTensor = makeL1Tensor<DstLayout, T>(dst, dstDesc);
    Catlass::Epilogue::Tile::CopyUb2L1Tla<ArchTag, decltype(srcTensor), decltype(dstTensor)>{}(dstTensor, srcTensor);
}

// fp8 on the vector routes, staged as bytes.
//
// The AIV backend refuses fp8 outright -- "type only supports pointer
// operations, scalar float type semantics are not supported" -- and the UB
// copy templates trip that, though the cube's DMA-only ones do not. Pointer
// operations *are* allowed, so the symbol keeps the fp8 name and signature the
// route resolves to, and only the copy template is instantiated on int8_t. A
// copy does not interpret the element type beyond stride and layout
// arithmetic, and fp8 is a byte, so the generated move is the same one.
//
// This is the byte-staging workaround that every fp8 kernel used to carry by
// hand, moved down to where it belongs: the caller now sees a typed fp8 tile.
#define REGISTER_GM_TO_UB_AS_BYTES(LayoutSrc, LayoutDst, DType)                                                       \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_gm_##LayoutSrc##_to_ub_##LayoutDst##_##DType(      \
        memref_t<__gm__ DType, 2>* src, memref_t<__ubuf__ DType, 1>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst)) \
    {                                                                                                                 \
        copyGMToUB<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, int8_t>(                     \
            reinterpret_cast<memref_t<__gm__ int8_t, 2>*>(src), reinterpret_cast<memref_t<__ubuf__ int8_t, 1>*>(dst), \
            TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                                \
    }

#define REGISTER_UB_TO_GM_AS_BYTES(LayoutSrc, LayoutDst, DType)                                                       \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_ub_##LayoutSrc##_to_gm_##LayoutDst##_##DType(      \
        memref_t<__ubuf__ DType, 1>* src, memref_t<__gm__ DType, 2>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst)) \
    {                                                                                                                 \
        copyUBToGM<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, int8_t>(                     \
            reinterpret_cast<memref_t<__ubuf__ int8_t, 1>*>(src), reinterpret_cast<memref_t<__gm__ int8_t, 2>*>(dst), \
            TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                                \
    }

#define REGISTER_UB_TO_L1_AS_BYTES(LayoutSrc, LayoutDst, DType)                                                  \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_ub_##LayoutSrc##_to_l1_##LayoutDst##_##DType( \
        memref_t<__ubuf__ DType, 1>* src, memref_t<__cbuf__ DType, 1>* dst, DESC_ABI_PARAMS(src),                \
        DESC_ABI_PARAMS(dst))                                                                                    \
    {                                                                                                            \
        copyUBToL1<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, int8_t>(                \
            reinterpret_cast<memref_t<__ubuf__ int8_t, 1>*>(src),                                                \
            reinterpret_cast<memref_t<__cbuf__ int8_t, 1>*>(dst), TENSOR_DESC_12(src), TENSOR_DESC_12(dst));     \
    }

extern "C" {
#if ((defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510) || (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510))

#define REGISTER_GM_TO_UB(LayoutSrc, LayoutDst, DType)                                                                \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_gm_##LayoutSrc##_to_ub_##LayoutDst##_##DType(      \
        memref_t<__gm__ DType, 2>* src, memref_t<__ubuf__ DType, 1>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst)) \
    {                                                                                                                 \
        copyGMToUB<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, DType>(                      \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                      \
    }

#define REGISTER_UB_TO_GM(LayoutSrc, LayoutDst, DType)                                                                \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_ub_##LayoutSrc##_to_gm_##LayoutDst##_##DType(      \
        memref_t<__ubuf__ DType, 1>* src, memref_t<__gm__ DType, 2>* dst, DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst)) \
    {                                                                                                                 \
        copyUBToGM<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, DType>(                      \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                      \
    }

#define REGISTER_UB_TO_UB(LayoutSrc, LayoutDst, DType)                                                           \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_ub_##LayoutSrc##_to_ub_##LayoutDst##_##DType( \
        memref_t<__ubuf__ DType, 1>* src, memref_t<__ubuf__ DType, 1>* dst, DESC_ABI_PARAMS(src),                \
        DESC_ABI_PARAMS(dst))                                                                                    \
    {                                                                                                            \
        copyUBToUB<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, DType>(                 \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                 \
    }

#define REGISTER_UB_TO_L1(LayoutSrc, LayoutDst, DType)                                                           \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_ub_##LayoutSrc##_to_l1_##LayoutDst##_##DType( \
        memref_t<__ubuf__ DType, 1>* src, memref_t<__cbuf__ DType, 1>* dst, DESC_ABI_PARAMS(src),                \
        DESC_ABI_PARAMS(dst))                                                                                    \
    {                                                                                                            \
        copyUBToL1<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, DType>(                 \
            src, dst, TENSOR_DESC_12(src), TENSOR_DESC_12(dst));                                                 \
    }

REGISTER_GM_TO_UB(RowMajor, RowMajor, float)
REGISTER_GM_TO_UB(RowMajor, RowMajor, half)
REGISTER_GM_TO_UB(RowMajor, RowMajor, bf16)
REGISTER_GM_TO_UB(RowMajor, RowMajor, int32_t)
REGISTER_GM_TO_UB(RowMajor, RowMajor, int16_t)
REGISTER_GM_TO_UB(RowMajor, RowMajor, int8_t)
// Unsigned staging. A copy does not interpret the element type beyond stride
// and layout arithmetic, so these are the signed moves at the same widths; they
// exist as their own symbols because the route resolves by element type.
// Without them an unsigned tile cannot reach UB for Compare/Cast. u64 is
// omitted deliberately: i64 has no GM<->UB route either.
REGISTER_GM_TO_UB(RowMajor, RowMajor, uint8_t)
REGISTER_GM_TO_UB(RowMajor, RowMajor, uint16_t)
REGISTER_GM_TO_UB(RowMajor, RowMajor, uint32_t)
REGISTER_GM_TO_UB_AS_BYTES(RowMajor, RowMajor, fp8_e4m3fn_t)
REGISTER_GM_TO_UB_AS_BYTES(RowMajor, RowMajor, fp8_e5m2_t)

REGISTER_UB_TO_GM(RowMajor, RowMajor, float)
REGISTER_UB_TO_GM(RowMajor, RowMajor, half)
REGISTER_UB_TO_GM(RowMajor, RowMajor, bf16)
REGISTER_UB_TO_GM(RowMajor, RowMajor, int32_t)
REGISTER_UB_TO_GM(RowMajor, RowMajor, int16_t)
REGISTER_UB_TO_GM(RowMajor, RowMajor, int8_t)
REGISTER_UB_TO_GM(RowMajor, RowMajor, uint8_t)
REGISTER_UB_TO_GM(RowMajor, RowMajor, uint16_t)
REGISTER_UB_TO_GM(RowMajor, RowMajor, uint32_t)
REGISTER_UB_TO_GM_AS_BYTES(RowMajor, RowMajor, fp8_e4m3fn_t)
REGISTER_UB_TO_GM_AS_BYTES(RowMajor, RowMajor, fp8_e5m2_t)

// fp8 rides the route as bytes: DataCopy moves blocks, so the element type only
// has to describe the width. int8_t is registered directly.
#define REGISTER_UB_TO_UB_AS_BYTES(LayoutSrc, LayoutDst, DType)                                                  \
    [aicore] __attribute__((always_inline)) void _mlir_ciface_copy_ub_##LayoutSrc##_to_ub_##LayoutDst##_##DType( \
        memref_t<__ubuf__ DType, 1>* src, memref_t<__ubuf__ DType, 1>* dst, DESC_ABI_PARAMS(src),                \
        DESC_ABI_PARAMS(dst))                                                                                    \
    {                                                                                                            \
        copyUBToUB<Catlass::Arch::Ascend950, LayoutTag::LayoutSrc, LayoutTag::LayoutDst, int8_t>(                \
            reinterpret_cast<memref_t<__ubuf__ int8_t, 1>*>(src),                                                \
            reinterpret_cast<memref_t<__ubuf__ int8_t, 1>*>(dst), TENSOR_DESC_12(src), TENSOR_DESC_12(dst));     \
    }

REGISTER_UB_TO_UB(RowMajor, RowMajor, float)
REGISTER_UB_TO_UB(RowMajor, RowMajor, half)
REGISTER_UB_TO_UB(RowMajor, RowMajor, bf16)
REGISTER_UB_TO_UB(RowMajor, RowMajor, int32_t)
REGISTER_UB_TO_UB(RowMajor, RowMajor, int16_t)
REGISTER_UB_TO_UB(RowMajor, RowMajor, int8_t)
REGISTER_UB_TO_UB_AS_BYTES(RowMajor, RowMajor, fp8_e4m3fn_t)
REGISTER_UB_TO_UB_AS_BYTES(RowMajor, RowMajor, fp8_e5m2_t)

REGISTER_UB_TO_L1(RowMajor, zN, float)
REGISTER_UB_TO_L1(RowMajor, zN, half)
REGISTER_UB_TO_L1(RowMajor, zN, bf16)
REGISTER_UB_TO_L1(zN, zN, float)
REGISTER_UB_TO_L1(zN, zN, half)
REGISTER_UB_TO_L1(zN, zN, bf16)
REGISTER_UB_TO_L1(RowMajor, zN, int8_t)
REGISTER_UB_TO_L1(zN, zN, int8_t)
REGISTER_UB_TO_L1_AS_BYTES(RowMajor, zN, fp8_e4m3fn_t)
REGISTER_UB_TO_L1_AS_BYTES(RowMajor, zN, fp8_e5m2_t)
REGISTER_UB_TO_L1_AS_BYTES(zN, zN, fp8_e4m3fn_t)
REGISTER_UB_TO_L1_AS_BYTES(zN, zN, fp8_e5m2_t)

#endif
}
