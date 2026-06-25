#include "../common.h"

#include "catlass/arch/arch.hpp"
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace {

struct TensorDesc2D {
    uint32_t shape0;
    uint32_t shape1;
    int64_t stride0;
    int64_t stride1;
    uint32_t coord0;
    uint32_t coord1;
    uint32_t originShape0;
    uint32_t originShape1;
};

struct TensorDesc4D {
    uint32_t shape0;
    uint32_t shape1;
    uint32_t shape2;
    uint32_t shape3;
    int64_t stride0;
    int64_t stride1;
    int64_t stride2;
    int64_t stride3;
    uint32_t coord0;
    uint32_t coord1;
    uint32_t originShape0;
    uint32_t originShape1;
};

template <typename T, size_t Dim>
CATLASS_DEVICE T *basePtr(memref_t<T, Dim> *memref) {
    return memref->aligned + memref->offset;
}

template <typename T, size_t Dim>
CATLASS_DEVICE uint32_t localAddr(memref_t<T, Dim> *memref) {
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(basePtr(memref)));
}

template <typename T, size_t Dim>
CATLASS_DEVICE uint32_t elementCount(memref_t<T, Dim> *memref) {
    uint32_t n = 1;
    for (size_t i = 0; i < Dim; ++i)
        n *= static_cast<uint32_t>(memref->sizes[i]);
    return n;
}

template <typename T>
CATLASS_DEVICE AscendC::GlobalTensor<T> makeGlobalTensor(__gm__ T *ptr) {
    AscendC::GlobalTensor<T> tensor;
    tensor.SetGlobalBuffer(ptr);
    return tensor;
}

template <typename T>
CATLASS_DEVICE auto makeRowMajorTlaLayout(const TensorDesc2D &desc) {
    // Row-major GM descriptors describe a tile view into an origin tensor.
    // Preserve origin shape/stride here; coord selects the viewed tile.
    return tla::MakeLayout(tla::MakeShape(desc.shape0, desc.shape1),
                           tla::MakeStride(desc.stride0, tla::Int<1>{}),
                           tla::MakeShape(desc.originShape0,
                                          desc.originShape1));
}

template <typename TensorDesc>
CATLASS_DEVICE auto makeTlaTileCoord(const TensorDesc &desc) {
    return tla::MakeCoord(desc.coord0, desc.coord1);
}

template <typename T>
CATLASS_DEVICE auto makezNTlaLayout(const TensorDesc4D &desc) {
    constexpr uint32_t eleNumPerC0 = Catlass::BYTE_PER_C0 / sizeof(T);
    constexpr uint32_t eleNumPerFractal =
        Catlass::BYTE_PER_FRACTAL / sizeof(T);
    return tla::MakeLayout(
        tla::MakeShape(
            tla::MakeShape(tla::Int<Catlass::C0_NUM_PER_FRACTAL>{},
                           desc.shape1),
            tla::MakeShape(tla::Int<eleNumPerC0>{}, desc.shape3)),
        tla::MakeStride(
            tla::MakeStride(tla::Int<eleNumPerC0>{},
                            tla::Int<eleNumPerFractal>{}),
            tla::MakeStride(tla::Int<1>{}, desc.stride3)),
        tla::MakeShape(desc.originShape0, desc.originShape1));
}

template <typename T>
CATLASS_DEVICE auto makeZzTlaLayout(const TensorDesc4D &desc) {
    constexpr uint32_t eleNumPerC0 = Catlass::BYTE_PER_C0 / sizeof(T);
    constexpr uint32_t eleNumPerFractal =
        Catlass::BYTE_PER_FRACTAL / sizeof(T);
    return tla::MakeLayout(
        tla::MakeShape(
            tla::MakeShape(tla::Int<Catlass::C0_NUM_PER_FRACTAL>{},
                           desc.shape1),
            tla::MakeShape(tla::Int<eleNumPerC0>{}, desc.shape3)),
        tla::MakeStride(
            tla::MakeStride(tla::Int<eleNumPerC0>{}, desc.stride1),
            tla::MakeStride(tla::Int<1>{},
                            tla::Int<eleNumPerFractal>{})),
        tla::MakeShape(desc.originShape0, desc.originShape1));
}

template <typename T>
CATLASS_DEVICE auto makenZTlaLayout(const TensorDesc4D &desc) {
    constexpr uint32_t eleNumPerC0 = Catlass::BYTE_PER_C0 / sizeof(T);
    constexpr uint32_t eleNumPerFractal =
        Catlass::BYTE_PER_FRACTAL / sizeof(T);
    return tla::MakeLayout(
        tla::MakeShape(
            tla::MakeShape(tla::Int<eleNumPerC0>{}, desc.shape1),
            tla::MakeShape(tla::Int<Catlass::C0_NUM_PER_FRACTAL>{},
                           desc.shape3)),
        tla::MakeStride(
            tla::MakeStride(tla::Int<1>{}, desc.stride1),
            tla::MakeStride(tla::Int<eleNumPerC0>{},
                            tla::Int<eleNumPerFractal>{})),
        tla::MakeShape(desc.originShape0, desc.originShape1));
}

template <typename T>
CATLASS_DEVICE auto makeL0CTlaLayout(const TensorDesc4D &desc) {
    constexpr uint32_t eleNumPerFractal = 256;
    return tla::MakeLayout(
        tla::MakeShape(
            tla::MakeShape(tla::Int<Catlass::C0_NUM_PER_FRACTAL>{},
                           desc.shape1),
            tla::MakeShape(tla::Int<Catlass::C0_NUM_PER_FRACTAL>{},
                           desc.shape3)),
        tla::MakeStride(
            tla::MakeStride(tla::Int<Catlass::C0_NUM_PER_FRACTAL>{},
                            tla::Int<eleNumPerFractal>{}),
            tla::MakeStride(tla::Int<1>{}, desc.stride3)),
        tla::MakeShape(desc.originShape0, desc.originShape1));
}

template <typename T>
CATLASS_DEVICE auto makeGMTensor(memref_t<__gm__ T, 2> *memref,
                                 const TensorDesc2D &desc) {
    return tla::MakeTensor(makeGlobalTensor(basePtr(memref)),
                           makeRowMajorTlaLayout<T>(desc),
                           makeTlaTileCoord(desc),
                           Catlass::Arch::PositionGM{});
}

CATLASS_DEVICE auto makeColumnMajorTlaLayout(const TensorDesc2D &desc) {
    return tla::MakeLayout(tla::MakeShape(desc.shape0, desc.shape1),
                           tla::MakeStride(tla::Int<1>{}, desc.stride1),
                           tla::MakeShape(desc.originShape0, desc.originShape1));
}

template <typename T>
CATLASS_DEVICE auto makeGMTensorColumnMajor(memref_t<__gm__ T, 2> *memref,
                                          const TensorDesc2D &desc) {
    return tla::MakeTensor(makeGlobalTensor(basePtr(memref)),
                           makeColumnMajorTlaLayout(desc),
                           makeTlaTileCoord(desc),
                           Catlass::Arch::PositionGM{});
}

template <typename T>
CATLASS_DEVICE auto makeGMzNTensor(memref_t<__gm__ T, 2> *memref, const TensorDesc4D &desc) {
    return tla::MakeTensor(makeGlobalTensor(basePtr(memref)), makezNTlaLayout<T>(desc),
                           makeTlaTileCoord(desc), Catlass::Arch::PositionGM{});
}

// L1/cbuf/L0: TlaCompile passes rank-1 dynamic strided memrefs (base e.g. memref<1024xf32, …>).
// GM operands for row-major tiles stay rank 2 (memref<?x?xf32, strided<[?,?], …>, gm>).
template <typename T, size_t L1Dim>
CATLASS_DEVICE auto makeL1Tensor(memref_t<__cbuf__ T, L1Dim> *memref,
                                   const TensorDesc4D &desc) {
    return tla::MakeTensor(
        AscendC::LocalTensor<T>(AscendC::TPosition::A1, localAddr(memref),
                                elementCount(memref)),
        makezNTlaLayout<T>(desc),
        makeTlaTileCoord(desc),
        Catlass::Arch::PositionL1{});
}

template <typename T, size_t L1Dim>
CATLASS_DEVICE auto makeL1nZTensor(memref_t<__cbuf__ T, L1Dim> *memref,
                                   const TensorDesc4D &desc) {
    return tla::MakeTensor(
        AscendC::LocalTensor<T>(AscendC::TPosition::A1, localAddr(memref),
                                elementCount(memref)),
        makenZTlaLayout<T>(desc),
        makeTlaTileCoord(desc),
        Catlass::Arch::PositionL1{});
}

template <typename T>
CATLASS_DEVICE auto makeL0ATensor(memref_t<__ca__ T, 1> *memref,
                                  const TensorDesc4D &desc) {
    // Frontend/lowering expose CA as a logical zN route, but Catlass
    // currently materializes the supported physical L0A layout as zZ.
    return tla::MakeTensor(
        AscendC::LocalTensor<T>(AscendC::TPosition::A2, localAddr(memref),
                                elementCount(memref)),
        makezNTlaLayout<T>(desc),
        makeTlaTileCoord(desc),
        Catlass::Arch::PositionL0A{});
}

template <typename T>
CATLASS_DEVICE auto makeL0BTensor(memref_t<__cb__ T, 1> *memref,
                                  const TensorDesc4D &desc) {
    // Frontend/lowering expose CB as a logical zN route, but Catlass
    // currently materializes the supported physical L0B layout as nZ.
    return tla::MakeTensor(
        AscendC::LocalTensor<T>(AscendC::TPosition::B2, localAddr(memref),
                                elementCount(memref)),
        makenZTlaLayout<T>(desc),
        makeTlaTileCoord(desc),
        Catlass::Arch::PositionL0B{});
}

template <typename T>
CATLASS_DEVICE auto makeL0CTensor(memref_t<__cc__ T, 1> *memref,
                                  const TensorDesc4D &desc) {
    return tla::MakeTensor(
        AscendC::LocalTensor<T>(AscendC::TPosition::CO1, localAddr(memref),
                                elementCount(memref)),
        makeL0CTlaLayout<T>(desc),
        makeTlaTileCoord(desc),
        Catlass::Arch::PositionL0C{});
}

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

template <class ArchTag, typename T>
CATLASS_DEVICE void copyCcToGmzN(memref_t<__cc__ T, 1> *src, memref_t<__gm__ T, 2> *dst,
                                 const TensorDesc4D &srcDesc, const TensorDesc4D &dstDesc) {
    auto srcTensor = makeL0CTensor(src, srcDesc);
    auto dstTensor = makeGMzNTensor(dst, dstDesc);
    Catlass::Gemm::Tile::CopyL0CToGmTla<ArchTag, decltype(srcTensor),
                                        decltype(dstTensor)>{}(dstTensor,
                                                               srcTensor);
}

} // namespace

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
