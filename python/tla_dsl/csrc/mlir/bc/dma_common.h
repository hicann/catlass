#pragma once

#include "common.h"

#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

using bf16 = bfloat16_t;

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
    return tla::MakeLayout(tla::MakeShape(desc.shape0, desc.shape1),
                           tla::MakeStride(desc.stride0, tla::Int<1>{}),
                           tla::MakeShape(desc.originShape0, desc.originShape1));
}

template <typename TensorDesc>
CATLASS_DEVICE auto makeTlaTileCoord(const TensorDesc &desc) {
    return tla::MakeCoord(desc.coord0, desc.coord1);
}

template <typename T>
CATLASS_DEVICE auto makezNTlaLayout(const TensorDesc4D &desc) {
    constexpr uint32_t eleNumPerC0 = Catlass::BYTE_PER_C0 / sizeof(T);
    constexpr uint32_t eleNumPerFractal = Catlass::BYTE_PER_FRACTAL / sizeof(T);
    return tla::MakeLayout(
        tla::MakeShape(
            tla::MakeShape(tla::Int<Catlass::C0_NUM_PER_FRACTAL>{}, desc.shape1),
            tla::MakeShape(tla::Int<eleNumPerC0>{}, desc.shape3)),
        tla::MakeStride(
            tla::MakeStride(tla::Int<eleNumPerC0>{}, tla::Int<eleNumPerFractal>{}),
            tla::MakeStride(tla::Int<1>{}, desc.stride3)),
        tla::MakeShape(desc.originShape0, desc.originShape1));
}

template <typename T>
CATLASS_DEVICE auto makeZzTlaLayout(const TensorDesc4D &desc) {
    constexpr uint32_t eleNumPerC0 = Catlass::BYTE_PER_C0 / sizeof(T);
    constexpr uint32_t eleNumPerFractal = Catlass::BYTE_PER_FRACTAL / sizeof(T);
    return tla::MakeLayout(
        tla::MakeShape(
            tla::MakeShape(tla::Int<Catlass::C0_NUM_PER_FRACTAL>{}, desc.shape1),
            tla::MakeShape(tla::Int<eleNumPerC0>{}, desc.shape3)),
        tla::MakeStride(
            tla::MakeStride(tla::Int<eleNumPerC0>{}, desc.stride1),
            tla::MakeStride(tla::Int<1>{}, tla::Int<eleNumPerFractal>{})),
        tla::MakeShape(desc.originShape0, desc.originShape1));
}

template <typename T>
CATLASS_DEVICE auto makenZTlaLayout(const TensorDesc4D &desc) {
    constexpr uint32_t eleNumPerC0 = Catlass::BYTE_PER_C0 / sizeof(T);
    constexpr uint32_t eleNumPerFractal = Catlass::BYTE_PER_FRACTAL / sizeof(T);
    return tla::MakeLayout(
        tla::MakeShape(
            tla::MakeShape(tla::Int<eleNumPerC0>{}, desc.shape1),
            tla::MakeShape(tla::Int<Catlass::C0_NUM_PER_FRACTAL>{}, desc.shape3)),
        tla::MakeStride(
            tla::MakeStride(tla::Int<1>{}, desc.stride1),
            tla::MakeStride(tla::Int<eleNumPerC0>{}, tla::Int<eleNumPerFractal>{})),
        tla::MakeShape(desc.originShape0, desc.originShape1));
}

template <typename T>
CATLASS_DEVICE auto makeL0CTlaLayout(const TensorDesc4D &desc) {
    constexpr uint32_t eleNumPerFractal = 256;
    return tla::MakeLayout(
        tla::MakeShape(
            tla::MakeShape(tla::Int<Catlass::C0_NUM_PER_FRACTAL>{}, desc.shape1),
            tla::MakeShape(tla::Int<Catlass::C0_NUM_PER_FRACTAL>{}, desc.shape3)),
        tla::MakeStride(
            tla::MakeStride(tla::Int<Catlass::C0_NUM_PER_FRACTAL>{}, tla::Int<eleNumPerFractal>{}),
            tla::MakeStride(tla::Int<1>{}, desc.stride3)),
        tla::MakeShape(desc.originShape0, desc.originShape1));
}

template <typename T>
CATLASS_DEVICE auto makeGMTensor(memref_t<__gm__ T, 2> *memref, const TensorDesc2D &desc) {
    return tla::MakeTensor(makeGlobalTensor(basePtr(memref)), makeRowMajorTlaLayout<T>(desc),
                           makeTlaTileCoord(desc), Catlass::Arch::PositionGM{});
}

template <typename T, size_t Dim>
CATLASS_DEVICE auto makeUBTensor(memref_t<__ubuf__ T, Dim> *memref, const TensorDesc2D &desc) {
    return tla::MakeTensor(AscendC::LocalTensor<T>(AscendC::TPosition::VECCALC, localAddr(memref),
                                 elementCount(memref)),
                           makeRowMajorTlaLayout<T>(desc), makeTlaTileCoord(desc),
                           Catlass::Arch::PositionUB{});
}

CATLASS_DEVICE auto makeColumnMajorTlaLayout(const TensorDesc2D &desc) {
    return tla::MakeLayout(tla::MakeShape(desc.shape0, desc.shape1),
                           tla::MakeStride(tla::Int<1>{}, desc.stride1),
                           tla::MakeShape(desc.originShape0, desc.originShape1));
}

template <typename T>
CATLASS_DEVICE auto makeGMTensorColumnMajor(memref_t<__gm__ T, 2> *memref, const TensorDesc2D &desc) {
    return tla::MakeTensor(makeGlobalTensor(basePtr(memref)), makeColumnMajorTlaLayout(desc),
                           makeTlaTileCoord(desc), Catlass::Arch::PositionGM{});
}

template <typename T>
CATLASS_DEVICE auto makeGMzNTensor(memref_t<__gm__ T, 2> *memref, const TensorDesc4D &desc) {
    return tla::MakeTensor(makeGlobalTensor(basePtr(memref)), makezNTlaLayout<T>(desc),
                           makeTlaTileCoord(desc), Catlass::Arch::PositionGM{});
}

template <typename T, size_t L1Dim>
CATLASS_DEVICE auto makeL1Tensor(memref_t<__cbuf__ T, L1Dim> *memref, const TensorDesc4D &desc) {
    return tla::MakeTensor(
        AscendC::LocalTensor<T>(AscendC::TPosition::A1, localAddr(memref), elementCount(memref)),
        makezNTlaLayout<T>(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionL1{});
}

template <typename T, size_t L1Dim>
CATLASS_DEVICE auto makeL1nZTensor(memref_t<__cbuf__ T, L1Dim> *memref, const TensorDesc4D &desc) {
    return tla::MakeTensor(
        AscendC::LocalTensor<T>(AscendC::TPosition::A1, localAddr(memref), elementCount(memref)),
        makenZTlaLayout<T>(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionL1{});
}

template <typename T>
CATLASS_DEVICE auto makeL0ATensor(memref_t<__ca__ T, 1> *memref, const TensorDesc4D &desc) {
    return tla::MakeTensor(
        AscendC::LocalTensor<T>(AscendC::TPosition::A2, localAddr(memref), elementCount(memref)),
        makezNTlaLayout<T>(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionL0A{});
}

template <typename T>
CATLASS_DEVICE auto makeL0BTensor(memref_t<__cb__ T, 1> *memref, const TensorDesc4D &desc) {
    return tla::MakeTensor(
        AscendC::LocalTensor<T>(AscendC::TPosition::B2, localAddr(memref), elementCount(memref)),
        makenZTlaLayout<T>(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionL0B{});
}

template <typename T>
CATLASS_DEVICE auto makeL0CTensor(memref_t<__cc__ T, 1> *memref, const TensorDesc4D &desc) {
    return tla::MakeTensor(
        AscendC::LocalTensor<T>(AscendC::TPosition::CO1, localAddr(memref), elementCount(memref)),
        makeL0CTlaLayout<T>(desc), makeTlaTileCoord(desc), Catlass::Arch::PositionL0C{});
}
