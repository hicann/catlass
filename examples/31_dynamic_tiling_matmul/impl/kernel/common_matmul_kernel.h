#ifndef COMMON_MATMUL_KERNEL_H
#define COMMON_MATMUL_KERNEL_H

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/kernel/dynamic_matmul.hpp"
#include "catlass/gemm/gemm_type.hpp"

template <class ArchTag, class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
CATLASS_DEVICE void CommomDynamicMatmul(GemmCoord &problemShape, GemmCoord &l1TileShape, GM_ADDR gmA, LayoutA &layoutA,
    GM_ADDR gmB, LayoutB &layoutB, GM_ADDR gmC, LayoutC &layoutC, Catlass::Arch::Resource<ArchTag> &resource)
{
    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    using DispatchPolicy = Catlass::Gemm::MmadAtlasA2Dynamic<enableShuffleK, enableShuffleK>;

    using Atype = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using Btype = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using Ctype = Catlass::Gemm::GemmType<ElementC, LayoutC>;

    using BlockMmad = Catlass::Gemm::Block::BlockMmad<DispatchPolicy, void, void, AType, BType, Ctype>;
    using BlockEpilogue = void;
    if (problemShape.m() > problemShape.n()) {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        // kernel level
        using MatmulKernel = Catlass::Gemm::Kernel::DynamicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;
        typename MatmulKernel::Params params{problemShape, l1TileShape, gmA, layoutA, gmB, layoutB, gmC, layoutC};
        // call a kernel
        MatmulKernel matmul;
        matmul(params, resource);
    } else {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
        // kernel level
        using MatmulKernel = Catlass::Gemm::Kernel::DynamicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;
        typename MatmulKernel::Params params{problemShape, l1TileShape, gmA, layoutA, gmB, layoutB, gmC, layoutC};
        // call a kernel
        MatmulKernel matmul;
        matmul(params, resource);
    }
}

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
CATLASS_GLOBAL void CommomMatmulKernel(__gm__ uint8_t *__restrict__ gmA, __gm__ uint8_t *__restrict__ gmB,
    __gm__ uint8_t *__restrict__ gmB, __gm__ uint8_t *__restrict__ tilingData)
{
    using ArchTag = Arch::AtlasA2;
    Catlass::Arch::Resource<ArchTag> resource;

    uint8_t tilingParams[48];
    *(uint64_t *)(tilingParams) = *(reinterpret_cast<__gm__ uint64_t *>(tilingData));
    *(uint32_t *)(tilingParams + 8) = *(reinterpret_cast<__gm__ uint64_t *>(tilingData + 8));
    *(uint64_t *)(tilingParams + 12) = *(reinterpret_cast<__gm__ uint64_t *>(tilingData + 16));
    *(uint64_t *)(tilingParams + 20) = *(reinterpret_cast<__gm__ uint64_t *>(tilingData + 24));
    *(uint64_t *)(tilingParams + 28) = *(reinterpret_cast<__gm__ uint64_t *>(tilingData + 32));
    *(uint64_t *)(tilingParams + 36) = *(reinterpret_cast<__gm__ uint64_t *>(tilingData + 40));

    // Parase the tiling parameters
    uint32_t m = *(reinterpret_cast<uint32_t *>(tilingParams));
    uint32_t n = *(reinterpret_cast<uint32_t *>(tilingParams + 4));
    uint32_t k = *(reinterpret_cast<uint32_t *>(tilingParams + 8));
    int64_t strideA = static_cast<int64_t>(*(reinterpret_cast<uint64_t *>(tilingParams + 12)));
    int64_t strideB = static_cast<int64_t>(*(reinterpret_cast<uint64_t *>(tilingParams + 20)));
    int64_t strideC = static_cast<int64_t>(*(reinterpret_cast<uint64_t *>(tilingParams + 28)));

    // To save space, tiling parameters (m1, n1, k1) are stored as uint8_t. Though uint8_t has a
    // maximum value of 255, since these values are always multiples of 16, they can be safely divided
    // by 16 before storage, ensuring they fit within the uint8_t range.
    // To restore the original m1, n1, k1, multiply them by 16.

    uint32_t m1 = *(reinterpret_cast<uint8_t *>(tilingParams + 38)) * 16;
    uint32_t n1 = *(reinterpret_cast<uint8_t *>(tilingParams + 39)) * 16;
    uint32_t k1 = *(reinterpret_cast<uint8_t *>(tilingParams + 40)) * 16;

    GemmCoord problemShape(m, n, k);
    GemmCoord l1TileShape(m1, n1, k1);
    LayoutA layoutA{m, k, strideA};
    LayoutB layoutB{k, n, strideB};
    LayoutC layoutC{m, n, strideC};
    CommonDynamicMatmul<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>(
        problemShape, l1TileShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, resource);
}

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
void LaunchCommonMatmulKernel(aclrtStream &stream, uint64_t fftsAddr, uint8_t *dA, uint8_t *dB, uint8_t *dC,
    uint8_t *dTilingParams, TilingParams &tilingParams)
{
    CommonMatmulKernel<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC>
        <<<tilingParams.blockDim, nullptr, stream>>>(dA, dB, dC, dTilingParams);
}

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
size_t CommonMatmulKernelGetWorkspaceSize(TilingParams& tilingParams) {
    return 0;
}

#endif  // COMMON_MATMUL_KERNEL_H