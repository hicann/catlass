
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/grouped_matmul_slice_k.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "catlass_test/common.hpp"

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC, class ElementGroupList>
inline TEMPLATE_RET_TYPE GroupedMatmulSliceK(aclrtStream stream, GemmCoord problemShape, uint32_t problemCount, uint8_t *deviceA, uint8_t *deviceB, uint8_t *deviceGroupList, uint8_t *deviceC) {
    constexpr uint32_t preloadStages = 1;
    constexpr uint32_t l1Stages = 2;
    constexpr uint32_t l0AStages = 4;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;
    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;

    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsync<
        preloadStages,
        l1Stages,
        l0AStages,
        l0BStages,
        l0CStages,
        enableUnitFlag,
        enableShuffleK>;
    using L1TileShape = GemmShape<128, 256, 256>;
    using L0TileShape = GemmShape<128, 256, 64>;

    using AType = Gemm::GemmType<half, LayoutA>;
    using BType = Gemm::GemmType<half, LayoutB>;
    using CType = Gemm::GemmType<half, LayoutC>;

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockEpilogue = void;
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

    // kernel level
    using MatmulKernel = Gemm::Kernel::GroupedMatmulSliceK<BlockMmad, BlockEpilogue, BlockScheduler, int64_t>;

    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

    MatmulKernel::Arguments arguments{
        problemShape, problemCount, deviceGroupList, deviceA, deviceB, deviceC};

    // call a kernel
    MatmulAdapter matmulOp;
    RUN_ADAPTER(matmulOp, arguments, stream, aicCoreNum);
}