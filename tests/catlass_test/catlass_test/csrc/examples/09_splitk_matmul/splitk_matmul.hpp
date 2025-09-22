#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/splitk_matmul.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "catlass_test/common.hpp"
template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
inline TEMPLATE_RET_TYPE SplitkMatmul(aclrtStream stream, GemmCoord problemShape, uint8_t *deviceA, uint8_t *deviceB, uint8_t *deviceC) {
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;
    using L1TileShape = GemmShape<128, 256, 256>;
    using L0TileShape = GemmShape<128, 256, 64>;

    using AType = Gemm::GemmType<half, LayoutA>;
    using BType = Gemm::GemmType<half, LayoutB>;
    using CType = Gemm::GemmType<float, LayoutC>;

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockEpilogue = void;

    // After the Matmul computation is completed, launch the ReduceAdd kernel to accumulate the partial sums.
    constexpr uint32_t computeLength = 32 * 1024 / sizeof(float);
    using ReduceAdd = Catlass::Gemm::Kernel::ReduceAdd<ArchTag, float, half, computeLength>;

    // Swizzle offset is 3 and direction is 0.
    using BlockScheduler = typename Gemm::Block::SplitkGemmIdentityBlockSwizzle<3, 0>;

    // kernel level
    using MatmulKernel = Gemm::Kernel::SplitkMatmul<BlockMmad, BlockEpilogue, BlockScheduler, ReduceAdd>;

    using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
    MatmulKernel::Arguments arguments{problemShape,
                                      aicCoreNum,
                                      sizeof(float),
                                      deviceA,
                                      deviceB,
                                      deviceC};
    MatmulAdapter matmulOp;
    RUN_ADAPTER(matmulOp, arguments, stream, aicCoreNum);
}