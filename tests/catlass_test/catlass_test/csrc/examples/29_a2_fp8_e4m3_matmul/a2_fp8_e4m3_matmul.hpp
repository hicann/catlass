#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/fp8_matmul.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "catlass_test/common.hpp"
using namespace Catlass;

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
inline TEMPLATE_RET_TYPE A2Fp8E4m3Matmul(aclrtStream stream, GemmCoord problemShape, uint8_t *deviceA, uint8_t *deviceB, uint8_t *deviceC) {
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadAtlasA2PingpongSliceK<true>;
    using L1TileShape = GemmShape<128, 256, 256>;
    using L0TileShape = GemmShape<128, 256, 64>;

    using AType = Gemm::GemmType<half, LayoutA>;
    using BType = Gemm::GemmType<half, LayoutB>;
    using CType = Gemm::GemmType<float, LayoutC>; // 原子加以float类型进行累加

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockEpilogue = void;

    if (options.problemShape.m() > options.problemShape.n()) {
        // Swizzle offset is 3 and direction is 0.
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

        // kernel level
        using MatmulKernel =
            Gemm::Kernel::FP8Matmul<BlockMmad, BlockEpilogue, BlockScheduler, mScalar, nScalar, splitkLength>;

        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        typename MatmulKernel::Arguments arguments{
            options.problemShape, deviceA, deviceB, deviceC, deviceWA, deviceWB, deviceWC, scalar, zeroPoint};
        MatmulAdapter matmulOp;
        RUN_ADAPTER(matmulOp, arguments, stream, aicCoreNum);
    } else {
        // Swizzle offset is 3 and direction is 1.
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

        // kernel level
        using MatmulKernel =
            Gemm::Kernel::FP8Matmul<BlockMmad, BlockEpilogue, BlockScheduler, mScalar, nScalar, splitkLength>;

        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        typename MatmulKernel::Arguments arguments{
            options.problemShape, deviceA, deviceB, deviceC, deviceWA, deviceWB, deviceWC, scalar, zeroPoint};
        MatmulAdapter matmulOp;
        RUN_ADAPTER(matmulOp, arguments, stream, aicCoreNum);
    }
}