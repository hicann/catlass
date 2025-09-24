#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/matmul_epilogue.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "catlass_test/common.hpp"

using namespace Catlass;

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC>
inline TEMPLATE_RET_TYPE MatmulAdd(aclrtStream stream, GemmCoord problemShape, uint8_t *deviceA, uint8_t *deviceB, uint8_t *deviceC) {

    // Define ArchTag
    using ArchTag = Arch::AtlasA2;

    // Block level, define BlockMmad
    constexpr bool enableUnitFlag = true;
    using MmadDispatchPolicy = Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
    using L1TileShape = GemmShape<128, 256, 256>;
    using L0TileShape = GemmShape<128, 256, 64>;
    using AType = Gemm::GemmType<ElementA, LayoutA>;
    using BType = Gemm::GemmType<ElementB, LayoutB>;
    using CType = Gemm::GemmType<ElementC, LayoutC>;
    using BlockMmad = Gemm::Block::BlockMmad<MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    // Block level, define BlockEpilogue
    using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2ElemWiseOneSource;
    using XType = CType;
    using DType = CType;
    using ComputeType = CType;
    constexpr uint32_t computeLength = 16384;
    constexpr uint32_t elementSize = Max<sizeof(ElementA), sizeof(ElementB), sizeof(ElementC)>::value;
    using TileElemWiseEpilogue = Epilogue::Tile::TileElemWiseAdd<ArchTag, ComputeType, computeLength>;
    using EpilogueTileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, XType, DType>;
    using BlockEpilogue = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, CType, XType, DType, TileElemWiseEpilogue, EpilogueTileCopy>;
    if (problemShape.m() > problemShape.n()) {
        // Define BlockScheduler
        // Swizzle offset is 3 and direction is 0.
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        // Kernel level
        using MatmulKernel = Gemm::Kernel::MatmulEpilogue<BlockMmad, BlockEpilogue, BlockScheduler>;
        // Prepare params
        typename MatmulKernel::Arguments arguments{problemShape, elementSize, deviceA, deviceB, deviceC};
        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        MatmulAdapter matmulOp;
        RunAdapter(matmulOp, arguments, stream, aicCoreNum, fftsAddr);
    } else {
        // Define BlockScheduler
        // Swizzle offset is 3 and direction is 1.
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
        // Kernel level
        using MatmulKernel = Gemm::Kernel::MatmulEpilogue<BlockMmad, BlockEpilogue, BlockScheduler>;
        // Prepare params
        typename MatmulKernel::Arguments arguments{problemShape, elementSize, deviceA, deviceB, deviceC};
        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;
        MatmulAdapter matmulOp;
        RUN_ADAPTER(matmulOp, arguments, stream, aicCoreNum, fftsAddr);
    }
}