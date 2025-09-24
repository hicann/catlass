
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/quant_matmul_multistage_workspace.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "catlass_test/common.hpp"

using namespace Catlass;

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementD, class LayoutD, class ElementScale, class ElementPerTokenScale>
inline TEMPLATE_RET_TYPE QuantMatmul(aclrtStream stream, GemmCoord problemShape, uint8_t *deviceA, uint8_t *deviceB, uint8_t *deviceScale, uint8_t *devicePerTokenScale, uint8_t *deviceD) {
    constexpr uint32_t workspaceStages = 2;

    using ArchTag = Arch::AtlasA2;
    constexpr uint32_t preloadStages = 1;
    constexpr uint32_t l1Stages = 2;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;
    constexpr bool enableUnitFlag = false;
    constexpr bool enableShuffleK = true;
    using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsyncWithCallback<
        preloadStages,
        l1Stages,
        l0AStages,
        l0BStages,
        l0CStages,
        enableUnitFlag,
        enableShuffleK>;
    using L1TileShape = GemmShape<128, 256, 512>;
    using L0TileShape = GemmShape<128, 256, 128>;

    using AType = Gemm::GemmType<ElementA, LayoutA>;
    using BType = Gemm::GemmType<ElementB, LayoutB>;
    using CType = Gemm::GemmType<ElementC, LayoutC>;

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    constexpr uint32_t ubStages = 2;
    using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
    using ScaleType = Gemm::GemmType<ElementScale, layout::VectorLayout>;
    using PerTokenScaleType = Gemm::GemmType<ElementPerTokenScale, layout::VectorLayout>;
    using DType = Gemm::GemmType<ElementD, layout::RowMajor>;

    using RowBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
    using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
    using OneBlkColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;

    using EpilogueTileShape = MatrixShape<32, 256>;
    using TileRowBroadcastMul = Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;
    using TileBroadcastOneBlk = Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType, EpilogueTileShape::ROW>;
    using TileOneBlkColumnBroadcastMul = Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag,
                                                                                      OneBlkColumnBroadcastMulType,
                                                                                      EpilogueTileShape>;
    using TileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, DType>;
    using TileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    using BlockEpilogue = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, CType, ScaleType, PerTokenScaleType, DType, TileRowBroadcastMul, TileBroadcastOneBlk, TileOneBlkColumnBroadcastMul, TileCopy, TileScheduler>;

    if (problemShape.m() > problemShape.n()) {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

        // kernel level
        using MatmulKernel = Gemm::Kernel::QuantMatmulMultiStageWorkspace<BlockMmad, BlockEpilogue, BlockScheduler, workspaceStages>;

        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

        typename MatmulKernel::Arguments arguments{
            problemShape, aicCoreNum, deviceA, deviceB, deviceScale, devicePerTokenScale, deviceD};

        MatmulAdapter matmulOp;
        RUN_ADAPTER(matmulOp, arguments, stream, aicCoreNum);
    } else {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

        // kernel level
        using MatmulKernel = Gemm::Kernel::QuantMatmulMultiStageWorkspace<BlockMmad, BlockEpilogue, BlockScheduler, workspaceStages>;

        using MatmulAdapter = Gemm::Device::DeviceGemm<MatmulKernel>;

        typename MatmulKernel::Arguments arguments{
            problemShape, aicCoreNum, deviceA, deviceB, deviceScale, devicePerTokenScale, deviceD};

        MatmulAdapter matmulOp;
        RUN_ADAPTER(matmulOp, arguments, stream, aicCoreNum);
    }
}