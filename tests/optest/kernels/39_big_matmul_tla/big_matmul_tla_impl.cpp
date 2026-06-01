#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_scheduler_l2_misplace_core.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/basic_matmul_tla.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include "catlass_kernel.h"
#include "common/kernel_runner.h"
#include "common/tile_shape_scaler.h"
#include "common/tile_shape_scaler_tla.h"

#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A half
#endif
#ifndef CATLASS_JIT_ELEMENT_B
#define CATLASS_JIT_ELEMENT_B half
#endif
#ifndef CATLASS_JIT_ELEMENT_C
#define CATLASS_JIT_ELEMENT_C half
#endif
#ifndef CATLASS_JIT_LAYOUT_A
#define CATLASS_JIT_LAYOUT_A RowMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_B
#define CATLASS_JIT_LAYOUT_B RowMajor
#endif
#ifndef CATLASS_JIT_LAYOUT_C
#define CATLASS_JIT_LAYOUT_C RowMajor
#endif

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementC = CATLASS_JIT_ELEMENT_C;

using LayoutTagA = Catlass::layout::CATLASS_JIT_LAYOUT_A;
using LayoutTagB = Catlass::layout::CATLASS_JIT_LAYOUT_B;
using LayoutTagC = Catlass::layout::CATLASS_JIT_LAYOUT_C;

using ArchTag = Catlass::Arch::AtlasA2;
using DispatchPolicy = Catlass::Gemm::MmadPingpong<ArchTag, true>;

using L1TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, half, tla::tuple<tla::C<128>, tla::C<256>, tla::C<256>>>::type;
using L0TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, half, tla::tuple<tla::C<128>, tla::C<256>, tla::C<64>>>::type;

using TileCopy =
    Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC>;
using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<
    DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
using BlockEpilogue = void;
using BlockScheduler = typename Catlass::Gemm::Block::BlockSchedulerL2MisplaceCore<
    tla::detail::isRowMajor<LayoutTagA>::value, tla::detail::isRowMajor<LayoutTagB>::value>;

using MatmulKernel = Catlass::Gemm::Kernel::BasicMatmulTla<BlockMmad, BlockEpilogue, BlockScheduler>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    Catlass::GemmCoord shape{params->m, params->n, params->k};

    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(shape.m(), shape.k());
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(shape.k(), shape.n());
    auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(shape.m(), shape.n());

    typename MatmulKernel::Arguments arguments{
        shape, params->inputAddr[0], layoutA, params->inputAddr[1], layoutB, params->outputAddr[0], layoutC};

    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
