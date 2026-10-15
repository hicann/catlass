/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue_matmul_permute.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_scheduler_aswt.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/matmul_mix_fixpipe_opti_matmul_permute.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"

#include "catlass_kernel.h"
#include "common/kernel_runner.h"
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

using namespace Catlass;
using namespace tla;

constexpr uint32_t S1 = 8;
constexpr uint32_t S2 = 4;

enum class MatmulPermuteKernelPreset
{
    Basic,
    VectorPack,
    AswtWindow
};

static MatmulPermuteKernelPreset SelectMatmulPermuteKernelPreset(uint32_t m, uint32_t n)
{
    if (m >= 1024 && n >= 1024) {
        return MatmulPermuteKernelPreset::VectorPack;
    }
    if (m >= 1024 && n >= 512) {
        return MatmulPermuteKernelPreset::AswtWindow;
    }
    return MatmulPermuteKernelPreset::Basic;
}

static_assert(S1 > 0 && S2 > 0, "MatmulPermute requires positive S1/S2.");

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
using ElementC = CATLASS_JIT_ELEMENT_C;
using ElementBias = void;

using LayoutTagA = layout::CATLASS_JIT_LAYOUT_A;
using LayoutTagB = layout::CATLASS_JIT_LAYOUT_B;
using LayoutTagC = layout::CATLASS_JIT_LAYOUT_C;

using ArchTag = Arch::Ascend950;
constexpr bool enableUnitFlag = true;
constexpr bool useHF32 = false;
using DispatchPolicy = Gemm::MmadPingpong<ArchTag, enableUnitFlag, useHF32>;

using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
constexpr bool splitM = std::is_same_v<ElementC, ElementAccumulator>;
constexpr auto copyMode = splitM ? Gemm::Tile::CopyL0CToUBMode::SPLIT_M : Gemm::Tile::CopyL0CToUBMode::NO_SPLIT;

using TileCopy = Gemm::Tile::PackedTileCopyTlaToUB<
    ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, void, copyMode>;
using TileMmad = Gemm::Tile::TileMmadTla<DispatchPolicy::ArchTag, ElementA, TileCopy::LayoutTagL1A>;

template <bool UseVectorPack, class Scheduler, class L1Shape, class L0Shape>
struct MatmulPermuteKernelSelector {
    using L1TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, half, L1Shape>::type;
    using L0TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, half, L0Shape>::type;
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, ElementBias, TileCopy, TileMmad>;
    using BlockEpilogue = Epilogue::Block::BlockEpilogue<
        Epilogue::EpilogueAscend950MatmulPermute<S1, S2, splitM, UseVectorPack>, ElementC, ElementC>;
    using BlockScheduler = Scheduler;
    using Kernel = Gemm::Kernel::KernelMatmulMixFixpipeOptiMatmulPermute<BlockMmad, BlockEpilogue, BlockScheduler>;
};

using BasicL1Base = tuple<C<64>, C<256>, C<128>>;
using BasicL0Base = tuple<C<64>, C<256>, C<64>>;
using VectorL1Base = tuple<C<128>, C<256>, C<128>>;
using VectorL0Base = tuple<C<128>, C<256>, C<64>>;
using BasicL1TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, half, BasicL1Base>::type;
using BasicL0TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, half, BasicL0Base>::type;
using VectorL1TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, half, VectorL1Base>::type;
using VectorL0TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, half, VectorL0Base>::type;
using BasicScheduler = Gemm::Block::BlockSchedulerAswt<BasicL1TileShape, BasicL0TileShape>;
using VectorScheduler = Gemm::Block::BlockSchedulerAswt<VectorL1TileShape, VectorL0TileShape>;
using WindowScheduler = Gemm::Block::BlockSchedulerAswt<VectorL1TileShape, VectorL0TileShape>;
using BasicKernelSelector = MatmulPermuteKernelSelector<false, BasicScheduler, BasicL1Base, BasicL0Base>;
using VectorKernelSelector = MatmulPermuteKernelSelector<true, VectorScheduler, VectorL1Base, VectorL0Base>;
using WindowKernelSelector = MatmulPermuteKernelSelector<false, WindowScheduler, VectorL1Base, VectorL0Base>;

template <class KernelSelector>
static void Run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    using MatmulKernel = typename KernelSelector::Kernel;
    using L1TileShape = typename KernelSelector::L1TileShape;
    typename MatmulKernel::Arguments arguments{
        GemmCoord{params->m, params->n, params->k}, params->inputAddr[0], params->inputAddr[1], params->outputAddr[0]};
    uint32_t taskNum = CeilDiv(params->m, tla::get<0>(L1TileShape{})) * CeilDiv(params->n, tla::get<1>(L1TileShape{}));
    uint32_t coreNum = taskNum < blockNum ? taskNum : blockNum;
    Catlass::RunKernel<MatmulKernel>(arguments, stream, coreNum);
}

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    const auto preset = SelectMatmulPermuteKernelPreset(params->m, params->n);
    switch (preset) {
        case MatmulPermuteKernelPreset::Basic:
            Run<BasicKernelSelector>(blockNum, stream, params);
            break;
        case MatmulPermuteKernelPreset::VectorPack:
            Run<VectorKernelSelector>(blockNum, stream, params);
            break;
        case MatmulPermuteKernelPreset::AswtWindow:
            Run<WindowKernelSelector>(blockNum, stream, params);
            break;
    }
}
