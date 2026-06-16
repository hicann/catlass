/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_scheduler_aswt.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/grouped_matmul_slice_m_aswt_tla.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"

#include "../common/common.h"
#include "catlass_kernel.h"
#include "common/kernel_runner.h"
#include "common/tile_shape_scaler_tla.h"

#ifndef CATLASS_JIT_FIXPIPE_QUANT_MODE
#define CATLASS_JIT_FIXPIPE_QUANT_MODE 0
#endif

#if CATLASS_JIT_FIXPIPE_QUANT_MODE == 0
constexpr Catlass::Gemm::Tile::ScaleGranularity QuantMode = Catlass::Gemm::Tile::ScaleGranularity::PER_CHANNEL;
#elif CATLASS_JIT_FIXPIPE_QUANT_MODE == 1
constexpr Catlass::Gemm::Tile::ScaleGranularity QuantMode = Catlass::Gemm::Tile::ScaleGranularity::PER_TENSOR;
#else
#error "Unsupported CATLASS_JIT_FIXPIPE_QUANT_MODE"
#endif

using namespace Catlass;
using namespace tla;

using ElementA = int8_t;
using ElementB = int8_t;
using ElementC = half;

using LayoutTagA = layout::RowMajor;
using LayoutTagB = layout::RowMajor;
using LayoutTagC = layout::RowMajor;

using ArchTag = Arch::Ascend950;
constexpr bool enableUnitFlag = true;
constexpr bool useHF32 = false;
using DispatchPolicy = Gemm::MmadDequant<ArchTag, enableUnitFlag, useHF32>;
using BaseL1 = tuple<C<256>, C<256>, C<256>>;
using BaseL0 = tuple<C<256>, C<256>, C<64>>;
using L1TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, int8_t, BaseL1>::type;
using L0TileShape = typename CatlassKernel::TileShapeScalerTLA<ElementA, int8_t, BaseL0>::type;

using TileCopy =
    Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC, void, false, QuantMode>;
using BlockMmad = Gemm::Block::BlockMmadTla<
    DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopy>;
using BlockEpilogue = void;

constexpr bool isGmm = true;
using BlockScheduler = typename Gemm::Block::BlockSchedulerAswt<L1TileShape, L0TileShape, isGmm>;
using MatmulKernel = Gemm::Kernel::GroupedMatmulSliceMAswtTla<BlockMmad, BlockEpilogue, BlockScheduler, int64_t>;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    const auto& fp = *static_cast<const CatlassKernel::MatmulFixPipeParams*>(params);

    uint32_t m = fp.m;
    uint32_t n = fp.n;
    uint32_t k = fp.k;
    uint32_t groupCount = fp.batch;

    auto layoutA = MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutC = MakeLayout<ElementC, LayoutTagC>(m, n);

    float scalePerTensor = fp.perTensorScale;

    typename MatmulKernel::Arguments arguments{
        GemmCoord{m, n, k},
        groupCount,
        fp.inputAddr[2],
        fp.inputAddr[0], layoutA,
        fp.inputAddr[1], layoutB,
        fp.outputAddr[0], layoutC,
        scalePerTensor,
        fp.inputAddr[3],
    };

    Catlass::RunKernel<MatmulKernel>(arguments, stream, blockNum);
}
