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

#include <algorithm>
#include <cstddef>
using std::size_t;

#include <kernel_operator.h>

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/mx_matmul_pertoken_perchannel_tla.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"

#include "catlass_kernel.h"
#include "common/kernel_runner.h"

#ifndef CATLASS_JIT_ELEMENT_A
#define CATLASS_JIT_ELEMENT_A float4_e2m1x2_t
#endif
#ifndef CATLASS_JIT_ELEMENT_B
#define CATLASS_JIT_ELEMENT_B float4_e2m1x2_t
#endif
#ifndef CATLASS_JIT_ELEMENT_MX_SCALE
#define CATLASS_JIT_ELEMENT_MX_SCALE float8_e8m0_t
#endif
#ifndef CATLASS_JIT_ELEMENT_SCALE
#define CATLASS_JIT_ELEMENT_SCALE float8_e4m3_t
#endif
#ifndef CATLASS_JIT_ELEMENT_PER_TOKEN_SCALE
#define CATLASS_JIT_ELEMENT_PER_TOKEN_SCALE float8_e4m3_t
#endif

using ElementA = CATLASS_JIT_ELEMENT_A;
using ElementB = CATLASS_JIT_ELEMENT_B;
#ifdef CATLASS_JIT_ELEMENT_C
using ElementC = CATLASS_JIT_ELEMENT_C;
#else
using ElementC = float;
#endif
using ElementMxScale = CATLASS_JIT_ELEMENT_MX_SCALE;
using ElementScale = CATLASS_JIT_ELEMENT_SCALE;
using ElementPerTokenScale = CATLASS_JIT_ELEMENT_PER_TOKEN_SCALE;
#ifdef CATLASS_JIT_ELEMENT_D
using ElementD = CATLASS_JIT_ELEMENT_D;
#else
using ElementD = float;
#endif

using LayoutTagA = Catlass::layout::RowMajor;
using LayoutTagB = Catlass::layout::RowMajor;
using LayoutTagC = Catlass::layout::RowMajor;
using LayoutTagD = Catlass::layout::RowMajor;
using LayoutTagScale = Catlass::layout::VectorLayout;
using LayoutTagPerTokenScale = Catlass::layout::VectorLayout;

using ArchTag = Catlass::Arch::Ascend950;
constexpr bool enableUnitFlag = true;
using DispatchPolicy = Catlass::Gemm::MmadMx<ArchTag, enableUnitFlag>;
using L1TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<512>>;
using L0TileShape = tla::Shape<tla::Int<256>, tla::Int<256>, tla::Int<256>>;
constexpr uint32_t workspaceStages = 2;
constexpr uint32_t ubStages = 2;

extern "C" void run(uint32_t blockNum, aclrtStream stream, const CatlassKernel::MatmulParams* params)
{
    uint32_t m = params->m;
    uint32_t n = params->n;
    uint32_t k = params->k;
    uint32_t mxScaleK = CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(k);

    uint8_t* deviceA = params->inputAddr[0];
    uint8_t* deviceB = params->inputAddr[1];
    uint8_t* deviceMxScaleA = params->inputAddr[2];
    uint8_t* deviceMxScaleB = params->inputAddr[3];
    uint8_t* deviceScale = params->inputAddr[4];
    uint8_t* devicePerTokenScale = params->inputAddr[5];
    uint8_t* deviceD = params->outputAddr[0];

    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagA, false>(m, mxScaleK);
    auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagB, true>(mxScaleK, n);

    LayoutTagD layoutD{m, n};
    LayoutTagScale layoutScale{n};
    LayoutTagPerTokenScale layoutPerTokenScale{m};

    using TileCopyMmad = Catlass::Gemm::Tile::PackedMxTileCopyTla<
        ArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementMxScale, decltype(layoutMxScaleA), ElementMxScale,
        decltype(layoutMxScaleB), ElementC, LayoutTagC, void>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape, L0TileShape, ElementA, ElementB, ElementC, void, TileCopyMmad>;
    using CType = Catlass::Gemm::GemmType<ElementC, Catlass::layout::RowMajor>;

    using EpilogueDispatchPolicy = Catlass::Epilogue::EpilogueAscend950PerTokenPerChannelQuant<ubStages>;
    using ScaleType = Catlass::Gemm::GemmType<ElementScale, Catlass::layout::VectorLayout>;
    using PerTokenScaleType = Catlass::Gemm::GemmType<ElementPerTokenScale, Catlass::layout::VectorLayout>;
    using DType = Catlass::Gemm::GemmType<ElementD, Catlass::layout::RowMajor>;
    using RowBroadcastMulType = Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>;
    using BroadcastOneBlkType = Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>;
    using OneBlkColumnBroadcastMulType = Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>;
    using EpilogueTileShape = Catlass::MatrixShape<32, 256>;
    using TileRowBroadcastMul =
        Catlass::Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;
    using TileBroadcastOneBlk =
        Catlass::Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType, EpilogueTileShape::ROW>;
    using TileOneBlkColumnBroadcastMul =
        Catlass::Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag, OneBlkColumnBroadcastMulType, EpilogueTileShape>;
    using TileCopyEpilogue =
        Catlass::Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, DType>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueHorizontalTileSwizzle;
    using BlockEpilogue = Catlass::Epilogue::Block::BlockEpilogue<
        EpilogueDispatchPolicy, CType, ScaleType, PerTokenScaleType, DType, TileRowBroadcastMul, TileBroadcastOneBlk,
        TileOneBlkColumnBroadcastMul, TileCopyEpilogue, TileScheduler>;
    using BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    using MatmulKernel = Catlass::Gemm::Kernel::MxMatmulPerTokenPerChannelTla<
        BlockMmad, BlockEpilogue, BlockScheduler, workspaceStages>;

    Catlass::GemmCoord problemShape{m, n, k};
    typename MatmulKernel::Arguments arguments{
        problemShape, blockNum,
        deviceA, layoutA,
        deviceB, layoutB,
        deviceMxScaleA, layoutMxScaleA,
        deviceMxScaleB, layoutMxScaleB,
        deviceScale, layoutScale,
        devicePerTokenScale, layoutPerTokenScale,
        deviceD, layoutD
    };

    uint64_t taskNum64 = static_cast<uint64_t>(CeilDiv(m, tla::get<0>(L1TileShape{}))) *
                         static_cast<uint64_t>(CeilDiv(n, tla::get<1>(L1TileShape{})));
    uint32_t taskNum = static_cast<uint32_t>(std::min(taskNum64, static_cast<uint64_t>(UINT32_MAX)));
    uint32_t aicCoreUsed = std::min(blockNum, taskNum);

    Catlass::RunKernel<MatmulKernel>(arguments, stream, aicCoreUsed);
}

