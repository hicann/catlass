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
#ifndef HSTU_INFER_LAUNCHER_HPP
#define HSTU_INFER_LAUNCHER_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_silu.hpp"

#include "../kernel/hstu_infer.hpp"

#include "golden.hpp"
#include "helper.hpp"

using namespace Catlass;
using namespace tla;

template <class Kernel>
CATLASS_GLOBAL __mix__(1, 2) void HstuInferKernel(typename Kernel::Params params)
{
    Kernel kernel;
    kernel(params);
}

template <
    class DType, class LayoutTagQO_ = Catlass::layout::NTD, class LayoutTagKV_ = Catlass::layout::NTD,
    bool ENABLE_PAGED_KV_CACHE_ = false>
void RunHSTUKernel(
    GM_ADDR qDevice, GM_ADDR kDevice, GM_ADDR vDevice, GM_ADDR oDevice, GM_ADDR qSeqDevice, GM_ADDR kvSeqDevice,
    GM_ADDR blockTableDevice, uint32_t batch, uint32_t numHeads, uint32_t embeddingSize, uint32_t kvHeads,
    uint32_t maxKvSeqlen, uint32_t numPagedBlocks, uint32_t pagedBlockSize, float siluScale, uint32_t maskType,
    aclrtStream stream, const uint32_t aicCoreNum, GM_ADDR* workspaceDevice)
{
    static constexpr uint32_t headDimCfg = 256;

    const uint32_t qkL1TileM = 96;
    const uint32_t qkL1TileN = 256;
    const uint32_t qkL1TileK = headDimCfg;
    const uint32_t qkL0TileM = qkL1TileM;
    const uint32_t qkL0TileN = qkL1TileN;
    const uint32_t qkL0TileK = 64;

    const uint32_t pvL1TileM = qkL1TileM;
    const uint32_t pvL1TileN = headDimCfg;
    const uint32_t pvL1TileK = qkL1TileN;
    const uint32_t pvL0TileM = pvL1TileM;
    const uint32_t pvL0TileN = pvL1TileN;
    const uint32_t pvL0TileK = 64;

    using ArchTag = Arch::Ascend950;
    using ElementQ = DType;
    using LayoutTagQ = layout::RowMajor;

    using ElementK = DType;
    using LayoutTagK = layout::ColumnMajor;

    using ElementV = DType;
    using LayoutTagV = layout::RowMajor;

    using ElementS = float;
    using LayoutTagS = layout::RowMajor;
    using SType = Gemm::GemmType<ElementS, LayoutTagS, AscendC::TPosition::VECCALC>;

    using ElementP = DType;
    using LayoutTagP = layout::zN;
    using PType = Gemm::GemmType<ElementP, LayoutTagP, AscendC::TPosition::A1>;

    using ElementO = DType;
    using LayoutTagO = layout::RowMajor;
    using OType = Gemm::GemmType<ElementO, LayoutTagO, AscendC::TPosition::GM>;

    using DispatchPolicyQK = Gemm::MmadHstuQK<ArchTag, ENABLE_PAGED_KV_CACHE_, false>;
    using L1TileShapeQK = Shape<Int<qkL1TileM>, Int<qkL1TileN>, Int<qkL1TileK>>;
    using L0TileShapeQK = Shape<Int<qkL0TileM>, Int<qkL0TileN>, Int<qkL0TileK>>;
    using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementQ, LayoutTagQ, ElementK, LayoutTagK, ElementS, LayoutTagS, void,
        Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<
        DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementQ, ElementK, ElementS, void, TileCopyQK>;

    // Epilogue Block模块，实现HSTU Infer中silu
    using DispatchPolicySiluScale = Epilogue::EpilogueSilu;
    using SiluTileShape = Shape<Int<qkL1TileM>, Int<qkL1TileN>>;
    using TileElemWiseSiluScale = Epilogue::Tile::TileElemWiseSiluScaleRegBase<ArchTag, ElementP, ElementS>;
    using TileCopySiluScale = Epilogue::Tile::TileCopy<ArchTag, PType, SType>;
    using EpilogueSiluScale = Epilogue::Block::BlockEpilogue<
        DispatchPolicySiluScale, PType, SType, SiluTileShape, TileElemWiseSiluScale, TileCopySiluScale>;

    // L1TileShape::N must be v embdding
    using DispatchPolicyPV = Gemm::MmadHstuPV<ArchTag, ENABLE_PAGED_KV_CACHE_, false>;
    using L1TileShapePV = Shape<Int<pvL1TileM>, Int<pvL1TileN>, Int<pvL1TileK>>;
    using L0TileShapePV = Shape<Int<pvL0TileM>, Int<pvL0TileN>, Int<pvL0TileK>>;
    using TileCopyPV =
        Gemm::Tile::PackedTileCopyTla<ArchTag, ElementP, LayoutTagP, ElementV, LayoutTagV, ElementO, LayoutTagO, void>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<
        DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV, ElementO, void, TileCopyPV>;

    using EpilogueCast = void;

    // Kernel level
    using HstuKernel =
        Gemm::Kernel::HstuInfer<BlockMmadQK, BlockMmadPV, EpilogueSiluScale, EpilogueCast, LayoutTagQO_, LayoutTagKV_>;

    typename HstuKernel::Arguments arguments{
        batch,    numHeads, embeddingSize, kvHeads, maxKvSeqlen, numPagedBlocks, pagedBlockSize, siluScale,
        maskType, qDevice,  kDevice,       vDevice, oDevice,     qSeqDevice,     kvSeqDevice,    blockTableDevice};
    if (!HstuKernel::CanImplement(arguments)) {
        std::cerr << "HstuKernel can not implement the arguments." << std::endl;
        return;
    }

    uint8_t* deviceWorkspace{nullptr};
    size_t sizeWorkspace = HstuKernel::GetWorkspaceSize(arguments);
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
        *workspaceDevice = deviceWorkspace;
    }

    auto params = HstuKernel::ToUnderlyingArguments(arguments, deviceWorkspace);

    HstuInferKernel<HstuKernel><<<aicCoreNum, nullptr, stream>>>(params);
}

#endif
