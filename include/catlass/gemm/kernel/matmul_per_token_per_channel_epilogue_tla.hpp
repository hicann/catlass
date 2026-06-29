/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_MATMUL_PER_TOKEN_PER_CHANNEL_EPILOGUE_TLA_HPP
#define CATLASS_GEMM_KERNEL_MATMUL_PER_TOKEN_PER_CHANNEL_EPILOGUE_TLA_HPP

#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Kernel {

/// Matmul kernel with per-token and per-channel epilogue quantization.
///
/// Compute:  out = (A @ B) * perTokenScale * perChannelScale
///
/// Architecture (Epilogue-based quantization):
///   AIC (Cube Core):
///     1. Load A from GM to L1 (or receive from L1 if prologue enabled)
///     2. Load B from GM to L1 (or receive from L1 if prologue enabled)
///     3. Perform matmul: C = A @ B
///     4. Write C to GM workspace (intermediate result)
///
///   AIV (Vector Core):
///     1. Wait for AIC to finish writing C
///     2. Load C from GM workspace to UB
///     3. Load perTokenScale and perChannelScale from GM to UB
///     4. Apply scales: D = C * perTokenScale * perChannelScale
///     5. Write D to GM output
///
/// @tparam BlockMmad_       Block-level MMAD (AIC core)
/// @tparam BlockEpilogue_   Block-level Epilogue for quantization (AIV core)
/// @tparam BlockScheduler_  Block scheduling strategy
/// @tparam WORKSPACE_STAGES Number of workspace stages for double/triple buffering
template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, uint32_t WORKSPACE_STAGES_ = 2>
class MatmulPerTokenPerChannelEpilogueTla {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using BlockEpilogue = BlockEpilogue_;
    using ElementPerTokenScale = typename BlockEpilogue::ElementPerTokenScale;
    using LayoutPerTokenScale = typename BlockEpilogue::LayoutPerTokenScale;
    using ElementPerChannelScale = typename BlockEpilogue::ElementScale;
    using LayoutPerChannelScale = typename BlockEpilogue::LayoutScale;
    // For compatibility with standard interface
    using ElementScale = ElementPerChannelScale;
    using LayoutScale = LayoutPerChannelScale;
    using ElementD = typename BlockEpilogue::ElementD;
    using LayoutD = typename BlockEpilogue::LayoutD;
    using EpilogueParams = typename BlockEpilogue::Params;

    using BlockScheduler = BlockScheduler_;

    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});

    /// Cross-core synchronization for AIC -> AIV handoff
    friend struct AicWaitFunc;
    friend struct AicSetFunc;

    struct AicWaitFunc {
        using GemmKernel = MatmulPerTokenPerChannelEpilogueTla<BlockMmad, BlockEpilogue, BlockScheduler, WORKSPACE_STAGES>;

        CATLASS_DEVICE
        AicWaitFunc() = default;

        CATLASS_DEVICE
        void operator()() const
        {
            Arch::CrossCoreWaitFlag(ptr->flagAivFinishComputeList[stageId]);
        }

        GemmKernel *ptr{nullptr};
        uint32_t stageId;
    };

    struct AicSetFunc {
        using GemmKernel = MatmulPerTokenPerChannelEpilogueTla<BlockMmad, BlockEpilogue, BlockScheduler, WORKSPACE_STAGES>;

        CATLASS_DEVICE
        AicSetFunc() = default;

        CATLASS_DEVICE
        void operator()() const
        {
            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(ptr->flagAicFinishStoreList[stageId]);
        }

        GemmKernel *ptr{nullptr};
        uint32_t stageId;
    };

    /// Parameters structure
    struct Params {
        GemmCoord problemShape;
        __gm__ ElementA *ptrA;
        LayoutA layoutA;
        __gm__ ElementB *ptrB;
        LayoutB layoutB;
        __gm__ ElementPerTokenScale *ptrPerTokenScale;
        LayoutPerTokenScale layoutPerTokenScale;
        __gm__ ElementPerChannelScale *ptrPerChannelScale;
        LayoutPerChannelScale layoutPerChannelScale;
        __gm__ ElementD *ptrD;
        LayoutD layoutD;
        LayoutC layoutC;  // Layout for intermediate C (workspace)
        GM_ADDR ptrWorkspace;  // Intermediate C storage in GM

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord problemShape_,
            GM_ADDR ptrA_, LayoutA layoutA_,
            GM_ADDR ptrB_, LayoutB layoutB_,
            GM_ADDR ptrPerTokenScale_, LayoutPerTokenScale layoutPerTokenScale_,
            GM_ADDR ptrPerChannelScale_, LayoutPerChannelScale layoutPerChannelScale_,
            GM_ADDR ptrD_, LayoutD layoutD_,
            LayoutC layoutC_,
            GM_ADDR ptrWorkspace_
        ) : problemShape(problemShape_),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrPerTokenScale(reinterpret_cast<__gm__ ElementPerTokenScale *>(ptrPerTokenScale_)),
            layoutPerTokenScale(layoutPerTokenScale_),
            ptrPerChannelScale(reinterpret_cast<__gm__ ElementPerChannelScale *>(ptrPerChannelScale_)),
            layoutPerChannelScale(layoutPerChannelScale_),
            ptrD(reinterpret_cast<__gm__ ElementD *>(ptrD_)), layoutD(layoutD_),
            layoutC(layoutC_),
            ptrWorkspace(ptrWorkspace_) {}
    };

    /// Arguments structure (user-facing)
    struct Arguments {
        GemmCoord problemShape;
        GM_ADDR deviceA;
        LayoutA layoutA;
        GM_ADDR deviceB;
        LayoutB layoutB;
        GM_ADDR devicePerTokenScale;
        LayoutPerTokenScale layoutPerTokenScale;
        GM_ADDR devicePerChannelScale;
        LayoutPerChannelScale layoutPerChannelScale;
        GM_ADDR deviceD;
        LayoutD layoutD;
        uint32_t aicCoreNum;
    };

    CATLASS_DEVICE
    MatmulPerTokenPerChannelEpilogueTla()
    {
        Arch::FlagID flagId = 0;
        for (uint32_t stageId = 0; stageId < WORKSPACE_STAGES; ++stageId) {
            flagAicFinishStoreList[stageId] = Arch::CrossCoreFlag(flagId++);
            flagAivFinishComputeList[stageId] = Arch::CrossCoreFlag(flagId++);
            aicWaitFuncList[stageId] = {this, stageId};
            aicSetFuncList[stageId] = {this, stageId};
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    /// AIC core: performs matrix multiplication and writes result to GM workspace
    /// Uses TLA tensor for all operands
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler blockScheduler;
        blockScheduler.Update(params.problemShape, Catlass::MatrixCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = blockScheduler.GetCoreLoops();

        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(params.ptrB);

        AscendC::LocalTensor<ElementC> ubC = AscendC::LocalTensor<ElementC>(AscendC::TPosition::VECCALC, 0, L1_TILE_M * L1_TILE_N);
        auto layoutTagC = layout::RowMajor{L1_TILE_M , L1_TILE_N};
        auto layoutC = tla::MakeLayoutFromTag(layoutTagC);

        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, params.layoutB, Arch::PositionGM{});

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t stageId = 0;
        uint32_t stageUsed = 0;
        auto tensorC = tla::MakeTensor(ubC[BlockEpilogue::TileShape::COUNT * stageId], layoutC, Arch::PositionUB{});

        for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
            tensorC = tla::MakeTensor(ubC[BlockEpilogue::TileShape::COUNT * stageId], layoutC, Arch::PositionUB{});
            auto blockCoord = blockScheduler.GetBlockCoord(loopIdx);
            auto actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

            Callback callbackBeforeFixpipe{};
            if (stageUsed == WORKSPACE_STAGES) {
                callbackBeforeFixpipe = MakeCallback(&aicWaitFuncList[stageId]);
            } else {
                ++stageUsed;
            }
            Callback callbackAfterFixpipe = MakeCallback(&aicSetFuncList[stageId]);

            auto tensorBlockA = GetTile(tensorA,
                                        tla::MakeCoord(blockCoord.m() * L1_TILE_M, blockCoord.k() * L1_TILE_K),
                                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
            auto tensorBlockB = GetTile(tensorB,
                                        tla::MakeCoord(blockCoord.k() * L1_TILE_K, blockCoord.n() * L1_TILE_N),
                                        tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
            auto tensorBlockC = GetTile(tensorC,
                                        tla::MakeCoord(0, 0),
                                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

            if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                blockMmad(
                    tensorBlockA, tensorBlockB, tensorBlockC,
                    actualBlockShape, {}, callbackBeforeFixpipe, callbackAfterFixpipe
                );
            } else {
                callbackBeforeFixpipe();
                blockMmad(
                    tensorBlockA, tensorBlockB, tensorBlockC,
                    actualBlockShape
                );
                callbackAfterFixpipe();
            }
            stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
        }


        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.template SynchronizeBlock<decltype(tensorC)>();
        }

        while (stageUsed > 0) {
            uint32_t aivComputeStageId = (stageId >= stageUsed) ?
                (stageId - stageUsed) : (stageId + WORKSPACE_STAGES - stageUsed);
            Arch::CrossCoreWaitFlag(flagAivFinishComputeList[aivComputeStageId]);
            --stageUsed;
        }
        AscendC::PipeBarrier<PIPE_ALL>();

    }

    /// AIV core: performs per-token per-channel quantization on matmul result
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
          AscendC::ICachePreLoad(1);
        BlockScheduler blockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N));
        BlockEpilogue blockEpilogue(resource);

        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrWorkspace));
        auto layoutC = layout::RowMajor{L1_TILE_M * coreNum * WORKSPACE_STAGES, L1_TILE_N};

        EpilogueParams epilogueParams{
            params.ptrPerChannelScale, params.layoutPerChannelScale,
            params.ptrPerTokenScale, params.layoutPerTokenScale,
            params.ptrD, params.layoutD
        };
        blockEpilogue.UpdateParams(epilogueParams);

        uint32_t coreLoops = blockScheduler.GetCoreLoops();
        GemmCoord blockShapeMNK = MakeCoord(L1_TILE_M, L1_TILE_N, L1_TILE_K);

        uint32_t stageId = 0;
        for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
            GemmCoord blockCoordMNK = blockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShapeMNK = blockScheduler.GetActualBlockShape(blockCoordMNK);

            MatrixCoord offsetC{(stageId * coreNum + coreIdx) * L1_TILE_M, 0};
            int64_t gmOffsetC = layoutC.GetOffset(offsetC);
            auto gmBlockC = gmC[gmOffsetC];
            auto layoutBlockC = layoutC.GetTileLayout(actualBlockShapeMNK.GetCoordMN());

            // Arch::CrossCoreWaitFlag(flagAicFinishStoreList[stageId]);
            blockEpilogue(blockShapeMNK, blockCoordMNK, actualBlockShapeMNK, gmBlockC, layoutBlockC, stageId, flagAicFinishStoreList[stageId], flagAivFinishComputeList[stageId]);
            // Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishComputeList[stageId]);

            stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    /// Check if arguments can be implemented
    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    /// Get workspace size (for intermediate C storage)
    static size_t GetWorkspaceSize(Arguments const &args)
    {
        size_t lenWorkspace = static_cast<size_t>(L1_TILE_M) * L1_TILE_N *
            args.aicCoreNum * WORKSPACE_STAGES;
        size_t sizeWorkspace = lenWorkspace * sizeof(ElementC);
        return sizeWorkspace;
    }

    /// Convert arguments to parameters
    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        // Create layoutC for intermediate workspace
        auto layoutTagC = layout::RowMajor{args.problemShape.m(), args.problemShape.n()};
        auto layoutC = tla::MakeLayoutFromTag(layoutTagC);  
        Params params{
            args.problemShape,
            args.deviceA, args.layoutA,
            args.deviceB, args.layoutB,
            args.devicePerTokenScale, args.layoutPerTokenScale,
            args.devicePerChannelScale, args.layoutPerChannelScale,
            args.deviceD, args.layoutD,
            layoutC,
            workspace
        };
        return params;
    }

private:
    Arch::CrossCoreFlag flagAicFinishStoreList[WORKSPACE_STAGES];
    Arch::CrossCoreFlag flagAivFinishComputeList[WORKSPACE_STAGES];

    AicWaitFunc aicWaitFuncList[WORKSPACE_STAGES];
    AicSetFunc aicSetFuncList[WORKSPACE_STAGES];
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_MATMUL_PER_TOKEN_PER_CHANNEL_EPILOGUE_TLA_HPP
