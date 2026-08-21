/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @brief matmul implementation for single q&k^t base tile.
 * This implementation is designed for the following senario:
 * A full q base tile is loaded to L1 from GM at the very beginning,
 * and it remains persistent until each k base tile is dealt
 * A full q*k^t base tile is loaded to UB from l0C, no workspace transit
 */
#ifndef GEMM_BLOCK_BLOCK_MMAD_RAIN_FUSION_ATTENTION_QK_HPP
#define GEMM_BLOCK_BLOCK_MMAD_RAIN_FUSION_ATTENTION_QK_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"

using namespace Catlass::Arch;

namespace Catlass::Gemm::Block {

template <
    class ArchTag_, class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_, class ElementC_,
    class ElementBias_, class TileCopy_, class TileMmad_>
struct BlockMmadTla<
    MmadAtlasA5RainQK<ArchTag_>, L1TileShape_, L0TileShape_, ElementA_, ElementB_, ElementC_, ElementBias_, TileCopy_,
    TileMmad_> {
public:
    using ArchTag = ArchTag_;
    using DispatchPolicy = MmadAtlasA5RainQK<ArchTag>;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using TileMmad = TileMmad_;
    using TileCopy = TileCopy_;

    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;
    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    static constexpr uint32_t L1A_BUF_NUM = DispatchPolicy::L1A_STAGES; // Q, 1-buf
    static constexpr uint32_t L1B_BUF_NUM = DispatchPolicy::L1B_STAGES; // K, 2-buf
    static constexpr uint32_t L0_STAGES = DispatchPolicy::L0_STAGES;
    // L1 tile shape
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{}); // s1
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{}); // s2
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{}); // d
    // L0 tile shape
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});
    // L1 buffer size
    static constexpr uint32_t L1A_BUF_SIZE = L1_TILE_M * L1_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L1B_BUF_SIZE = L1_TILE_N * L1_TILE_K * sizeof(ElementB);
    static constexpr uint32_t BLOCK_L1_SIZE = L1A_BUF_SIZE * L1A_BUF_NUM + L1B_BUF_SIZE * L1B_BUF_NUM;

    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / L0_STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / L0_STAGES;
    static constexpr uint32_t L0C_HALF_BUF_SIZE = ArchTag::L0C_SIZE / 2;
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_HALF_BUF_SIZE / L0_STAGES;

    static constexpr uint32_t V0_V1_FLAG_ID_OFFSET = 16; // 核间同步mode4，AIC侧需要两个flagId分别对应两个AIV

    // Check L1/L0TileShape
    static_assert(
        L1_TILE_M == L0_TILE_M && L1_TILE_K == L0_TILE_K,
        "The situation where the basic blocks of L1 and L0 differ on the m, k axes is not supported yet");

    __aicore__ inline BlockMmadTla(Arch::Resource<ArchTag>& resource)
    {
        for (uint32_t i = 0; i < L1A_BUF_NUM; i++) {
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(L1A_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < L1B_BUF_NUM; i++) {
            l1BTensor[i] =
                resource.l1Buf.template GetBufferByByte<ElementB>(L1A_BUF_SIZE * L1A_BUF_NUM + L1B_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < L0_STAGES; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i);
        }
    }

    __aicore__ inline ~BlockMmadTla()
    {}

    template <class TensorA>
    __aicore__ inline void loadQGM(TensorA& gATensor, uint32_t rowNum, uint32_t embed)
    {
        using CopyGmToL1A = typename TileCopy_::template CopyGmToL1A<TensorA>;
        CopyGmToL1A copyGmToL1A;

        auto l1ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL1A>(rowNum, embed);
        auto l1ATensorTla = tla::MakeTensor(l1ATensor[0], l1ALayoutTla, Arch::PositionL1{});
        auto l1ATensorTlaTile = GetTile(l1ATensorTla, tla::MakeCoord(0, 0), tla::MakeShape(rowNum, embed));
        auto gATensorTlaTile = GetTile(gATensor, tla::MakeCoord(0, 0), tla::MakeShape(rowNum, embed));

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0); // L1 event_id: 0
        copyGmToL1A(l1ATensorTlaTile, gATensorTlaTile);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline void SetCrossCoreSync(Arch::CrossCoreFlag& crossCoreFlag)
    {
        // in mode 4, AIC set for 2 AIVs seperately
        if constexpr (MODE == 4) {
            uint16_t flagIdV0 = crossCoreFlag.id;
            uint16_t flagIdV1 = flagIdV0 + V0_V1_FLAG_ID_OFFSET;
            Arch::CrossCoreFlag crossCoreFlagV1(flagIdV1);
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlagV1);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline void WaitCrossCoreSync(Arch::CrossCoreFlag& crossCoreFlag)
    {
        // in mode 4, AIC wait for 2 AIVs seperately
        if constexpr (MODE == 4) {
            uint16_t flagIdV0 = crossCoreFlag.id;
            uint16_t flagIdV1 = flagIdV0 + V0_V1_FLAG_ID_OFFSET;
            Arch::CrossCoreFlag crossCoreFlagV1(flagIdV1);
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlagV1);
        }
    }

    __aicore__ inline uint32_t GetCurLoopCounter(uint32_t outerLoopItr, uint32_t loopNum, uint32_t subLoopItr)
    {
        return outerLoopItr * loopNum + subLoopItr;
    }

    template <class TensorB, class TensorC>
    __aicore__ inline void operator()(
        TensorB& gBTensor, TensorC& ubCTensor, AscendC::GlobalTensor<int64_t> gSelectIdx, GemmCoord actualOriShape,
        uint32_t kvSTileIdx, uint32_t kvSeqlen, uint32_t kvSBaseTile, uint32_t blockShapeY, uint32_t selectNum,
        uint32_t& kvYBlockNum, uint64_t prefixSumL0AStages, uint64_t prefixSumL0BStages,
        Arch::CrossCoreFlag mm1ToSmFlag)
    {
        using CopyL0CToDst = typename TileCopy_::template CopyL0CToDst<TensorC>;
        CopyL0CToDst copyL0CToDstSub0;
        CopyL0CToDst copyL0CToDstSub1;

        uint32_t rowNum = actualOriShape[0];          // Block层actualM, s1
        uint32_t curBaseTileSize = actualOriShape[1]; // Block层actualN, s2
        uint32_t embed = actualOriShape[2];           // Block层actualK, d

        auto l1ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL1A>(rowNum, embed);
        auto l1ATensorTla = tla::MakeTensor(l1ATensor[0], l1ALayoutTla, Arch::PositionL1{}); // L1A: Query

        uint32_t nL1Loop = CeilDiv<L1_TILE_N>(curBaseTileSize);
        uint32_t mL0Loop = CeilDiv<L0_TILE_M>(rowNum);
        uint32_t kL0Loop = CeilDiv<L0_TILE_K>(embed);

        for (uint32_t nL1Itr = 0; nL1Itr < nL1Loop; nL1Itr++) {
            uint32_t n1Actual = (nL1Itr == nL1Loop - 1) ? (curBaseTileSize - nL1Itr * L1_TILE_N) : L1_TILE_N;
            uint32_t nL1LoopCounter = GetCurLoopCounter(kvSTileIdx, nL1Loop, nL1Itr);
            uint32_t l1BBufId = nL1LoopCounter % L1B_BUF_NUM;
            uint32_t l1BEventId = l1BBufId + L1A_BUF_NUM; // L1 event_id: 1,2
            auto l1BLayoutTla = tla::MakeLayout<ElementB, LayoutTagL1B>(embed, n1Actual);
            auto l1BTensorTla = tla::MakeTensor(l1BTensor[l1BBufId], l1BLayoutTla, Arch::PositionL1{}); // L1B: Key

            // load K full base tile to L1
            uint32_t gatheredKvStartOffset = kvSTileIdx * kvSBaseTile + nL1Itr * L1_TILE_N;
            uint32_t gatheredYBlockIdx = gatheredKvStartOffset / blockShapeY;
            uint32_t oriYBlockIdx = gSelectIdx.GetValue(gatheredYBlockIdx);
            uint32_t processedSize = 0; // 逐稀疏block搬移填充基本块过程中，已处理的累积序列长度
            uint32_t yBlockInnerOffset = gatheredKvStartOffset % blockShapeY;
            uint32_t oriKvYOffset = oriYBlockIdx * blockShapeY + yBlockInnerOffset; // origin y offset

            using CopyGmToL1B = typename TileCopy_::template CopyGmToL1B<TensorB>;
            CopyGmToL1B copyGmToL1B;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventId);

            while (processedSize < n1Actual && gatheredYBlockIdx < selectNum && oriYBlockIdx < kvYBlockNum &&
                   oriKvYOffset < kvSeqlen) {
                uint32_t yBlockActualSize = (gatheredYBlockIdx == selectNum - 1 && oriYBlockIdx == kvYBlockNum - 1 &&
                                             kvSeqlen % blockShapeY != 0) ?
                                                (kvSeqlen - blockShapeY * oriYBlockIdx) :
                                                blockShapeY;
                uint32_t remainingInYBlock = yBlockActualSize - yBlockInnerOffset;
                uint32_t remainingNInL1B = n1Actual - processedSize;

                uint32_t curCopyYSize = min(remainingNInL1B, remainingInYBlock);
                if (curCopyYSize == 0) {
                    break;
                }

                auto l1BTensorTlaTile = GetTile(
                    l1BTensorTla, tla::MakeCoord(0, processedSize), tla::MakeShape(embed, curCopyYSize)); // L1B Tile
                auto gBTensorTlaTile =
                    GetTile(gBTensor, tla::MakeCoord(0, oriKvYOffset), tla::MakeShape(embed, curCopyYSize)); // gmB Tile
                copyGmToL1B(l1BTensorTlaTile, gBTensorTlaTile);

                // 为下一次循环刷新循环变量
                processedSize += curCopyYSize;
                yBlockInnerOffset += curCopyYSize;
                oriKvYOffset += curCopyYSize;
                if (yBlockInnerOffset >= yBlockActualSize) {
                    gatheredYBlockIdx++;
                    if (gatheredYBlockIdx >= selectNum) {
                        break;
                    }
                    yBlockInnerOffset = 0;
                    oriYBlockIdx = gSelectIdx.GetValue(gatheredYBlockIdx);
                    oriKvYOffset = oriYBlockIdx * blockShapeY;
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);

            uint32_t nL0Loop = CeilDiv<L0_TILE_N>(n1Actual);
            for (uint32_t nL0Itr = 0; nL0Itr < nL0Loop; nL0Itr++) {
                uint32_t n0Actual = (nL0Itr == nL0Loop - 1) ? (n1Actual - nL0Itr * L0_TILE_N) : L0_TILE_N;
                uint32_t nL0LoopCounter = GetCurLoopCounter(nL1Itr, nL0Loop, nL0Itr);
                // l0C nbuffer chunked only in n loop
                uint32_t l0CLoopCounter = GetCurLoopCounter(nL1LoopCounter, nL0Loop, nL0Itr);
                uint32_t l0CBufId = l0CLoopCounter % L0_STAGES;
                uint32_t l0CEventId = l0CBufId; // L0C event_id: 0,1
                auto l0CLayoutTla = tla::MakeLayoutL0C(rowNum, n0Actual);
                auto l0CTensorTla = tla::MakeTensor(l0CTensor[l0CBufId], l0CLayoutTla, Arch::PositionL0C{}); // L0C

                for (uint32_t mL0Itr = 0; mL0Itr < mL0Loop; mL0Itr++) {
                    uint32_t m0Actual = (mL0Itr == mL0Loop - 1) ? (rowNum - mL0Itr * L0_TILE_M) : L0_TILE_M;
                    // different m chunks will be concated in the same piece of l0C buffer
                    auto l0CTensorTlaTile = GetTile(
                        l0CTensorTla, tla::MakeCoord(mL0Itr * L0_TILE_M, 0),
                        tla::MakeShape(m0Actual, n0Actual)); // L0C Tile

                    for (uint32_t kL0Itr = 0; kL0Itr < kL0Loop; kL0Itr++) {
                        uint32_t k0Actual = (kL0Itr == kL0Loop - 1) ? (embed - kL0Itr * L0_TILE_K) : L0_TILE_K;
                        uint32_t l0ALoopCounter = prefixSumL0AStages + GetCurLoopCounter(mL0Itr, kL0Loop, kL0Itr);
                        uint32_t l0BLoopCounter =
                            prefixSumL0BStages + GetCurLoopCounter(nL0LoopCounter, kL0Loop, kL0Itr);
                        uint32_t l0ABufId = l0ALoopCounter % L0_STAGES;
                        uint32_t l0BBufId = l0BLoopCounter % L0_STAGES;
                        uint32_t l0AEventId = l0ABufId;             // L0A event_id: 0,1
                        uint32_t l0BEventId = l0BBufId + L0_STAGES; // L0B event_id: 2,3

                        auto l1BTensorTlaTile = GetTile(
                            l1BTensorTla, tla::MakeCoord(kL0Itr * L0_TILE_K, nL0Itr * L0_TILE_N),
                            tla::MakeShape(k0Actual, n0Actual));
                        auto l0BLayoutTla = tla::MakeLayout<ElementB, LayoutTagL0B>(k0Actual, n0Actual);
                        auto l0BTensorTla = tla::MakeTensor(l0BTensor[l0BBufId], l0BLayoutTla, Arch::PositionL0B{});
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                        copyL1ToL0B(l0BTensorTla, l1BTensorTlaTile);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                        if ((nL0Itr == nL0Loop - 1) && (mL0Itr == mL0Loop - 1) && (kL0Itr == kL0Loop - 1)) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventId);
                        }

                        auto l1ATensorTlaTile = GetTile(
                            l1ATensorTla, tla::MakeCoord(mL0Itr * L0_TILE_M, kL0Itr * L0_TILE_K),
                            tla::MakeShape(m0Actual, k0Actual));
                        auto l0ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL0A>(m0Actual, k0Actual);
                        auto l0ATensorTla = tla::MakeTensor(l0ATensor[l0ABufId], l0ALayoutTla, Arch::PositionL0A{});
                        bool l0AReuseAcrossN = ((mL0Loop == 1) && (kL0Loop <= L0_STAGES));
                        bool moveL0A = l0AReuseAcrossN && (nL1Itr == 0) && (nL0Itr == 0);
                        if (moveL0A) {
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                            copyL1ToL0A(l0ATensorTla, l1ATensorTlaTile);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                        }

                        bool initMmad = (kL0Itr == 0);
                        uint32_t l0TileMAligned = RoundUp(m0Actual, 16);
                        if (moveL0A) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                        }
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                        if (mL0Itr == 0 && kL0Itr == 0) {
                            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
                        }
                        tileMmad(
                            l0CTensorTlaTile, l0ATensorTla, l0BTensorTla, l0TileMAligned, n0Actual, k0Actual, initMmad);
                        if (l0AReuseAcrossN && (nL1Itr == nL1Loop - 1) && (nL0Itr == nL0Loop - 1)) {
                            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                    } // end for(kL0Loop)
                } // end for(mL0Loop)

                // fixpipe
                if (nL0Itr == 0) {
                    // reverse crossCoreSync, do fixpipe only after ubCTensor is fully released
                    WaitCrossCoreSync<4, PIPE_FIX>(mm1ToSmFlag);
                }
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
                // kernel传输ubCTensor的时候确保其shape的m,n是32B对齐（8个b32数据）
                // evenly split l0CTensorTla to 2 AIV:
                //     AIV0: valid rows [0, mFixPAligned8 / 2 - 1]
                //     AIV1: valid rows [mFixPAligned8 / 2, rowNum - 1]
                uint32_t mFixPAligned8 = RoundUp(rowNum, 8);
                uint32_t mPerSubCore = mFixPAligned8 / 2;
                uint32_t nFixPAligned16 = RoundUp(n0Actual, 16); // b16
                auto ubCTensorTlaTile = GetTile(
                    ubCTensor, tla::MakeCoord(0, nL0Itr * L0_TILE_N), tla::MakeShape(mPerSubCore, nFixPAligned16));
                auto l0CTensorTlaTileSub0 =
                    GetTile(l0CTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(mPerSubCore, n0Actual));
                auto l0CTensorTlaTileSub1 =
                    GetTile(l0CTensorTla, tla::MakeCoord(mPerSubCore, 0), tla::MakeShape(mPerSubCore, n0Actual));
                copyL0CToDstSub0(ubCTensorTlaTile, l0CTensorTlaTileSub0, false, 0); // AIV0
                copyL0CToDstSub1(ubCTensorTlaTile, l0CTensorTlaTileSub1, true, 0);  // AIV1
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
            } // end for(nL0Loop)
        } // end for(nL1Loop)

        // crossCoreSync after all fixpipe complete
        SetCrossCoreSync<4, PIPE_FIX>(mm1ToSmFlag);
    }

protected:
    AscendC::LocalTensor<ElementA> l1ATensor[L1A_BUF_NUM];
    AscendC::LocalTensor<ElementB> l1BTensor[L1B_BUF_NUM];
    AscendC::LocalTensor<ElementA> l0ATensor[L0_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[L0_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[L0_STAGES];

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
};

} // namespace Catlass::Gemm::Block
#endif // GEMM_BLOCK_BLOCK_MMAD_RAIN_FUSION_ATTENTION_QK_HPP
