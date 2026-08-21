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
 * @brief matmul implementation for single p&v base tile.
 * This implementation is designed for the following senario:
 * A full p base tile is loaded to L1 from UB, no workspace transit
 * A full v base tile is loaded to L1 from GM, relevant instructions launched before p base tile crossCore wait
 * A full p*v base tile is loaded to UB from l0C, no workspace transit
 */
#ifndef GEMM_BLOCK_BLOCK_MMAD_RAIN_FUSION_ATTENTION_PV_HPP
#define GEMM_BLOCK_BLOCK_MMAD_RAIN_FUSION_ATTENTION_PV_HPP

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
    MmadAtlasA5RainPV<ArchTag_>, L1TileShape_, L0TileShape_, ElementA_, ElementB_, ElementC_, ElementBias_, TileCopy_,
    TileMmad_> {
public:
    using ArchTag = ArchTag_;
    using DispatchPolicy = MmadAtlasA5RainPV<ArchTag>;
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

    static constexpr uint32_t L1A_BUF_NUM = DispatchPolicy::L1A_STAGES; // P, 3-buf
    static constexpr uint32_t L1B_BUF_NUM = DispatchPolicy::L1B_STAGES; // V, 2-buf
    static constexpr uint32_t L0_STAGES = DispatchPolicy::L0_STAGES;
    // L1 tile shape
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{}); // s1
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{}); // d
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{}); // s2
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
        L1_TILE_M == L0_TILE_M && L1_TILE_N == L0_TILE_N,
        "The situation where the basic blocks of L1 and L0 differ on the m, n axes is not supported yet");

    __aicore__ inline BlockMmadTla(Arch::Resource<ArchTag>& resource, uint32_t l1BufAddrStart)
    {
        for (uint32_t i = 0; i < L1A_BUF_NUM; i++) {
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart + L1A_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < L1B_BUF_NUM; i++) {
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(
                l1BufAddrStart + L1A_BUF_SIZE * L1A_BUF_NUM + L1B_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < L0_STAGES; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(
                L0C_HALF_BUF_SIZE + L0C_PINGPONG_BUF_SIZE * i);
        }
    }

    __aicore__ inline ~BlockMmadTla()
    {}

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
        Arch::CrossCoreFlag smToMm2Flag, Arch::CrossCoreFlag mm2ToReFlag)
    {
        using CopyL0CToDst = typename TileCopy_::template CopyL0CToDst<TensorC>;

        uint32_t rowNum = actualOriShape[0];          // Block层actualM, s1
        uint32_t embed = actualOriShape[1];           // Block层actualN, d
        uint32_t curBaseTileSize = actualOriShape[2]; // Block层actualK, s2

        uint32_t l1ABufId = kvSTileIdx % L1A_BUF_NUM;
        uint32_t l1BBufId = kvSTileIdx % L1B_BUF_NUM;
        uint32_t l1BEventId = l1BBufId + L1A_BUF_NUM; // L1B event id: 3,4

        auto l1ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL1A>(rowNum, curBaseTileSize);
        auto l1ATensorTla = tla::MakeTensor(l1ATensor[l1ABufId], l1ALayoutTla, Arch::PositionL1{}); // L1A: P
        auto l1BLayoutTla = tla::MakeLayout<ElementB, LayoutTagL1B>(curBaseTileSize, embed);
        auto l1BTensorTla = tla::MakeTensor(l1BTensor[l1BBufId], l1BLayoutTla, Arch::PositionL1{}); // L1B: V

        // load V full base tile to L1
        uint32_t gatheredKvStartOffset = kvSTileIdx * kvSBaseTile; // kvSBaseTile==L1_TILE_K
        uint32_t gatheredYBlockIdx = gatheredKvStartOffset / blockShapeY;
        uint32_t oriYBlockIdx = gSelectIdx.GetValue(gatheredYBlockIdx);
        uint32_t processedSize = 0; // 逐稀疏block搬移填充基本块过程中，已处理的累积序列长度
        uint32_t yBlockInnerOffset = gatheredKvStartOffset % blockShapeY;
        uint32_t oriKvYOffset = oriYBlockIdx * blockShapeY + yBlockInnerOffset; // origin y offset

        using CopyGmToL1B = typename TileCopy_::template CopyGmToL1B<TensorB>;
        CopyGmToL1B copyGmToL1B;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventId);

        while (processedSize < curBaseTileSize && gatheredYBlockIdx < selectNum && oriYBlockIdx < kvYBlockNum &&
               oriKvYOffset < kvSeqlen) {
            uint32_t yBlockActualSize =
                (gatheredYBlockIdx == selectNum - 1 && oriYBlockIdx == kvYBlockNum - 1 && kvSeqlen % blockShapeY != 0) ?
                    (kvSeqlen - blockShapeY * oriYBlockIdx) :
                    blockShapeY;
            uint32_t remainingInYBlock = yBlockActualSize - yBlockInnerOffset;
            uint32_t remainingNInL1B = curBaseTileSize - processedSize;

            uint32_t curCopyYSize = min(remainingNInL1B, remainingInYBlock);
            if (curCopyYSize == 0) {
                break;
            }

            auto l1BTensorTlaTile = GetTile(
                l1BTensorTla, tla::MakeCoord(processedSize, 0), tla::MakeShape(curCopyYSize, embed)); // L1B Tile
            auto gBTensorTlaTile =
                GetTile(gBTensor, tla::MakeCoord(oriKvYOffset, 0), tla::MakeShape(curCopyYSize, embed)); // gmB Tile
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

        // forward crossCoreSync from OnlineSoftmax to mm2
        WaitCrossCoreSync<4, PIPE_MTE1>(smToMm2Flag);
        // P full base tile already on L1
        uint32_t mL0Loop = CeilDiv<L0_TILE_M>(rowNum);
        uint32_t nL0Loop = CeilDiv<L0_TILE_N>(embed);
        uint32_t kL0Loop = CeilDiv<L0_TILE_K>(curBaseTileSize);

        for (uint32_t nL0Itr = 0; nL0Itr < nL0Loop; nL0Itr++) {
            uint32_t n0Actual = (nL0Itr == nL0Loop - 1) ? (embed - nL0Itr * L0_TILE_N) : L0_TILE_N;
            uint32_t nLoopCounter = GetCurLoopCounter(kvSTileIdx, nL0Loop, nL0Itr);
            // l0C nbuffer chunked only in n loop
            uint32_t l0CLoopCounter = nLoopCounter;
            uint32_t l0CBufId = l0CLoopCounter % L0_STAGES;
            uint32_t l0CEventId = l0CBufId + L0_STAGES; // L0C event id: 2,3
            auto l0CLayoutTla = tla::MakeLayoutL0C(rowNum, n0Actual);
            auto l0CTensorTla = tla::MakeTensor(l0CTensor[l0CBufId], l0CLayoutTla, Arch::PositionL0C{}); // L0C

            for (uint32_t mL0Itr = 0; mL0Itr < mL0Loop; mL0Itr++) {
                uint32_t m0Actual = (mL0Itr == mL0Loop - 1) ? (rowNum - mL0Itr * L0_TILE_M) : L0_TILE_M;
                // different m chunks will be concated in the same piece of l0C buffer
                auto l0CTensorTlaTile = GetTile(
                    l0CTensorTla, tla::MakeCoord(mL0Itr * L0_TILE_M, 0),
                    tla::MakeShape(m0Actual, n0Actual)); // L0C Tile

                for (uint32_t kL0Itr = 0; kL0Itr < kL0Loop; kL0Itr++) {
                    uint32_t k0Actual = (kL0Itr == kL0Loop - 1) ? (curBaseTileSize - kL0Itr * L0_TILE_K) : L0_TILE_K;
                    uint32_t l0ALoopCounter = prefixSumL0AStages + GetCurLoopCounter(mL0Itr, kL0Loop, kL0Itr);
                    uint32_t l0BLoopCounter = prefixSumL0BStages + GetCurLoopCounter(nL0Itr, kL0Loop, kL0Itr);
                    uint32_t l0ABufId = l0ALoopCounter % L0_STAGES;
                    uint32_t l0BBufId = l0BLoopCounter % L0_STAGES;
                    uint32_t l0AEventId = l0ABufId;             // L0A event id: 0,1
                    uint32_t l0BEventId = l0BBufId + L0_STAGES; // L0B event id: 2,3

                    auto l1BTensorTlaTile = GetTile(
                        l1BTensorTla, tla::MakeCoord(kL0Itr * L0_TILE_K, nL0Itr * L0_TILE_N),
                        tla::MakeShape(k0Actual, n0Actual));
                    auto l0BLayoutTla = tla::MakeLayout<ElementB, LayoutTagL0B>(k0Actual, n0Actual);
                    auto l0BTensorTla = tla::MakeTensor(l0BTensor[l0BBufId], l0BLayoutTla, Arch::PositionL0B{});

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                    copyL1ToL0B(l0BTensorTla, l1BTensorTlaTile);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                    bool isLastLoop = (mL0Itr == mL0Loop - 1) && (nL0Itr == nL0Loop - 1) && (kL0Itr == kL0Loop - 1);
                    if (isLastLoop) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventId);
                    }

                    auto l1ATensorTlaTile = GetTile(
                        l1ATensorTla, tla::MakeCoord(mL0Itr * L0_TILE_M, kL0Itr * L0_TILE_K),
                        tla::MakeShape(m0Actual, k0Actual));
                    auto l0ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL0A>(m0Actual, k0Actual);
                    auto l0ATensorTla = tla::MakeTensor(l0ATensor[l0ABufId], l0ALayoutTla, Arch::PositionL0A{});

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                    copyL1ToL0A(l0ATensorTla, l1ATensorTlaTile);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                    // reverse crossCoreSync for P
                    if (isLastLoop) {
                        SetCrossCoreSync<4, PIPE_MTE1>(smToMm2Flag);
                    }

                    bool initMmad = (kL0Itr == 0);
                    uint32_t l0TileMAligned = RoundUp(m0Actual, 16);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                    if (mL0Itr == 0 && kL0Itr == 0) {
                        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
                    }
                    tileMmad(
                        l0CTensorTlaTile, l0ATensorTla, l0BTensorTla, l0TileMAligned, n0Actual, k0Actual, initMmad);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                } // end for(kL0Loop)
            } // end for(mL0Loop)

            if (nL0Itr == 0) {
                // reverse crossCoreSync, do fixPipe only after ubCTensor is fully released
                WaitCrossCoreSync<4, PIPE_FIX>(mm2ToReFlag);
            }
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            // kernel传输ubCTensor的时候确保其shape的m,n是32B对齐（8个b32数据）
            // evenly split l0CTensorTla to 2 AIV:
            //     AIV0: valid rows [0, mFixPAligned8 / 2 - 1]
            //     AIV1: valid rows [mFixPAligned8 / 2, rowNum - 1]
            uint32_t mFixPAligned8 = RoundUp(rowNum, 8);
            uint32_t nFixPAligned8 = RoundUp(n0Actual, 8); // fp32
            CopyL0CToDst copyL0CToDst;
            auto ubCTensorTlaTile =
                GetTile(ubCTensor, tla::MakeCoord(0, nL0Itr * L0_TILE_N), tla::MakeShape(mFixPAligned8, nFixPAligned8));
            copyL0CToDst(ubCTensorTlaTile, l0CTensorTla);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
        } // end for(nL0Loop)

        // crossCoreSync after all fixPipe move
        SetCrossCoreSync<4, PIPE_FIX>(mm2ToReFlag);
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
#endif // GEMM_BLOCK_BLOCK_MMAD_RAIN_FUSION_ATTENTION_PV_HPP
