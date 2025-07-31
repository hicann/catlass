/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* Thie file is part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License");
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text if the License.
*/

#ifndef CATLASS_GEMM_BLOCK_MMAD_W4A8_LOCAL_HPP
#define CATLASS_GEMM_BLOCK_MMAD_W4A8_LOCAL_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource/hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"

namespace Catlass::Gemm::Block {

template <
    class ArchTag,
    class ElementIn_,
    class ElementOut_,
    class Layout_,
    class TileShape_,
    uint32_t STAGES = 2
>
struct PrologueW4A8 {
    using ElementIn = ElementIn_;
    using ElementOut = ElementOut_;
    using Layout = Layout_;
    using TileShape = TileShape_;

    static constexpr uint32_t ELE_NUM_PER_BLK_INT8 = BYTE_PER_BLK / sizeof(ElementIn);

    static constexpr uint32_t COMPUTE_LEN = 16 *1024;

    /// Constructor
    CATLASS_DEVICE
    PrologueW4A8(Arch::Resource<ArchTag> &resource, uint32_t ubBufAddrStart = 0) {
        if (g_coreType == AscendC::AIV) {
            uint32_t ubOffset = ubBufAddrStart;
            uint32_t ubInSize = COMPUTE_LEN * sizeof(ElementIn) / 2;
            uint32_t ubOutSize = COMPUTE_LEN * sizeof(ElementOut);
            uint32_t ubWorkSpaceSize = COMPUTE_LEN * sizeof(half);

            for (uint32_t i = 0; i < STAGES; ++i) {
                unInTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementIn>(ubOffset);
                ubOffset += ubInSize;
                unOutTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementOut>(ubOffset);
                ubOffset += ubOutSize;
                unWorkSpaceList[i] = resource.ubBuf.template GetBufferByByte<half>(ubOffset);
                ubOffset += ubWorkSpaceSize;

                ubEventList[i] = i;
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ubEventList[i]);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(ubEventList[i]);
            }
        }
    }

    CATLASS_DEVICE
    void Operator()(AscendC::GlobalTensor<ElementOut> const &gmDst,
        AscendC::GlobalTensor<ElementIn> const &gmSrc, Layout const &layoutDst, Layout const &layoutSrc)
        {
            uint32_t tilesNum = layoutSrc.shape(0)
            uint32_t tileLen = layoutSrc.shape(1);
            uint32_t tileLenRoundInt8 = RoundUp(layoutSrc.shape(1), ELE_NUM_PER_BLK_INT8);
            uint32_t tileLenRoundInt4 = RoundUp(layoutSrc.shape(1) + 1) / 2,ELE_NUM_PER_BLK_INT8;
            uint64_t tileStrideSrc = layoutSrc.stride(0);
            uint64_t tileStrideDst = layoutDst.stride(0);
            if constexpr (std::is_same_v<Layout,layout::ColumnMajor>) {
                tilesNum = layoutSrc.shape(1);
                tileLen = layoutSrc.shape(0);
                tileLenRoundInt8 = RoundUp(layoutSrc.shape(0), ELE_NUM_PER_BLK_INT8);
                tileLenRoundInt4 = RoundUp(layoutSrc.shape(0) + 1) / 2,ELE_NUM_PER_BLK_INT8;
                tileStrideSrc = layoutSrc.stride(1);
                tileStrideDst = layoutDst.stride(1);
            }
            uint32_t tilesPerAiv = tileNum / AscendC::GetSubBlockNum();
            if (AscendC::GetSubBlockIdx() < (tilesNum % AscendC::GetSubBlockNum())) {
                tilesPerAiv++;
            }
            uint64_t taskOffsetSrc = AscendC::GetSubBlockIdx() * tilesPerAiv * ((tileStrideSrc + 1) / 2);
            uint64_t taskOffsetDst = AscendC::GetSubBlockIdx() * tilesPerAiv * tileStrideDst;
            if (AscendC::GetSubBlockIdx() >= (tilesNum % AscendC::GetSubBlockNum())) {
                taskOffsetSrc += (tilesNum % AscendC::GetSubBlockNum()) * ((tileStrideSrc + 1) / 2);
                taskOffsetDst += (tilesNum % AscendC::GetSubBlockNum()) * tileStrideDst;
            }
            uint32_t tilesPerLoop = 32;
            uint32_t loops = CeilDiv(tilesPerAiv, tilesPerLoop);
            uint32_t pingpong = 0
            for (uint32_t loopIdx = 0; loopIdx < loops; loopIdx++) {
                uint32_t actualTiles = tilesPerLoop;
                if (loopIdx == loops - 1) {
                    actualTiles = tilesPerAiv - loopIdx * tilesPerLoop;
                }
                uint64_t tileOffsetSrc = loopIdx * tilesPerLoop * ((tileStrideSrc + 1) / 2);
                AscendC::DataCopyExtParams DataCopyExtParamsIn(
                    actualTiles, (tileLen+1)/2 *sizeof(ElementIn), 
                    ((tileStrideSrc+1) / 2 - (tileLen+1) / 2) * sizeof(ElementIn),
                    (tileLenRoundInt4 - (tileLen+1) / 2) * ELE_NUM_PER_BLK_INT8, 0
                );
                AscendC::DataCopyPadExtParams<ElementIn> padParams(false, 0, 0, 0);

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ubEventList[pingpong]);
                AscendC::DataCopyPad(unInTensorList[pingpong],
                    gmSrc[taskOffsetSrc + tileOffsetSrc], DataCopyParamsIn,padParams);
                
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(ubEventList[pingpong]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(ubEventList[pingpong]);

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(ubEventList[pingpong]);

                int4_workspace[pingpong] = ubInTensorList[pingpong].template ReinterpretCast<AscendC::int4b_t>();

                AscendC::SetFlag<AscendC::HardEvent::S_V>(ubEventList[pingpong]);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(ubEventList[pingpong]);

                AscendC::Cast(ubWorkSpaceList[pingpong], int4_workspace[pingpong],
                    AscendC::RoundMode::CAST_NONE, actualTiles * tileLenRoundInt4 *2);

                AscendC::PipeBarrier<PIPI_V>();

                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ubEventList[pingpong]);

                AscendC::Cast(ubOutTensorList[pingpong], ubWorkSpaceList[pingpong],
                    AscendC::RoundMode::CAST_NONE, actualTiles * tileLenRoundInt4 *2);
                
                AscendC::SetFlag<AscendC::HardEvent::V_MET3>(ubEventList[pingpong]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MET3>(ubEventList[pingpong]);

                uint64_t tileOffsetDst = loopIdx * tilesPerLoop * tileStrideDst;
                AscendC::DataCopyExtParams DataCopyParasOut(
                    actualTiles,tileLen * sizeof(ElementOut),
                    (tileLenRoundInt4 * 2 - tileLen) * ELE_NUM_PER_BLK_INT8,
                    (tileStrideDst - tileLen) * sizeof(ElementOut),0
                )
                AscendC::DataCopyPad(gmDst[taskOffsetDst + tileOffsetDst],
                    ubOutTensorList[pingpong], DataCopyParasOut);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(ubEventList[pingpong]);

                pingpong = (pingpong + 1) % STAGES;
            }
        }
        
    ///Destructor
    CATLASS_DEVICE
    ~PrologueCastW4A8() {
        if (g_coreType == AscendC::AIV) {
            for (uint32_t i = 0; i < STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ubEventList[i]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(ubEventList[i]);
            }
        }
    }

protected:
    //Data members
    AscendC::LocalTensor<ElementIn> unInTensorList[STAGES];
    AscendC::LocalTensor<ElementOut> unOutTensorList[STAGES];
    AscendC::LocalTensor<half> unWorkSpaceList[STAGES];
    AscendC::LocalTensor<AscendC::int4b_t> int4_workspace[STAGES];

    int32_t ubEventList[STAGES];  
};


template <
    bool ENABLE_UNIT_FLAG_,
    bool ENABLE_SHUFFLE_K_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_
>
struct BlockMmad <
    MmadAtlasA2W4A8Local<ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>,
    L1TileShape_,
    L0TileShape_,
    AType_,
    BType_,
    CType_,
    BiasType_,
    TileCopy_,
    TileMmad_
> {
public:
    // Type Aliases
    using DispatchPolicy = MmadAtlasA2W4A8Local<ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>;
    using ArchTag = typename TileMmad_::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using TileMmad = TileMmad_;
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using ElementAccumulator = 
        typename Gemm::helper::ElementAccumulator<ElementA, ElementB>::ElementAccumulator;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<
        ArchTag, ElementAccumulator, CType_, Tile::ScaleGranularity::PER_TENSOR>;
    using LayoutAInL1 = typename CopyGmToL1A::LayoutSrc;
    using LayoutBInL1 = typename CopyGmToL1B::LayoutSrc;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutCInL0 = layout::zN;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA,LayoutA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB,LayoutB>;

    using TileShapeB = MatrixShape<L1TileShape::K, L1TileShape::N>;
    using PrologueCastB = PrologueW4A8<ArchTag, int8_t, ElementB, LayoutB, TileShapeB>;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr bool ENABLE_SHUFFLE_K = DispatchPolicy::ENABLE_SHUFFLE_K;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L1B_SIZE = L1TileShape::K * L1TileShape::N * sizeof(ElementB);
    static constexpr uint32_t L0A_SIZE = Arch::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = Arch::L0B_SIZE;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;

    //Check L1TileShape
    static_assert((L1A_SIZE * STAGES + L1B_SIZE * STAGES) <= Arch::L1_SIZE,
        "L1TileShape exceeding the L1 space!");
    
    //Check L0TileShape
    static constexpr uint32_t L0A_TILE_SIZE = L0TileShape::M * L0TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0TileShape::K * L0TileShape::N * sizeof(ElementB);
    static_assert((L0A_TILE_SIZE * STAGES) <= Arch::L0_SIZE,
        "L0TileShape exceeding the L0A space!");
    static_assert((L0B_TILE_SIZE * STAGES) <= Arch::L0_SIZE,
        "L0TileShape exceeding the L0B space!");

    static_assert(L1TileShape::M == L0TileShape::M && L1TileShape::N == L0TileShape::N,
        "The situation where the basic blocks of L1 and L0 differ on the m and n axes is not supported yet");
    
    //Check L1TileShape
    CATLASS_DEVICE
    BlockMmad(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0) : prologueCastB(resource)
    {
        if (g_coreType == AscendC::AIV) {
            uint32_t l1AOffset = l1BufAddrStart;
            uint32_t l1BOffset = l1AOffset + L1A_SIZE * STAGES;
            // Init buffers
            for (uint32_t i = 0; i < STAGES; i++) {
                L1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + i * L1A_SIZE);
                L1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + i * L1B_SIZE);
                L0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(i * L0A_PINGPONG_BUF_SIZE);
                L0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(i * L0B_PINGPONG_BUF_SIZE);

                l1AEventList[i] = i;
                l1BEventList[i] = i + STAGES;
                l0AEventList[i] = i;
                l0BEventList[i] = i + STAGES;
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            }
            l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);

            Catlass::Arch::CrossCoreSetFlag<0x02, PIPE_MTE2>(notifyAiv[0]);
            Catlass::Arch::CrossCoreSetFlag<0x02, PIPE_MTE2>(notifyAiv[1]);
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockMmad() 
    {
        if (g_coreType == AscendC::AIC) {
            for (uint32_t i = 0; i < STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            }
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        } else {
            Catlass::Arch::CrossCoreWaitFlag(notifyAiv[0]);
            Catlass::Arch::CrossCoreWaitFlag(notifyAiv[1]);
        }
    }

    /// Prologue: cast int4_t to int8_t (w4a8)
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<int8_t> const &gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<int8_t> const &gmNextB,
        AscendC::GlobalTensor<ElementB> const &gmBW,
        GemmCoord const &actualShape, GemmCoord const &actualShapeNext,
        bool isFirstBlock, bool hasNextBlock, GemmCoord const &problemShape) 
    {
        uint32_t kTileCount = CeilDiv<L1TileShape::K>(actualShape.k());

        uint32_t wkspStrideB = L1TileShape::N;
        if (std::is_same_v<LayoutB, layout::ColumnMajor>) {
            wkspStrideB = L1TileShape::K;
        }

        uint32_t startTileIdx = 0;
        if constexpr (ENABLE_SHUFFLE_K) {
            startTileIdx = AscendC::GetBlockIdx() / 2;
        }
        uint32_t firstTileIdx = startTileIdx * kTileCount;
        uint32_t lastTileIdx = (startTileIdx + kTileCount - 1) * kTileCount;
        uint32_t kActual = 
            (isFirstBlock < kTileCount) ? LiTileShape::K :(actualShape.k() - firstTileIdx * L1TileShape::K);

        // k loop
        for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; kLoopIdx++) {
            uint32_t shuffleKIdx = (startTileIdx + kLoopIdx) % kTileCount;
            if (shuffleKIdx == firstTileIdx && isFirstBlock) {
                auto gmTileB = gmB[shuffleKIdx * L1TileShape::K * ((problemShape.n() + 1) / 2)];
                if constexpr (std::is_same_v<LayoutB, layout::ColumnMajor>) {
                    gmTileB = gmB[(shuffleKIdx * L1TileShape::K + 1)* / 2];
                }

                //Load first matrix B tile from GM to L1
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kActual, problemShape.n()));
                auto layoutWB = LayoutB{kActual, problemShape.n(),wkspStrideB};

                Catlass::Arch::CrossCoreSetFlag(notifyAiv[l1ListId]);
                prologueCastB(gmBW[l1ListId * L1TileShape::K * L1TileShape::N],
                    gmTileB, layoutWB, layoutTileB);
                Catlass::Arch::CrossCoreSetFlag<0x02, PIPE_MTE3>(notifyAiv[l1ListId]);
            }

            uint32_t l1ListIdNext = (l1ListId + 1 < STAGES) ? (l1ListId + 1) : 0;
            uint32_t kActualNext{0};

            if (shuffleKIdx != lastTileIdx) {
                uint32_t shuffleKIdxNext = (startTileIdx + kLoopIdx + 1) % kTileCount;
                //Get GM tensor for next stage
                kActualNext = (shuffleKIdxNext < kTileCount - 1) ? 
                    L1TileShape::K :(actualShapeNext.k() - shuffleKIdxNext * L1TileShape::K);
                auto gmTileB = gmB[shuffleKIdxNext * L1TileShape::K  * ((problemShape.n() + 1) / 2)];
                if constexpr (std::is_same_v<LayoutB, layout::ColumnMajor>) {
                    gmTileB = gmB[(shuffleKIdxNext * L1TileShape::K + 1) / 2];
                }
                //Load first matrix B tile from GM to L1
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kActualNext, problemShape.n()));
                auto layoutWB = LayoutB{kActualNext, problemShape.n(), wkspStrideB};

                Catlass::Arch::CrossCoreSetFlag(notifyAiv[l1ListIdNext]);
                prologueCastB(gmBW[l1ListIdNext * L1TileShape::K * L1TileShape::N],
                    gmTileB, layoutWB, layoutTileB);
                Catlass::Arch::CrossCoreSetFlag<0x02, PIPE_MTE3>(notifyAiv[l1ListIdNext]);
            }

            if (shuffleKIdx == lastTileIdx && hasNextBlock) {
                kActualNext = (firstTileIdx < kTileCount - 1) ? 
                    L1TileShape::K :(actualShapeNext.k() - firstTileIdx * L1TileShape::K);
                auto gmTileB = gmNextB[firstTileIdx * L1TileShape::K * ((problemShape.n() + 1) / 2)];
                if constexpr (std::is_same_v<LayoutB, layout::ColumnMajor>) {
                    gmTileB = gmNextB[(firstTileIdx * L1TileShape::K + 1) / 2];
                }
                //Load first matrix A tile from GM to L1
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kActualNext, problemShape.n()));
                auto layoutWB = LayoutB{kActualNext, problemShape.n(), wkspStrideB};

                Catlass::Arch::CrossCoreSetFlag(notifyAiv[l1ListIdNext]);
                prologueCastB(gmBW[l1ListIdNext * L1TileShape::K * L1TileShape::N],
                    gmTileB, layoutWB, layoutTileB);
                Catlass::Arch::CrossCoreSetFlag<0x02, PIPE_MTE3>(notifyAiv[l1ListIdNext]);
            }
            l1ListId = l1ListIdNext;
            kActual = kActualNext;
        }
    }

    //Perform a block-scoped matrix multiply-accumulate
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmB,
        AscendC::GlobalTensor<ElementC> const &gmC, LayoutC const &layoutC,
        AscendC::GlobalTensor<ElementB> const &gmNextA,
        GemmCoord const &actualShape, GemmCoord const &actualShapeNext,
        bool isFirstBlock, bool hasNextBlock, uint64_t deqScalar) 
    {
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());

        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);
        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mRound, nRound));

        uint32_t kTileCount = CeilDiv<L1TileShape::K>(actualShape.k());

        uint32_t wkspStrideB = L1TileShape::N;
        if (std::is_same_v<LayoutB, layout::ColumnMajor>) {
            wkspStrideB = L1TileShape::K;
        }

        if constexpr (ENABLE_UNIT_FLAG) {
            Catlass::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }
        uint32_t startTileIdx = 0;
        if constexpr (ENABLE_SHUFFLE_K) {
            startTileIdx = AscendC::GetBlockIdx();
        }
        uint32_t firstTileIdx = startTileIdx * kTileCount;
        uint32_t lastTileIdx = (startTileIdx + kTileCount - 1) * kTileCount;
        uint32_t kActual = 
            (isFirstBlock < kTileCount - 1 ) ? L1TileShape::K :(actualShape.k() - firstTileIdx * L1TileShape::K);
        
        uint32_t mPartLoop = CeilDiv<L1TileShape::M>(mRound);
        uint32_t nPartLoop = CeilDiv<L1TileShape::N>(nRound);

        // k loop
        for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; kLoopIdx++) {
            uint32_t shuffleKIdx = (startTileIdx + kLoopIdx) % kTileCount;
            // Load first matrix A tile in total kernel loop from GM to L1
            if (shuffleKIdx == firstTileIdx && isFirstBlock) {
                MatrixCoord gmTileAOffset{0, shuffleKIdx * L1TileShape::K};
                auto gmTileA = gmA[layoutA.GetOffset(gmTileAOffset)];
                
                //Load first matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), kActual));
                CopyGmToL1A(L1ATensorList[l1ListId], gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);

                //Load first matrix B tile from L1 to L0
                Catlass::Arch::CrossCoreSetFlag(notifyAiv[l1ListId]);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l1ListId]);
                auto layoutTileB = LatoutB{kActual, actualShape.n(), wkspStrideB};
                CopyL1ToL0B(L0BTensorList[l0ListId], gmB[l1ListId * L1TileShape::K * L1TileShape::N], layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l0BEventList[l1ListId]);
                Catlass::Arch::CrossCoreSetFlag<0x02, PIPE_MTE3>(notifyAiv[l1ListId]);
                
            }

            uint32_t l1ListIdNext = (l1ListId + 1 < STAGES) ? (l1ListId + 1) : 0;
            uint32_t kActualNext{0};

            // preload next tile from GM to L1
            if (shuffleKIdx != lastTileIdx) {
                uint32_t shuffleKIdxNext = (startTileIdx + kLoopIdx + 1) % kTileCount;
                // Get L1 tensor for next stage
                auto l1ATensor = L1ATensorList[l1ListIdNext];
                auto l1BTensor = L1BTensorList[l1ListIdNext];
                // Get GM tensor for next stage
                kActualNext = (shuffleKIdxNext < kTileCount - 1) ? 
                    L1TileShape::K :(actualShape.k() - shuffleKIdxNext * L1TileShape::K);
                MatrixCoord gmTileAOffset{0, shuffleKIdxNext * L1TileShape::K};
                auto gmTileA = gmA[layoutA.GetOffset(gmTileAOffset)];

                //load next matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), kActualNext));
                CopyGmToL1A(l1ATensor, gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                //Load next matrix B tile from GM to L1
                Catlass::Arch::CrossCoreSetFlag(notifyAic[l1ListIdNext]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                auto layoutTileB = LatoutB{kActualNext, actualShape.n(), wkspStrideB};
                CopyGmToL1B(l1BTensor, gmB[l1ListIdNext * L1TileShape::K * L1TileShape::N], layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                Catlass::Arch::CrossCoreSetFlag<0x02, PIPE_MTE3>(notifyAiv[l1ListIdNext]);
            }
            if (shuffleKIdx == lastTileIdx && hasNextBlock) {
                // Get L1 tensor for next stage
                auto l1ATensor = L1ATensorList[l1ListIdNext];
                auto l1BTensor = L1BTensorList[l1ListIdNext];
                // Get GM tensor for next stage
                kActualNext = (firstTileIdx < kTileCount - 1) ?
                    L1TileShape::K :(actualShapeNext.k() - firstTileIdx * L1TileShape::K);
                MatrixCoord gmTileAOffset{0, firstTileIdx * L1TileShape::K};
                auto gmTileA = gmNextA[layoutA.GetOffset(gmTileAOffset)];

                //Load next matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShapeNext.m(), kActualNext));
                CopyGmToL1A(l1ATensor, gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                //Load next matrix B tile from GM to L1
                Catlass::Arch::CrossCoreSetFlag(notifyAic[l1ListIdNext]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                auto layoutTileB = LatoutB{kActualNext, actualShapeNext.n(), wkspStrideB};
                CopyGmToL1B(l1BTensor, gmB[l1ListIdNext * L1TileShape::K * L1TileShape::N], layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                Catlass::Arch::CrossCoreSetFlag<0x02, PIPE_MTE2>(notifyAiv[l1ListIdNext]);
            }

            //Get L1 tensor for current stage
            auto l1ATensor = L1ATensorList[l1ListId];
            auto l1BTensor = L1BTensorList[l1ListId];

            //Get the loop nums on L0
            uint32_t kPartLoop = CeilDiv<L0TileShape::K>(kActual);

            uint32_t l0ABufId = 0
            uint32_t l0BBufId = 0;

            for (int mPartIdx= 0; mPartIdx < mPartLoop; mPartIdx++) {
                uint32_t mPartActual = (mPartIdx < mPartLoop - 1) ? L0TileShape::M : 
                    (mRound - mPartIdx * L0TileShape::M);
                
                for (int kPartIdx = 0; kPartIdx < kPartLoop; kPartIdx++) {
                    uint32_t kPartActual = (kPartIdx < kPartLoop - 1) ? L0TileShape::K : 
                        (kActual - kPartIdx * L0TileShape::K);
                    
                    // Locate the current tile on L0A
                    auto l0ATile = L0ATensorList[l0ListId];
                    LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mPartActual, kPartActual);
                    //Locate the current tile of matrix A on L1
                    MatrixCoord l1AOffset{mPartIdx * L0TileShape::M, kPartIdx * L0TileShape::K};
                    auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1AOffset)];

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ABufId]);
                    if ((mPartIdx == 0) && (kPartIdx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
                    }

                    copyL1toL0A(l0ATile, l1ATile, layoutAInL0, layoutAInL1);

                    if ((mPartIdx ==mPartLoop - 1) && (kPartIdx == kPartLoop - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                    }

                    for (int nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++) {
                        uint32_t nPartActual = (nPartIdx < nPartLoop - 1) ? L0TileShape::N : 
                            (nRound - nPartIdx * L0TileShape::N);
                        
                        // Locate the current tile on L0B
                        auto l0BTile = L0BTensorList[l0BBufId];
                        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kPartActual, nPartActual);
                        // Locate the current tile of matrix B on L1
                        MatrixCoord l1BOffset{kPartIdx * L0TileShape::K, nPartIdx * L0TileShape::N};
                        auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BOffset)];

                        // Wait for mmad finished
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BBufId]);
                        // If the current tile is the first one on the k&n axes, wait for loading matrix B from GM to L1
                        if ((kPartIdx == 0) && (nPartIdx == 0)) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
                        }

                        // Load current tile from L1 to L0
                        copyL1toL0B(l0BTile, l1BTile, layoutBInL0, layoutBInL1);

                        // If the current tile is the last one on the k&n axes, notify to load matrix B from GM to L0
                        if ((kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                        }
                        // Notify to do mmad
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                        //Locate the current tile on L0C
                        MatrixCoord l0COffset{mPartIdx * L0TileShape::M, nPartIdx * L0TileShape::N};
                        auto l0CTile = l0CTensor[layoutInL0C.GetOffset(l0COffset)];

                        // Compute the matrix multiplication on L0A and L0B and write the result to the accumulator
                        // Wait for loading L0B
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                        // If the current tile is the first one on the k axes, the accumulator needs to be reset to 0
                        bool initC = ((kLoopIdx == 0) && (kPartIdx == 0));
                        // If the unit flag is enabled, the unit flag is set to according to the calculation progress
                        uint8_t unitFlag = 0b00;
                        if constexpr (ENABLE_UNIT_FLAG) {
                            if ((kLoopIdx = kTileCount - 1) && (kPartIdx == kPartLoop - 1) &&
                                (nPartIdx == nPartLoop - 1) && (mPartIdx == mPartLoop - 1)) {
                                unitFlag = 0b11; 
                            } else {
                                unitFlag = 0b10; // last tile on k and n axes
                            } 
                        }
                        // Perform calculation operations
                        tileMmad(l0CTile, l0ATile, l0BTile, mPartActual, nPartActual, kPartActual, initC, unitFlag);

                        // Notify to move the next L0B tile
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BBufId]);

                        l0BBufId = ((l0BBufId + 1) < STAGES) ? (l0BBufId + 1) : 0;
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ABufId]);
                    l0ABufId = ((l0ABufId + 1) < STAGES) ? (l0ABufId + 1) : 0;
                }
            }

            l1ListId = l1ListIdNext;
            kActual = kActualNext;
        }

        // copy block out
        LayoutC layoutBlock = layoutC.GetTileLayout(actualShape.GetCoordMN());

        if constexpr (ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            CopyL0CToGm(gmC, l0CTensor, layoutBlock, layoutInL0C, deqScalar);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        } else {
            CopyL0CToGm(gmC, l0CTensor, layoutBlock, layoutInL0C, 0b11);
        }
    }

protected:
    ///Data members
    AscendC::LocalTensor<ElementA> L1ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> L1BTensorList[STAGES];
    AscendC::LocalTensor<ElementA> L0ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> L0BTensorList[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;

    int32_t l1AEventList[STAGES];
    int32_t l1BEventList[STAGES];
    int32_t l0AEventList[STAGES];
    int32_t l0BEventList[STAGES];

    Arch::CrossCoreFlag notifyAic[STAGES] = { EVENT_ID0, EVENT_ID1 };
    Arch::CrossCoreFlag notifyAiv[STAGES] = { EVENT_ID2, EVENT_ID3 };

    uint32_t l1ListId{0};

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
    PrologueCastB prologueCastB;
};

} // namespacec Catlass::Gemm::Block

#endif // CATLASS_GEMM_BLOCK_MMAD_ATLAS_A2W4A8_LOCAL_HPP
