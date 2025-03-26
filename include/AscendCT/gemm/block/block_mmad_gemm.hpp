/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ASCENDCT_GEMM_BLOCK_BLOCK_GEMM_PRELOAD
#define ASCENDCT_GEMM_BLOCK_BLOCK_GEMM_PRELOAD

#include "AscendCT/AscendCT.hpp"
#include "AscendCT/gemm/helper.hpp"
#include "AscendCT/gemm/tile/tile_copy.hpp"
#include "AscendCT/gemm/tile/tile_mmad.hpp"
#include "AscendCT/matmul_coord.hpp"
#include "AscendCT/gemm/dispatch_policy.hpp"
#include "AscendCT/arch/resource.hpp"

namespace AscendCT::gemm::block{
template<
    bool ENABLE_UNIT_FLAG_,
    bool ENABLE_SHUFFLE_K_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class XType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_
>
struct BlockGemm<
    gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>,
    L1TileShape_,
    L0TileShape_,
    AType_,
    BType_,
    XType_,
    BiasType_,
    TileCopy_,
    TileMmad_
>{
public:
    using DispatchPolicy = gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementX = typename XType_::Element;
    using LayoutX = typename XType_::Layout;

    using TileMmad = TileMmad_;
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
    using ElementAccumulator =
        typename gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using LayoutAInL1 = typename CopyGmToL1A::LayoutDst;
    using LayoutBInL1 = typename CopyGmToL1B::LayoutDst;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutXInL0 = layout::zN;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr bool ENABLE_SHUFFLE_K = DispatchPolicy::ENABLE_SHUFFLE_K;
    const uint32_t L1Size = ArchTag::L1_SIZE;
    const uint32_t L1ASize = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    const uint32_t L1BSize = L1TileShape::K * L1TileShape::N * sizeof(ElementB);
    const uint32_t cSize = L1TileShape::M * L1TileShape::N * sizeof(ElementAccumulator);
    const uint32_t BlockCnt = L1TileShape::M * L1TileShape::N;
    const uint32_t L0ASize = ArchTag::L0A_SIZE;
    const uint32_t L0BSize = ArchTag::L0B_SIZE;
    const uint32_t L0CSize = ArchTag::L0C_SIZE;
    const uint32_t L0A_PINGPONG_BUF_LEN = (L0ASize / STAGES);
    const uint32_t L0B_PINGPONG_BUF_LEN = (L0BSize / STAGES);
    static constexpr bool RowOrColumn = std::is_same<LayoutA, layout::RowMajor>::value && std::is_same<LayoutB, layout::RowMajor>::value;
    const uint32_t l0XBlockNum = ArchTag::L0C_SIZE / cSize;
    
    ASCENDCT_DEVICE
    BlockGemm(arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0){
        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + L1ASize * STAGES;
        for(uint32_t i = 0; i < STAGES; i++){
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1ASize * i);
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1BSize * i);
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_LEN * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_LEN * i);
            // Assign event ID for each stages
            l1AEventList[i] = i;
            l1BEventList[i] = i + STAGES;
            l0AEventList[i] = i;
            l0BEventList[i] = i + STAGES;
            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        l0XTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
    }
    // destroy function
    ASCENDCT_DEVICE
    ~BlockGemm(){
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
    }

    ASCENDCT_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementX> const &gmX, LayoutX const &layoutX,
        AscendC::GlobalTensor<ElementA> const &gmNextBlockA,
        AscendC::GlobalTensor<ElementB> const &gmNextBlockB,
        MatmulCoord const &actualShape, MatmulCoord const &actualShapeNext,
        bool isFirstBlock, bool hasNextBlock, uint32_t singleIdx
    ){ 
        uint32_t K = actualShape.k();
        uint32_t maxKPerBlock = L1TileShape::K;
        uint32_t KLoops = CeilDiv(K, maxKPerBlock);
        // uint32_t startTileIdx{0};
        uint32_t startTileIdx = AscendC::GetBlockIdx();
        // if constexpr(!std::is_same<ElementA, float>::value && ENABLE_SHUFFLE_K){
        //     startTileIdx = AscendC::GetBlockIdx();
        // }
        uint32_t firstTileIdx = startTileIdx % KLoops; 
        uint32_t lastTileIdx = (startTileIdx + KLoops - 1) % KLoops; 
        uint32_t KGmActual = (firstTileIdx == KLoops - 1) ? (K - firstTileIdx * maxKPerBlock) : maxKPerBlock;
        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);
        for(uint32_t KIdx = 0; KIdx < KLoops; KIdx++){
            uint32_t shuffleKIdx = (startTileIdx + KIdx) % KLoops;
            if(shuffleKIdx == firstTileIdx && isFirstBlock){ 
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), KGmActual)); 
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(KGmActual, actualShape.n()));
                MatrixCoord gmTileAOffset{0, shuffleKIdx * maxKPerBlock}; 
                auto gmTileA = gmA[layoutA.GetOffset(gmTileAOffset)];
                MatrixCoord gmTileBOffset{shuffleKIdx * maxKPerBlock, 0}; 
                auto gmTileB = gmB[layoutB.GetOffset(gmTileBOffset)];
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]); 
                copyGmToL1A(l1ATensor[l1ListId], gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                copyGmToL1B(l1BTensor[l1ListId], gmTileB, layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
            }
            l1ListIdNext = 1 - l1ListId;
            uint32_t KGmActualNext = 0; 
            if(shuffleKIdx != lastTileIdx){ 
                uint32_t shuffleKIdxNext = (startTileIdx + KIdx + 1) % KLoops;
                KGmActualNext = (shuffleKIdxNext == KLoops - 1) ? (K - shuffleKIdxNext * maxKPerBlock) : maxKPerBlock; 
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), KGmActualNext)); 
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(KGmActualNext, actualShape.n()));
                MatrixCoord gmTileAOffset{0, shuffleKIdxNext * maxKPerBlock}; 
                auto gmTileA = gmA[layoutA.GetOffset(gmTileAOffset)];
                MatrixCoord gmTileBOffset{shuffleKIdxNext * maxKPerBlock, 0}; 
                auto gmTileB = gmB[layoutB.GetOffset(gmTileBOffset)];
                if(shuffleKIdxNext % 2 == 1){ 
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                    copyGmToL1B(l1BTensor[l1ListIdNext], gmTileB, layoutBInL1, layoutTileB);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                    copyGmToL1A(l1ATensor[l1ListIdNext], gmTileA, layoutAInL1, layoutTileA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);
                }else{
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                    copyGmToL1A(l1ATensor[l1ListIdNext], gmTileA, layoutAInL1, layoutTileA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                    copyGmToL1B(l1BTensor[l1ListIdNext], gmTileB, layoutBInL1, layoutTileB);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                }
            }
            if(shuffleKIdx == lastTileIdx && hasNextBlock){ 
                KGmActualNext = (firstTileIdx == KLoops - 1) ? (K - firstTileIdx * maxKPerBlock) : maxKPerBlock;
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShapeNext.m(), KGmActualNext)); 
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(KGmActualNext, actualShapeNext.n()));
                MatrixCoord gmTileAOffset{0, firstTileIdx * maxKPerBlock}; 
                auto gmNextTileA = gmNextBlockA[layoutA.GetOffset(gmTileAOffset)];
                MatrixCoord gmTileBOffset{firstTileIdx * maxKPerBlock, 0}; 
                auto gmNextTileB = gmNextBlockB[layoutB.GetOffset(gmTileBOffset)];
                if(shuffleKIdx % 2 == 0){ 
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                    copyGmToL1B(l1BTensor[l1ListIdNext], gmNextTileB, layoutBInL1, layoutTileB);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                    copyGmToL1A(l1ATensor[l1ListIdNext], gmNextTileA, layoutAInL1, layoutTileA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);
                }else{
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                    copyGmToL1A(l1ATensor[l1ListIdNext], gmNextTileA, layoutAInL1, layoutTileA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                    copyGmToL1B(l1BTensor[l1ListIdNext], gmNextTileB, layoutBInL1, layoutTileB);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                }
            }

            uint32_t KL0TileSize = L0TileShape::K;
            uint32_t KL0Loops = CeilDiv(KGmActual, KL0TileSize);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
            auto l1ATile = l1ATensor[l1ListId];
            auto l1BTile = l1BTensor[l1ListId];
            uint32_t MActual{0};
            uint32_t NActual{0};
            for(uint32_t KL0Idx = 0; KL0Idx < KL0Loops; KL0Idx++){
                uint32_t KL0Actual = (KL0Idx == KL0Loops - 1) ? (KGmActual - KL0Idx * KL0TileSize) : KL0TileSize;
                LayoutAInL0 layoutAInL0; LayoutBInL0 layoutBInL0;
                uint32_t l1TileAOffset = layoutAInL1.GetOffset(MatrixCoord(0, KL0Idx * KL0TileSize));
                uint32_t l1TileBOffset = layoutBInL1.GetOffset(MatrixCoord(KL0Idx * KL0TileSize, 0));
                AscendC::LocalTensor<ElementA> l0TileA;  AscendC::LocalTensor<ElementB> l0TileB;
                auto l1TileA = l1ATile[l1TileAOffset];
                auto l1TileB = l1BTile[l1TileBOffset];
                if constexpr(RowOrColumn){
                    l0TileA = l0ATensor[l0ListId];
                    l0TileB = l0BTensor[l0ListId];
                    MActual = L1TileShape::M;
                    NActual = L1TileShape::N;
                    layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(actualShape.m(), KL0Actual);
                    layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(actualShape.n(), KL0Actual);
                }else{
                    l0TileA = l0BTensor[l0ListId];
                    l0TileB = l0ATensor[l0ListId];
                    NActual = L1TileShape::M;
                    MActual = L1TileShape::N;
                    layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(KL0Actual, actualShape.m());
                    layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(KL0Actual, actualShape.n());
                }
                if(shuffleKIdx % 2 == 0){
                    if(KL0Idx % 2 == 0){ 
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0ListId]);
                        copyL1ToL0B(l0TileB, l1TileB, layoutBInL0, layoutBInL1);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0ListId]);
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ListId]);
                        copyL1ToL0A(l0TileA, l1TileA, layoutAInL0, layoutAInL1);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0ListId]);
                    }else{
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ListId]);
                        copyL1ToL0A(l0TileA, l1TileA, layoutAInL0, layoutAInL1);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0ListId]);
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0ListId]);
                        copyL1ToL0B(l0TileB, l1TileB, layoutBInL0, layoutBInL1);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0ListId]);
                    }
                }else{
                    if(KL0Idx % 2 == 0){
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ListId]);
                        copyL1ToL0A(l0TileA, l1TileA, layoutAInL0, layoutAInL1);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0ListId]);
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0ListId]);
                        copyL1ToL0B(l0TileB, l1TileB, layoutBInL0, layoutBInL1);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0ListId]);
                    }else{
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0ListId]);
                        copyL1ToL0B(l0TileB, l1TileB, layoutBInL0, layoutBInL1);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0ListId]);
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ListId]);
                        copyL1ToL0A(l0TileA, l1TileA, layoutAInL0, layoutAInL1);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0ListId]);
                    }
                }
                if(KL0Idx == KL0Loops - 1){ 
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                    l1ListId = l1ListIdNext;
                    KGmActual = KGmActualNext;
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0ListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0ListId]);
                tileMmad(l0XTensor[(singleIdx % l0XBlockNum) * BlockCnt], l0TileA, l0TileB, MActual, NActual, KL0Actual, (KIdx == 0) && (KL0Idx == 0));
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ListId]);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0ListId]);
                l0ListId = 1 - l0ListId;
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>((int32_t)(singleIdx % l0XBlockNum));
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>((int32_t)(singleIdx % l0XBlockNum));
        auto layoutInL0X = LayoutXInL0::MakeLayoutInL0C(MakeCoord(L1TileShape::M, L1TileShape::N)); 
        LayoutX layoutBlock = layoutX.GetTileLayout(MakeCoord(actualShape.m(), actualShape.n()));
        copyL0CToGm(gmX, l0XTensor[(singleIdx % l0XBlockNum) * BlockCnt], layoutBlock, layoutInL0X);
    }
private:
    AscendC::LocalTensor<ElementA> l1ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensor[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0XTensor;
    // Multi-stage event id list
    int32_t l1AEventList[STAGES];
    int32_t l1BEventList[STAGES];
    int32_t l0AEventList[STAGES];
    int32_t l0BEventList[STAGES];

    // The id of current stage
    uint32_t l1ListId{0};
    uint32_t l0ListId{0};
    uint32_t l1ListIdNext{0};

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
};
}

#endif // ASCENDCT_GEMM_BLOCK_BLOCK_GEMM_PRELOAD