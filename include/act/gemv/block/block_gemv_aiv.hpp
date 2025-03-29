/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #ifndef ACT_GEMV_BLOCK_BLOCK_GEMV_AIV_HPP
 #define ACT_GEMV_BLOCK_BLOCK_GEMV_AIV_HPP
 
 #include "act/act.hpp"
 #include "act/arch/resource.hpp"
 #include "act/coord.hpp"
 #include "act/gemv_coord.hpp"
 #include "act/gemv/helper.hpp"
 #include "act/layout/layout.hpp"
 #include "act/detail/alignment.hpp"
 #include "act/gemm/dispatch_policy.hpp"

 namespace Act::Gemv::Block {
 template <
     bool ENABLE_UNIT_FLAG_,
     class UBTileShape_,
     class AType_,
     class XType_,
     class YType_,
     class BiasType_,
     class TileCopy_,
     class TileVmad_,
     class TileVmuls_
 >
 struct BlockGemv <
     Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>,
     UBTileShape_,
     AType_,
     XType_,
     YType_,
     BiasType_,
     TileCopy_,
     TileVmad_,
     TileVmuls_
 > {
 public:
     // Type Aliases
     using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG_>;
     using ArchTag = typename DispatchPolicy::ArchTag;
     using UBTileShape = UBTileShape_;
     using ElementA = typename AType_::Element;
     using LayoutA = typename AType_::Layout;
     using ElementX = typename XType_::Element;
     using LayoutX = typename XType_::Layout;
     using ElementY = typename YType_::Element;
     using LayoutY = typename YType_::Layout;
     using TileVmad = TileVmad_;
     using TileVmuls = TileVmuls_;
     using VecCopyGmToUb = typename TileCopy_::VecCopyGmToUb;
     using VecCopyUbToGm = typename TileCopy_::VecCopyUbToGm;
     using MatrixCopyGmToUb = typename TileCopy_::MatrixCopyGmToUb;
     using ElementAccumulator =
         typename Gemv::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;
    
     using UBAlignHelper = Gemv::helper::UBAlignHelper<ElementA, LayoutA>;
     using TensorCoord = layout::VectorLayout::TensorCoord;
     static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
     static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
     static constexpr uint32_t Abuf_SIZE_ = 128 * 1024; 
     static constexpr uint32_t Xbuf_SIZE_ = 16 * 1024;  
     static constexpr uint32_t Ybuf_SIZE_ = 16 * 1024;  
     static constexpr uint32_t workspace_SIZE_ = 32 * 1024; 
     ACT_DEVICE
     BlockGemv(){}
     /// Construct
     ACT_DEVICE
     BlockGemv(Arch::Resource<ArchTag> &resource, uint32_t UBufAddrStart = 0)
     {
        
        
         uint32_t UbAOffset = UBufAddrStart;
         uint32_t UbXOffset = UBufAddrStart + Abuf_SIZE_;
         uint32_t UbYOffset = UBufAddrStart + Abuf_SIZE_+Xbuf_SIZE_;
         uint32_t UbWOffset = UBufAddrStart + Abuf_SIZE_+Xbuf_SIZE_+Ybuf_SIZE_;
         // Init buffers
         for (uint32_t i = 0; i < STAGES; i++) {
             // Assign L1/L0A/L0B space for each stages
             UbATensorList[i] = resource.ubBuf.template GetBufferByByte<ElementA>(UbAOffset+i * (Abuf_SIZE_/2));
             UbXTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementX>(UbXOffset+i * (Xbuf_SIZE_/2));
             UbYTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementY>(UbYOffset+i * (Ybuf_SIZE_/2));
             UbWTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementAccumulator>(UbWOffset+i * (workspace_SIZE_/2));
 
             // Assign event ID for each stages
             UbInAEventList[i] = i;
             UbInXEventList[i] = i + STAGES;
             UbOutEventList[i] = i;
             
             // The event id that needs to be set before the loop
             AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
             AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
             AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[i]);
         }
     }
 
     /// Destructor
     ACT_DEVICE
     ~BlockGemv()
     {
         for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInAEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(UbInXEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(UbOutEventList[i]);
         }
         
     }
 
     ACT_DEVICE
     void operator()(
         AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
         AscendC::GlobalTensor<ElementX> const &gmX, LayoutX const &layoutX,
         AscendC::GlobalTensor<ElementY> const &gmY, LayoutY const &layoutY,
         AscendC::GlobalTensor<ElementY> const &gmY_read,
         GemvCoord const &actualShape,
         float alpha,
         float beta
        )
     {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)(UbOutEventList[UbOutListId]));
        vecCopyGmToUb(UbYTensorList[UbOutListId], gmY_read,actualShape.m());
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbOutEventList[UbOutListId]));  
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbOutEventList[UbOutListId]));
        tileVmuls(
            UbYTensorList[UbOutListId], 
            UbYTensorList[UbOutListId], 
            (ElementY)beta,
            actualShape.m()
        );
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventList[UbOutListId]));  
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbOutEventList[UbOutListId]));
        
        TileMRound = RoundUp(UBTileShape::M,UBAlignHelper::ALIGN);
        TileNRound = RoundUp(UBTileShape::N,UBAlignHelper::ALIGN);
        strideA = layoutA.stride(1) * TileNRound;
        m_actual = (actualShape.m() < TileMRound) ? actualShape.m():TileMRound;
        n_actual = (actualShape.n() < TileNRound) ? actualShape.n():TileNRound;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListId]));
        vecCopyGmToUb(UbXTensorList[UbInListId], gmX,n_actual);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInListId]));
        
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
        auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
        auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
        matrixCopyGmToUb(UbATensorList[UbInListId], gmA, layoutAInUb, layoutTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId]));
         // main loop
        uint32_t Nloop = CeilDiv(actualShape.n(),TileNRound);
        for(uint32_t LoopIdx = 0;LoopIdx < Nloop; LoopIdx++){
            m_actual = (actualShape.m() < TileMRound) ? actualShape.m():TileMRound;
            n_actual = (LoopIdx == Nloop - 1) ? (actualShape.n() - LoopIdx * TileNRound) : TileNRound;
            y_actual = m_actual;
            x_actual = n_actual;

            uint32_t UbInListIdNext = (UbInListId + 1 < STAGES) ? (UbInListId + 1) : 0;
            if(LoopIdx < Nloop - 1) {
                uint32_t LoopIdxNext = LoopIdx + 1;
                uint32_t m_actual_next = m_actual;
                uint32_t n_actual_next = (LoopIdxNext == Nloop - 1) ? (actualShape.n() - LoopIdxNext * TileNRound) : TileNRound;
                uint32_t y_actual_next = m_actual_next;
                uint32_t x_actual_next = n_actual_next;
                // Get L1 tensor for next stage
                auto matrixTensor = UbATensorList[UbInListIdNext];
                auto vecTensor = UbXTensorList[UbInListIdNext];

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListIdNext]));
                vecCopyGmToUb(vecTensor, gmX[LoopIdxNext * TileNRound],x_actual_next);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInListIdNext]));
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListIdNext]));
                auto layoutAInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(m_actual_next, n_actual_next));
                matrixCopyGmToUb(matrixTensor, gmA[LoopIdxNext * strideA], layoutAInUb, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListIdNext]));
                 
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInXEventList[UbInListId]));
            tileVmuls(
                UbXTensorList[UbInListId], 
                UbXTensorList[UbInListId], 
                (ElementA)alpha,
                x_actual
            );
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((event_t)(UbInAEventList[UbInListId])); 
            auto layoutComputeInUb = layoutA.GetTileLayout(MakeCoord(TileMRound, TileNRound));
            auto layoutTileCompute = layoutA.GetTileLayout(MakeCoord(m_actual, n_actual));
            tileVmad(
                UbYTensorList[UbOutListId],
                UbXTensorList[UbInListId],
                UbATensorList[UbInListId],
                UbWTensorList[UbInListId],
                layoutComputeInUb,
                layoutTileCompute
            );
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInAEventList[UbInListId]));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(UbInXEventList[UbInListId]));
            UbInListId = UbInListIdNext;
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((event_t)(UbOutEventList[UbOutListId]));
        auto layoutDstY = layoutY.GetTileLayout(TensorCoord(y_actual));
        auto layoutComputeInUb = layoutY.GetTileLayout(TensorCoord(y_actual));
        vecCopyUbToGm(gmY, UbYTensorList[UbOutListId],layoutDstY, layoutComputeInUb); 
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)(UbOutEventList[UbOutListId]));
        UbOutListId = (UbOutListId + 1 < STAGES) ? (UbOutListId + 1) : 0;
     }
 
 protected:
     // Multi-stage tensors list
     AscendC::LocalTensor<ElementA> UbATensorList[STAGES];
     AscendC::LocalTensor<ElementX> UbXTensorList[STAGES];
     AscendC::LocalTensor<ElementY> UbYTensorList[STAGES];
     AscendC::LocalTensor<ElementAccumulator> UbWTensorList[STAGES];
 
     // Multi-stage event id list
     int32_t UbInAEventList[STAGES];
     int32_t UbInXEventList[STAGES];
     int32_t UbOutEventList[STAGES];
 
     // The id of current stage
     uint32_t UbOutListId{0};
     uint32_t UbInListId{0};

     uint32_t m_actual, n_actual,x_actual, y_actual;
     uint32_t TileMRound,TileNRound;
     uint32_t strideA;

     TileVmad tileVmad;
     TileVmuls tileVmuls;
     MatrixCopyGmToUb matrixCopyGmToUb;
     VecCopyGmToUb vecCopyGmToUb;
     VecCopyUbToGm vecCopyUbToGm;
     
 };
 
 } // namespace Act::Gemv::Block
 
 #endif // ACT_GEMV_BLOCK_BLOCK_GEMV_AIV_HPP
 