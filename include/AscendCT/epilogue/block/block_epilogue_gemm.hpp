/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #ifndef ASCENDCT_EPILOGUE_BLOCK_EPILOGUE_GEMM_HPP
 #define ASCENDCT_EPILOGUE_BLOCK_EPILOGUE_GEMM_HPP
 
 #include "AscendCT/AscendCT.hpp"
 #include "AscendCT/epilogue/dispatch_policy.hpp"
 #include "AscendCT/matmul_coord.hpp"
 #include "AscendCT/matrix_coord.hpp"
 #include "AscendCT/epilogue/tile/tile_copy.hpp"
 #include "AscendCT/gemm/helper.hpp"
 
 namespace AscendCT::epilogue::block {
 template<
     class CType_,
     class XType_, 
     class DType_, 
     class TileElemWiseEpilogueAdd_, 
     class TileElemWiseEpilogueMuls_, 
     class TileElemWiseCastC_,
     class TileElemWiseCastD_,
     class TileCopy_
 >
 class BlockEpilogue<
     EpilogueAtlasA2ElemWiseOneSource,
     CType_,
     XType_, 
     DType_, 
     TileElemWiseEpilogueAdd_,
     TileElemWiseEpilogueMuls_,
     TileElemWiseCastC_,
     TileElemWiseCastD_,
     TileCopy_
 >{
 public:
     // Type aliases
     using DispatchPolicy = EpilogueAtlasA2ElemWiseOneSource;
     using ArchTag = typename DispatchPolicy::ArchTag;
     using ElementC = typename CType_::Element;
     using LayoutC = typename CType_::Layout;
     using ElementX = typename XType_::Element;
     using LayoutX = typename XType_::Layout;
     using ElementD = typename DType_::Element;
     using LayoutD = typename DType_::Layout;
     using TileElemWiseEpilogueAdd = TileElemWiseEpilogueAdd_;
     using TileElemWiseEpilogueMuls = TileElemWiseEpilogueMuls_;
     using TileElemWiseCastC = TileElemWiseCastC_;
     using TileElemWiseCastD = TileElemWiseCastD_;
     using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
     using CopyGmToUbX = typename TileCopy_::CopyGmToUbX;
     using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;
     
     static constexpr uint32_t STAGES = 2;
     const uint32_t UBSize = ArchTag::UB_SIZE;
     // static constexpr bool RowOrColumn = std::is_same<LayoutC, AscendCT::layout::RowMajor>::value && std::is_same<LayoutX, AscendCT::layout::RowMajor>::value;
     static constexpr bool isNeedCast = std::is_same<ElementC, bfloat16_t>::value; 
     static constexpr uint32_t COMPUTE_LENGTH = TileElemWiseEpilogueAdd::COMPUTE_LENGTH;
     static constexpr uint32_t OPERANDS_NUM = DispatchPolicy::OPERANDS_NUM;
 
     using ElementCompute = typename AscendCT::gemm::helper::ElementAccumulatorSelector<ElementC, ElementD>::ElementAccumulator;
     using ElementScalar = ElementCompute;
 
     // Check if ArchTag is matched
     static_assert(std::is_same_v<typename TileElemWiseEpilogueAdd::ArchTag, ArchTag>, "Tile epilogue's ArchTag mismatch");
     static_assert(std::is_same_v<typename TileElemWiseEpilogueMuls::ArchTag, ArchTag>, "Tile epilogue's ArchTag mismatch");
     // Check if compute length is valid
     static_assert(COMPUTE_LENGTH * OPERANDS_NUM * sizeof(ElementCompute) <= ArchTag::UB_SIZE, "UB out of bounds");
 
     ASCENDCT_DEVICE
     BlockEpilogue(arch::Resource<ArchTag> &resource, MatmulCoord blockShape_, uint32_t ubByteStart = 0) : blockShape(blockShape_){
         uint32_t maxMPerBlock = blockShape.m();
         uint32_t maxNPerBlock = blockShape.n();
         uint32_t tileSize = maxMPerBlock * maxNPerBlock / STAGES;
         uint32_t ubCSize = tileSize * sizeof(ElementC);
         uint32_t ubXSize = tileSize * sizeof(ElementX);
         uint32_t ubDSize = tileSize * sizeof(ElementD);
         uint32_t ubCSizeCast = tileSize * sizeof(ElementCompute);
         uint32_t ubDSizeCast = tileSize * sizeof(ElementCompute);
         ubCTensor = resource.ubBuf.template GetBufferByByte<ElementC>(ubByteStart);
         ubByteStart += ubCSize;
         ubXTensor = resource.ubBuf.template GetBufferByByte<ElementX>(ubByteStart); 
         ubByteStart += ubXSize;
         ubDTensor = resource.ubBuf.template GetBufferByByte<ElementD>(ubByteStart);
         ubByteStart += ubDSize;
         if constexpr (isNeedCast){
             ubCTensorCast = resource.ubBuf.template GetBufferByByte<ElementCompute>(ubByteStart);
             ubByteStart += ubCSizeCast;
             ubDTensorCast = resource.ubBuf.template GetBufferByByte<ElementCompute>(ubByteStart);;
             ubByteStart += ubDSizeCast;
         }
         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
     }
 
     ASCENDCT_DEVICE
     ~BlockEpilogue(){
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
     }
 
     ASCENDCT_DEVICE
     void operator()(
         ElementScalar alpha, ElementScalar beta, 
         AscendC::GlobalTensor<ElementC> gmBlockC, LayoutC layoutC, 
         AscendC::GlobalTensor<ElementD> gmBlockD, LayoutD layoutD, 
         AscendC::GlobalTensor<ElementX> gmBlockX, LayoutX layoutX, 
         MatmulCoord actualShape
     ){ 
         uint32_t MActual = actualShape.m();
         uint32_t NActual = actualShape.n(); 
         uint32_t maxMPerBlock = blockShape.m() / STAGES; 
         uint32_t maxNPerBlock = blockShape.n(); 
         uint32_t aivIndex = AscendC::GetSubBlockIdx(); 
         uint32_t MActualAIV0 = (MActual < maxMPerBlock) ? MActual : maxMPerBlock;
         uint32_t MActualAIV1 = (MActual < maxMPerBlock) ? 0 : (MActual - maxMPerBlock);
         uint32_t MUbActual = aivIndex == 1 ? MActualAIV1 : MActualAIV0;
         uint32_t NUbActual = NActual;
         LayoutC layoutInUb{maxMPerBlock, maxNPerBlock};
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
         auto layoutTileC = layoutC.GetTileLayout(MakeCoord(MUbActual, NUbActual));
         auto layoutCInUb = layoutInUb.GetTileLayout(MakeCoord(MUbActual, NUbActual));
         MatrixCoord gmTileCOffset{aivIndex * maxMPerBlock, 0};
         auto gmTileC = gmBlockC[layoutC.GetOffset(gmTileCOffset)];
         copyGmToUbC(ubCTensor, gmTileC, layoutCInUb, layoutTileC);
         AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
         AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
         if constexpr (isNeedCast){
             tileElemWiseCastC(ubCTensorCast, ubCTensor);
             AscendC::PipeBarrier<PIPE_V>();
             tileElemWiseEpilogueMuls(ubCTensorCast,ubCTensorCast, (ElementCompute)beta);
         }else{
             tileElemWiseEpilogueMuls(ubCTensor,ubCTensor, (ElementC)beta);
         }
         AscendC::PipeBarrier<PIPE_V>();
         AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
         auto layoutTileX = layoutX.GetTileLayout(MakeCoord(MUbActual, NUbActual));
         auto layoutXInUb = layoutInUb.GetTileLayout(MakeCoord(MUbActual, NUbActual));
         MatrixCoord gmTileXOffset{aivIndex * maxMPerBlock, 0}; 
         auto gmTileX = gmBlockX[layoutX.GetOffset(gmTileXOffset)];
         copyGmToUbX(ubXTensor, gmTileX, layoutXInUb, layoutTileX);
         AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
         AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
         tileElemWiseEpilogueMuls(ubXTensor, ubXTensor, (ElementX)alpha);
         AscendC::PipeBarrier<PIPE_V>();
         if constexpr (isNeedCast){
             tileElemWiseEpilogueAdd(ubDTensorCast,ubXTensor,ubCTensorCast);
         }else{
             tileElemWiseEpilogueAdd(ubDTensor,ubXTensor,ubCTensor);
         }
         AscendC::PipeBarrier<PIPE_V>();
         if constexpr (isNeedCast){
             tileElemWiseCastD(ubDTensor, ubDTensorCast);
         }
         AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
         AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
         auto layoutDInGm = layoutD.GetTileLayout(MakeCoord(MUbActual, NUbActual));
         auto layoutTileD = layoutInUb.GetTileLayout(MakeCoord(MUbActual, NUbActual));
         MatrixCoord gmTileDOffset{aivIndex * maxMPerBlock, 0}; 
         auto gmTileD = gmBlockD[layoutD.GetOffset(gmTileDOffset)];
         copyUbToGmD(gmTileD, ubDTensor, layoutDInGm, layoutTileD);
         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
         AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
     }
 private:
     MatmulCoord blockShape;
 
     AscendC::LocalTensor<ElementC> ubCTensor;
     AscendC::LocalTensor<ElementX> ubXTensor;
     AscendC::LocalTensor<ElementD> ubDTensor;
     AscendC::LocalTensor<ElementCompute> ubCTensorCast;
     AscendC::LocalTensor<ElementCompute> ubDTensorCast;
 
     CopyGmToUbC copyGmToUbC;
     CopyGmToUbX copyGmToUbX;
     CopyUbToGmD copyUbToGmD;
 
     TileElemWiseEpilogueAdd tileElemWiseEpilogueAdd;
     TileElemWiseEpilogueMuls tileElemWiseEpilogueMuls;
     TileElemWiseCastC tileElemWiseCastC;
     TileElemWiseCastD tileElemWiseCastD;
 };
 }
 
 #endif // ASCENDCT_EPILOGUE_BLOCK_EPILOGUE_GEMM_HPP