/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #ifndef ACOT_GEMV_KERNEL_GEMV_HPP
 #define ACOT_GEMV_KERNEL_GEMV_HPP
 
 #include "acot/acot.hpp"
 #include "acot/arch/resource.hpp"
 #include "acot/coord.hpp"
 #include "acot/matmul_coord.hpp"
 #include "acot/matrix_coord.hpp"
 
 namespace acot::gemv::kernel {
 
 // Template for Matmul kernel. Compute C = A * B
 template <
     class BlockGemv_,
     class BlockEpilogue_,
     class TileScheduler_
 >
 class KernelGemv {
 public:
     using BlockGemv = BlockGemv_;
     using ArchTag = typename BlockGemv::ArchTag;
     using UBTileShape = typename BlockGemv::UBTileShape;
     using ElementA = typename BlockGemv::ElementA;
     using LayoutA = typename BlockGemv::LayoutA;
     using ElementX = typename BlockGemv::ElementX;
     using ElementY = typename BlockGemv::ElementY;
     using ElementAccumulator = typename BlockGemv::ElementAccumulator;
 
     using TileScheduler = TileScheduler_;
 
     /// Parameters structure
     struct Params {
         // Data members
         MatmulCoord problemShape;
         GM_ADDR ptrA;
         LayoutA layoutA;
         GM_ADDR ptrX;
         GM_ADDR ptrY;
         GM_ADDR ptrY_read;
         float alpha;
         float beta;
 
         // Methods
         ACOT_DEVICE
         Params() {}
 
         ACOT_DEVICE
         Params(MatmulCoord const &problemShape_,  GM_ADDR ptrA_, LayoutA layoutA_,  GM_ADDR ptrX_,
            GM_ADDR ptrY_,GM_ADDR ptrY_read_,float alpha_,float beta_)
             : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrX(ptrX_),
               ptrY(ptrY_),ptrY_read(ptrY_read_),alpha(alpha_),beta(beta_) {}
     };

     
     // Methods
    //  ACOT_DEVICE
    //  KernelGemv(){
    //     blockGemv = BlockGemv(resource);
    //  }
    //  KernelGemv():resource(), blockGemv(resource) {
    //  }

     template <int32_t CORE_TYPE = g_coreType>
     ACOT_DEVICE
     void operator()(Params const &params){
     };
 
     /// Executes one Matmul
     template <>
     ACOT_DEVICE
     void operator()<AscendC::AIC>(Params const &params) {
        
     }
 
     template <>
     ACOT_DEVICE
     void operator()<AscendC::AIV>(Params const &params) {
        // TileScheduler matmulTileScheduler(params.problemShape, MakeCoord(UBTileShape::M, UBTileShape::N));
        arch::Resource<ArchTag> resource;
        BlockGemv blockGemv(resource);
         uint32_t align = BYTE_PER_C0 / sizeof(ElementA);
         uint32_t maxmPerBlock_round = RoundUp(UBTileShape::M,align);
         uint32_t maxnPerBlock_round = RoundUp(UBTileShape::N,align);
         uint32_t loopnum = CeilDiv(params.problemShape.m(),maxmPerBlock_round);

         uint32_t element_stride_matrix = sizeof(ElementA) / sizeof(uint8_t);
         uint32_t element_stride_vector_out = sizeof(ElementY) / sizeof(uint8_t);
         uint32_t offset_matrix;
         uint32_t offset_vector_out;

         // Represent the full gm
         AscendC::GlobalTensor<ElementA> gmA;
         gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
         AscendC::GlobalTensor<ElementX> gmX;
         gmX.SetGlobalBuffer((__gm__ ElementX *)params.ptrX);
         AscendC::GlobalTensor<ElementY> gmY;
         gmY.SetGlobalBuffer((__gm__ ElementY *)params.ptrY);
         AscendC::GlobalTensor<ElementY> gmY_read;
         gmY_read.SetGlobalBuffer((__gm__ ElementY *)params.ptrY_read);
        //  for(uint32_t i = 0;i < 4;i++){
        //     AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((event_t)(i));
        // }
        // for(uint32_t i = 0;i < 2;i++){
        //     AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)(i));
        // }
         for(uint32_t loop_id = 0;loop_id < loopnum;loop_id++){
            uint32_t aiv_id = AscendC::GetBlockIdx()/2+AscendC::GetSubBlockIdx();
            uint32_t aiv_num = AscendC::GetBlockNum()/2 * AscendC::GetSubBlockNum();
            // uint32_t aiv_id = AscendC::GetBlockIdx();
            // uint32_t aiv_num = AscendC::GetBlockNum() * AscendC::GetTaskRation();
            if(loop_id % aiv_num != aiv_id)continue;

            if constexpr (std::is_same_v<LayoutA, acot::layout::ColumnMajor>) {
                offset_matrix = loop_id * maxmPerBlock_round;
                offset_vector_out = loop_id * maxmPerBlock_round;
            } else {
                offset_matrix = loop_id * maxmPerBlock_round * params.problemShape.n();
                offset_vector_out = loop_id * maxmPerBlock_round;
            }
            uint32_t m_actual = (loop_id == loopnum - 1) ? params.problemShape.m() - (loop_id * maxmPerBlock_round) : maxmPerBlock_round;
            uint32_t n_actual = params.problemShape.n();
            
            MatmulCoord actualBlockShape = MatmulCoord{m_actual,n_actual,1};
            // Compute block-scoped matrix multiply-add
            blockGemv(gmA[offset_matrix], params.layoutA,
                gmX, 
                gmY[offset_vector_out], 
                gmY_read[offset_vector_out],
                actualBlockShape,
                params.alpha,
                params.beta
            );
        }
        // for(uint32_t i = 0;i < 4;i++){
        //     AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((event_t)(i));
        // }
        // for(uint32_t i = 0;i < 2;i++){
        //     AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((event_t)(i));
        // }
     }
     private:
     
 };
 
 } // namespace acot::matmul::kernel
 
 #endif // ACOT_MATMUL_KERNEL_MATMUL_HPP