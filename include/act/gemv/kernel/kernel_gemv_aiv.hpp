/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #ifndef ACT_GEMV_KERNLE_GEMV_AIV_HPP
 #define ACT_GEMV_KERNLE_GEMV_AIV_HPP
 #include "act/act.hpp"
 #include "act/arch/resource.hpp"
 #include "act/coord.hpp"
 #include "act/gemv_coord.hpp"

 namespace Act::Gemv::Kernel {
 
 // Template for Gemv kernel. Compute Y = αA * x + βY
 template <
     class BlockGemv_,
     class BlockEpilogue_
 >
 class KernelGemv {
 public:
     using BlockGemv = BlockGemv_;
     using ArchTag = typename BlockGemv::ArchTag;
     using UBTileShape = typename BlockGemv::UBTileShape;
     using ElementA = typename BlockGemv::ElementA;
     using LayoutA = typename BlockGemv::LayoutA;
     using ElementX = typename BlockGemv::ElementX;
     using LayoutX = typename BlockGemv::LayoutX;
     using ElementY = typename BlockGemv::ElementY;
     using LayoutY = typename BlockGemv::LayoutY;
     using ElementAccumulator = typename BlockGemv::ElementAccumulator;
 
 
     /// Parameters structure
     struct Params {
         // Data members
         GemvCoord problemShape;
         GM_ADDR ptrA;
         LayoutA layoutA;
         GM_ADDR ptrX;
         LayoutX layoutX;
         GM_ADDR ptrY;
         LayoutY layoutY;
         GM_ADDR ptrY_read;
         float alpha;
         float beta;
         uint32_t SPLIT;
 
         // Methods
         ACT_HOST_DEVICE
         Params() {}
 
         ACT_HOST_DEVICE
         Params(GemvCoord const &problemShape_,  GM_ADDR ptrA_, LayoutA layoutA_,  GM_ADDR ptrX_,LayoutX layoutX_,
            GM_ADDR ptrY_,LayoutY layoutY_,GM_ADDR ptrY_read_,float alpha_,float beta_,uint32_t SPLIT_)
             : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrX(ptrX_),layoutX(layoutX_),
               ptrY(ptrY_),layoutY(layoutY_),ptrY_read(ptrY_read_),alpha(alpha_),beta(beta_),SPLIT(SPLIT_) {}
     };

    
     // Methods
     ACT_DEVICE
     KernelGemv(){}

     template <int32_t CORE_TYPE = g_coreType>
     ACT_DEVICE
     void operator()(Params const &params){
     };
 
     /// Executes one Matmul
     template <>
     ACT_DEVICE
     void operator()<AscendC::AIC>(Params const &params) {
     }
 
     template <>
     ACT_DEVICE
     void operator()<AscendC::AIV>(Params const &params) {
        AscendC::SetAtomicNone();
        Arch::Resource<ArchTag> resource;
        BlockGemv blockGemv(resource);
         uint32_t align = BYTE_PER_C0 / sizeof(ElementA);
         uint32_t maxmPerBlock_round = RoundUp(UBTileShape::M,align);
         uint32_t maxnPerBlock_round = RoundUp(UBTileShape::N,align);

        //add split k
         uint32_t N_Split = RoundDown(params.problemShape.n(),params.SPLIT)/params.SPLIT;
         uint32_t Mloopnum = CeilDiv(params.problemShape.m(),maxmPerBlock_round);
         int32_t loopnum;
        float Realbeta= params.beta;
         if constexpr (std::is_same_v<LayoutA, Act::layout::ColumnMajor>){
            loopnum = Mloopnum * params.SPLIT;
            Realbeta = params.beta - 1.0f;
         }else{
            loopnum = Mloopnum;
         }
         
         uint32_t offset_matrix;
         uint32_t offset_vector_out;
         uint32_t offset_vector_in = 0;

         // Represent the full gm
         AscendC::GlobalTensor<ElementA> gmA;
         gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
         AscendC::GlobalTensor<ElementX> gmX;
         gmX.SetGlobalBuffer((__gm__ ElementX *)params.ptrX);
         AscendC::GlobalTensor<ElementY> gmY;
         gmY.SetGlobalBuffer((__gm__ ElementY *)params.ptrY);
         AscendC::GlobalTensor<ElementY> gmY_read;
         gmY_read.SetGlobalBuffer((__gm__ ElementY *)params.ptrY_read);
         uint32_t aiv_num = AscendC::GetBlockNum()*AscendC::GetTaskRation();
         for(uint32_t loop_id = 0;loop_id < loopnum;loop_id++){
            uint32_t aiv_id = AscendC::GetBlockIdx();   
            if(loop_id % aiv_num != aiv_id)continue;
            uint32_t m_actual = ((int32_t)loop_id > (int32_t)(loopnum - params.SPLIT - 1) ) ? params.problemShape.m() - ((loop_id/params.SPLIT) * maxmPerBlock_round) : maxmPerBlock_round;
            uint32_t n_actual = params.problemShape.n();

            if constexpr (std::is_same_v<LayoutA, Act::layout::ColumnMajor>) {
                offset_matrix = (loop_id % params.SPLIT) * N_Split*params.problemShape.m()+(loop_id/params.SPLIT) * maxmPerBlock_round;
                offset_vector_out = (loop_id/params.SPLIT) * maxmPerBlock_round;
                offset_vector_in = (loop_id % params.SPLIT) * N_Split; 
                
                if((loop_id%params.SPLIT) == params.SPLIT - 1){
                    n_actual = params.problemShape.n() - N_Split * (params.SPLIT - 1);
                }
                else{
                    n_actual = N_Split;
                }
            } else {
                offset_matrix = loop_id * maxmPerBlock_round * params.problemShape.n();
                offset_vector_out = loop_id * maxmPerBlock_round;
            }
            GemvCoord actualBlockShape = GemvCoord{m_actual,n_actual};
            
            float realbeta = (loop_id % params.SPLIT == 0) ? Realbeta:0.0f;

            blockGemv(gmA[offset_matrix], params.layoutA,
                gmX[offset_vector_in], params.layoutX,
                gmY[offset_vector_out], params.layoutY,
                gmY_read[offset_vector_out],
                actualBlockShape,
                params.alpha,
                realbeta
            );

        }
     }
 };
 
 } 
 
 #endif // ACT_GEMV_KERNLE_GEMV_AIV_HPP