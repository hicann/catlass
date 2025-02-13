/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #ifndef ACOT_GEMV_TILE_TILE_VMAD_HPP
 #define ACOT_GEMV_TILE_TILE_VMAD_HPP
 
 #include "acot/acot.hpp"
 #include "acot/layout/layout.hpp"
 #include "acot/gemv_aiv/gemv_type.hpp"
 
 
namespace acot::gemv::tile {
 
template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType type for B matrix operand
    class XType,
    /// MatmulType type for C matrix operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
 
 struct TileVmad {
     static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileVmad, can not find the specialization.");
 };
 template <
    class ElementA,
    class ElementX,
    class ElementY
>
 /// Partial specialization for AtlasA2, RowMajor in and zN out.
 struct TileVmad<arch::AtlasA2, 
                gemv::GemvType<ElementA, layout::RowMajor>,
                gemv::GemvType<ElementX, layout::RowMajor>,
                gemv::GemvType<ElementY, layout::RowMajor>,
                void> {
    using ElementAccumulator =
        typename gemv::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;
 
     static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);      //32B,一个block的大小
 
     // Mehtods
 
     ACOT_DEVICE
     TileVmad() {};
 
     ACOT_DEVICE
     void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementX> srcTensor_v,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementAccumulator> temp,
          uint32_t m_actual, 
          uint32_t n_actual, 
          uint32_t m_round, 
          uint32_t n_round
    ) {
          uint32_t repeat_size = ELE_NUM_PER_C0 * 8; // 每个repeat处理8个block
          uint32_t mask = repeat_size;
          uint32_t repeat_num = n_actual / repeat_size;
          uint32_t remain = n_actual % repeat_size;

          AscendC::BinaryRepeatParams params;
          params.dstBlkStride = 1;
          params.src0BlkStride = 1;
          params.src1BlkStride = 1;
          params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
          params.src0RepStride = params.dstRepStride;
          params.src1RepStride = 0;

          for (uint32_t i = 0; i < repeat_num; i++) {
              uint32_t offset = i * repeat_size;
              if (i == 0) {
                  AscendC::Mul<ElementA, true>(
                      srcTensor_m,
                      srcTensor_m,
                      srcTensor_v,
                      mask,
                      m_actual,
                      params
                  );
              } else {
                  AscendC::MulAddDst<ElementA, ElementA, true>(
                      srcTensor_m,
                      srcTensor_m[offset],
                      srcTensor_v[offset],
                      mask,
                      m_actual,
                      params
                  );
              }
              AscendC::PipeBarrier<PIPE_V>();
          }

          if (remain > 0) {
              uint32_t offset = repeat_num * repeat_size;
              if (offset + remain > n_round) {
                  remain = n_round - offset;
              }
              uint64_t remain_mask = remain;
              if (repeat_num == 0) {
                  AscendC::Mul<ElementA, true>(
                      srcTensor_m,
                      srcTensor_m,
                      srcTensor_v,
                      remain_mask,
                      m_actual,
                      params
                  );
              } else {
                  AscendC::MulAddDst<ElementA, ElementA, true>(
                      srcTensor_m,
                      srcTensor_m[offset],
                      srcTensor_v[offset],
                      remain_mask,
                      m_actual,
                      params
                  );
              }
          }

          // 修正Reduce和Add操作
          uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size;
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::WholeReduceSum<ElementA, true>(
              srcTensor_m,
              srcTensor_m,
              reduce_mask,
              m_actual,
              1,
              1,
              RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0
          );

          uint64_t add_mask = (m_actual < repeat_size) ? m_actual : repeat_size;
          params.dstRepStride = 8;
          params.src0RepStride = 8;
          params.src1RepStride = 8;

          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Add<ElementA, true>(
              dstTensor,
              srcTensor_m,
              dstTensor,
              add_mask,
              CeilDiv(m_round, repeat_size), // 修正为m_round
              params
          );
    };
  };
 /// Partial specialization for AtlasA2, ColumnMajor in and nZ out.
 template <
    class ElementA,
    class ElementX,
    class ElementY
>
 /// Partial specialization for AtlasA2, RowMajor in and zN out.
 struct TileVmad<arch::AtlasA2, 
                gemv::GemvType<ElementA, layout::ColumnMajor>,
                gemv::GemvType<ElementX, layout::ColumnMajor>,
                gemv::GemvType<ElementY, layout::ColumnMajor>,
                void> {
    using ElementAccumulator =
        typename gemv::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;
 
     static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);      //32B,一个block的大小
 
     // Mehtods
 
     ACOT_DEVICE
     TileVmad() {};
 
     ACOT_DEVICE
     void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementX> srcTensor_v,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementAccumulator> temp,
          uint32_t m_actual, 
          uint32_t n_actual, 
          uint32_t m_round, 
          uint32_t n_round
    ) {
        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementAccumulator);
        uint32_t elem_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementA);
        uint32_t mask = temp_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual,temp_repeat_size);
        uint32_t temstride = (std::is_same_v<ElementA, ElementAccumulator>)?8:4;
        AscendC::UnaryRepeatParams params;
        if constexpr(std::is_same_v<ElementA, ElementAccumulator>){
          params.dstBlkStride = 1;
          params.srcBlkStride = 1;
          params.dstRepStride = 8;
          params.srcRepStride = 8;
          AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)(0));
          for(uint32_t i = 0;i < n_actual;i++){
            AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)(0));
            auto pix = srcTensor_v.GetValue(i);
            AscendC::SetFlag<AscendC::HardEvent::S_V>((event_t)(0));
            AscendC::WaitFlag<AscendC::HardEvent::S_V>((event_t)(0));
            AscendC::Axpy<ElementY,ElementA,true>(  //vaxpy
              dstTensor,
              srcTensor_m[i*m_round],
              pix,
              (uint64_t)mask,
              repeattimes,
              params
            );
            AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)(0));
          }
          AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)(0));
        }
        else{
          params.dstBlkStride = 1;
          params.srcBlkStride = 1;
          params.dstRepStride = 8;
          params.srcRepStride = 4;  //mix运算,后四个block不会算
          AscendC::Duplicate<ElementAccumulator>(  //VEC_DUP
            temp,
            (ElementAccumulator)0.0,
            temp_repeat_size,     //mask
            CeilDiv(m_round,temp_repeat_size),   //repeattimes
            1,   //blkstride
            8   //repstride
          );
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)(0));
          for(uint32_t i = 0;i < n_actual;i++){
            AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)(0));
            auto pix = srcTensor_v.GetValue(i);
            AscendC::SetFlag<AscendC::HardEvent::S_V>((event_t)(0));
            AscendC::WaitFlag<AscendC::HardEvent::S_V>((event_t)(0));
            AscendC::Axpy<ElementAccumulator,ElementA,true>(
              temp,
              srcTensor_m[i*m_round],
              pix,
              (uint64_t)mask,
              repeattimes,
              params
            );
            AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)(0));
          }
          AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)(0));
          AscendC::PipeBarrier<PIPE_V>();
          params.dstRepStride = 4;  //以数据类型大的为准，即以float为准
          params.srcRepStride = 8;  
          AscendC::Cast<ElementA,ElementAccumulator,true>(
            srcTensor_m,
            temp,
            AscendC::RoundMode::CAST_NONE,
            (uint64_t)mask,
            repeattimes,
            params
          );
          AscendC::BinaryRepeatParams addparams;
          addparams.dstBlkStride = 1;
          addparams.src0BlkStride = 1;
          addparams.src1BlkStride = 1;
          addparams.dstRepStride = 8;
          addparams.src0RepStride = 8;
          addparams.src1RepStride = 8;
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Add<ElementA,true>(
            dstTensor,
            srcTensor_m,
            dstTensor,
            (uint64_t)elem_repeat_size,
            CeilDiv(m_round,elem_repeat_size),
            addparams
          );
        }
    }
  };
}
 

 #endif // ACOT_MATMUL_TILE_COPY_GM_TO_L1_HPP
 