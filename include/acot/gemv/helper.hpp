/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #ifndef ACOT_GEMV_HELPER_HPP
 #define ACOT_GEMV_HELPER_HPP
 
 #include "acot/acot.hpp"
 #include "acot/layout/layout.hpp"
 
 namespace acot::gemv::helper {
 
 template<class Element, class Layout>
 struct UBAlignHelper {
     static_assert(DEPENDENT_FALSE<Element>, "Unsupported align helper, can not find the specialization.");
 };
 
 template<class Element>
 struct UBAlignHelper<Element, layout::RowMajor> {
     static constexpr uint32_t ALIGN = BYTE_PER_C0 / sizeof(Element);
 };
 
 template<class Element>
 struct UBAlignHelper<Element, layout::ColumnMajor> {
     static constexpr uint32_t ALIGN = BYTE_PER_C0 / sizeof(Element);
 };
 
 
 template<class ElementA, class ElementB>
 struct ElementAccumulatorSelector {
     static_assert(DEPENDENT_FALSE<ElementA>,
         "Unsupported element accumulator selector, can not find the specialization.");
 };


 // aic
 template <class Element, class Layout>
 struct L1AlignHelper
 {
     static_assert(DEPENDENT_FALSE<Element>, "Unsupported align helper, can not find the specialization.");
 };

 // 下列的各项行优先、列优先的对齐维度，需要在具体实现单核算子的时候重新review
 template <class Element>
 struct L1AlignHelper<Element, layout::RowMajor>
 {
     static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
     static constexpr uint32_t M_ALIGNED = C0_NUM_PER_FRACTAL;
     static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
     static constexpr uint32_t N_ALIGNED = ELE_NUM_PER_C0;
 };

 template <class Element>
 struct L1AlignHelper<Element, layout::ColumnMajor>
 {
     static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

     // 使用函数来根据类型确定 N_ALIGNED
     static constexpr uint32_t getNAligned()
     {
         if constexpr (std::is_same<Element, int8_t>::value)
         {
             return ELE_NUM_PER_C0 / sizeof(Element); // 对于 int8 类型，对齐32
         }
         else
         {
             return C0_NUM_PER_FRACTAL; // 对于其他类型，对齐16
         }
     }

     static constexpr uint32_t getMAligned()
     {
         if constexpr (std::is_same<Element, int8_t>::value)
         {
             return ELE_NUM_PER_C0 / sizeof(Element); // 对于 int8 类型，对齐32
         }
         else
         {
             return C0_NUM_PER_FRACTAL; // 对于其他类型，对齐16
         }
     }

     // 提供一个静态常量，调用 getNAligned 函数来初始化 N_ALIGNED
     static constexpr uint32_t N_ALIGNED = getNAligned();
     static constexpr uint32_t M_ALIGNED = getMAligned();
 };

 template <class Element>
 struct L1AlignHelper<Element, layout::zN>
 {
     static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
     static constexpr uint32_t M_ALIGNED = C0_NUM_PER_FRACTAL;
     static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
     static constexpr uint32_t N_ALIGNED = ELE_NUM_PER_C0;
 };

 template <class Element>
 struct L1AlignHelper<Element, layout::nZ>
 {
     static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
     static constexpr uint32_t M_ALIGNED = ELE_NUM_PER_C0;
     static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
     static constexpr uint32_t N_ALIGNED = C0_NUM_PER_FRACTAL;
 };

 // 模板特例化
 // 根据输入矩阵、向量类型，选择对应的L0C中的数据类型
 template<>
 struct ElementAccumulatorSelector<half, half> {
     using ElementAccumulator = float;
 };
 
 template<>
 struct ElementAccumulatorSelector<float, float> {
     using ElementAccumulator = float;
 };
 
 template<>
 struct ElementAccumulatorSelector<uint8_t, uint8_t> {
     using ElementAccumulator = int32_t;
 };

 template <>
 struct ElementAccumulatorSelector<int8_t, int8_t>
 {
     using ElementAccumulator = int32_t;
 };

 template<>
 struct ElementAccumulatorSelector<bfloat16_t, bfloat16_t> {
     using ElementAccumulator = float;
 };
 
 } // namespace acot::matmul::helper
 
 #endif // ACOT_MATMUL_HELPER_HPP
 