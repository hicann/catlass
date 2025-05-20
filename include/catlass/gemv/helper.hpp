/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #ifndef CATLASS_GEMV_HELPER_HPP
 #define CATLASS_GEMV_HELPER_HPP
 
 #include "catlass/catlass.hpp"
 #include "catlass/layout/layout.hpp"
 #include "catlass/gemm/gemm_type.hpp"
 namespace Catlass::Gemv::helper {
 
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
 
 template<class GmAType>
 struct IsAtoaddSelector {
     static_assert(DEPENDENT_FALSE<GmAType>,
         "Unsupported layout selector, can not find the specialization.");
 };
 
 template<class Element>
 struct IsAtoaddSelector<Gemm::GemmType<Element, layout::RowMajor>> {
    static constexpr bool value = false;
 };

 template<class Element>
 struct IsAtoaddSelector<Gemm::GemmType<Element, layout::ColumnMajor>> {
    static constexpr bool value = true;
 };

 template <class Element, class Layout>
 struct L1AlignHelper
 {
     static_assert(DEPENDENT_FALSE<Element>, "Unsupported align helper, can not find the specialization.");
 };

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

     static constexpr uint32_t getNAligned()
     {
         if constexpr (std::is_same<Element, int8_t>::value)
         {
             return ELE_NUM_PER_C0 / sizeof(Element); 
         }
         else
         {
             return C0_NUM_PER_FRACTAL; 
         }
     }

     static constexpr uint32_t getMAligned()
     {
         if constexpr (std::is_same<Element, int8_t>::value)
         {
             return ELE_NUM_PER_C0 / sizeof(Element); 
         }
         else
         {
             return C0_NUM_PER_FRACTAL; 
         }
     }

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

 template<class ElementA, class ElementB>
 struct ElementAccumulatorSelector {
     static_assert(DEPENDENT_FALSE<ElementA>,
         "Unsupported element accumulator selector, can not find the specialization.");
 };

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

 template <>
 struct ElementAccumulatorSelector<int32_t, int32_t>
 {
     using ElementAccumulator = int32_t;
 };

 template<>
 struct ElementAccumulatorSelector<bfloat16_t, bfloat16_t> {
     using ElementAccumulator = float;
 };
 
 } // namespace Catlass::Gemv::helper
 
 #endif // CATLASS_GEMV_HELPER_HPP
 