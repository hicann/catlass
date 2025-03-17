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
 
 template<>
 struct ElementAccumulatorSelector<bfloat16_t, bfloat16_t> {
     using ElementAccumulator = float;
 };
 
 } // namespace acot::matmul::helper
 
 #endif // ACOT_MATMUL_HELPER_HPP
 