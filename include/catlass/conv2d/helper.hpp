/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV2D_HELPER_HPP
#define CATLASS_CONV2D_HELPER_HPP

#include "catlass/catlass.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/conv2d/conv2d_type.hpp"

namespace Catlass::Conv2d::helper {

template<class Element, class Layout>
struct L1AlignHelper {
    static_assert(DEPENDENT_FALSE<Element>, "Unsupported align helper, can not find the specialization.");
};

template<class Element>
struct L1AlignHelper<Element, layout::Fmap> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t HOWO_ALIGNED = C0_NUM_PER_FRACTAL;
};

template<class Element>
struct L1AlignHelper<Element, layout::Filter> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t COUT_ALIGNED = C0_NUM_PER_FRACTAL;
};

template<class ElementFmap, class ElementFilter>
struct ElementAccumulatorSelector {
    static_assert(DEPENDENT_FALSE<ElementFmap>,
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
struct ElementAccumulatorSelector<int8_t, int8_t> {
    using ElementAccumulator = int32_t;
};

template<>
struct ElementAccumulatorSelector<bfloat16_t, bfloat16_t> {
    using ElementAccumulator = float;
};

template<class GmFmapType>
struct L1ATypeSelector {
    static_assert(DEPENDENT_FALSE<GmFmapType>,
        "Unsupported layout selector, can not find the specialization.");
};

template<class Element>
struct L1ATypeSelector<Conv2d::Conv2dType<Element, layout::Fmap>> {
    using L1AType = Conv2d::Conv2dType<Element, layout::Fmap, AscendC::TPosition::A1>;
};

template<class GmFilterType>
struct L1BTypeSelector {
    static_assert(DEPENDENT_FALSE<GmFilterType>,
        "Unsupported layout selector, can not find the specialization.");
};

template<class Element>
struct L1BTypeSelector<Conv2d::Conv2dType<Element, layout::Filter>> {
    using L1BType = Conv2d::Conv2dType<Element, layout::Filter, AscendC::TPosition::A1>;
};

///////////////////////////////////////
} // namespace Catlass::Gemm::helper

#endif // CATLASS_CONV2D_HELPER_HPP
