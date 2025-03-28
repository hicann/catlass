/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACT_GEMM_HELPER_HPP
#define ACT_GEMM_HELPER_HPP

#include "act/act.hpp"
#include "act/layout/layout.hpp"
#include "tla/layout.hpp"

namespace Act::Gemm::helper {

template<class Element, class Layout>
struct L1AlignHelper {
    static_assert(DEPENDENT_FALSE<Element>, "Unsupported align helper, can not find the specialization.");
};

template<class Element>
struct L1AlignHelper<Element, layout::RowMajor> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t M_ALIGNED = C0_NUM_PER_FRACTAL;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = ELE_NUM_PER_C0;
};

template<class Element>
struct L1AlignHelper<Element, layout::ColumnMajor> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t M_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = C0_NUM_PER_FRACTAL;
};

template<class Element>
struct L1AlignHelper<Element, layout::PaddingRowMajor> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t M_ALIGNED = C0_NUM_PER_FRACTAL;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = ELE_NUM_PER_C0;
};

template<class Element>
struct L1AlignHelper<Element, layout::PaddingColumnMajor> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t M_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = C0_NUM_PER_FRACTAL;
};

template<class Element>
struct L1AlignHelper<Element, layout::zN> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t M_ALIGNED = C0_NUM_PER_FRACTAL;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = ELE_NUM_PER_C0;
};

template<class Element>
struct L1AlignHelper<Element, layout::nZ> {
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
struct ElementAccumulatorSelector<int8_t, int8_t> {
    using ElementAccumulator = int32_t;
};

template<>
struct ElementAccumulatorSelector<bfloat16_t, bfloat16_t> {
    using ElementAccumulator = float;
};

template<class GmAType>
struct L1ATypeSelector {
    static_assert(DEPENDENT_FALSE<GmAType>,
        "Unsupported layout selector, can not find the specialization.");
};

template<class Element>
struct L1ATypeSelector<Gemm::GemmType<Element, layout::RowMajor>> {
    using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template<class Element>
struct L1ATypeSelector<Gemm::GemmType<Element, layout::PaddingRowMajor>> {
    using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template<class Element>
struct L1ATypeSelector<Gemm::GemmType<Element, layout::ColumnMajor>> {
    using L1AType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>;
};

template<class Element>
struct L1ATypeSelector<Gemm::GemmType<Element, layout::PaddingColumnMajor>> {
    using L1AType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>;
};

template<class GmBType>
struct L1BTypeSelector {
    static_assert(DEPENDENT_FALSE<GmBType>,
        "Unsupported layout selector, can not find the specialization.");
};

template<class Element>
struct L1BTypeSelector<Gemm::GemmType<Element, layout::RowMajor>> {
    using L1BType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template<class Element>
struct L1BTypeSelector<Gemm::GemmType<Element, layout::zN>> {
    using L1BType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template<class Element>
struct L1BTypeSelector<Gemm::GemmType<Element, layout::PaddingRowMajor>> {
    using L1BType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template<class Element>
struct L1BTypeSelector<Gemm::GemmType<Element, layout::ColumnMajor>> {
    using L1BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>;
};

template<class Element>
struct L1BTypeSelector<Gemm::GemmType<Element, layout::PaddingColumnMajor>> {
    using L1BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>;
};

template<class Element, class Layout, class Enable = void>
struct L1AlignHelperTla {
    static_assert(DEPENDENT_FALSE<Element>, "Unsupported align helper tla, can not find the specialization.");
};

template<class Element, class Layout>
struct L1AlignHelperTla<Element, Layout, std::enable_if_t<tla::detail::isRowMajor<Layout>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t M_ALIGNED = C0_NUM_PER_FRACTAL;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = ELE_NUM_PER_C0;
};

template<class Element, class Layout>
struct L1AlignHelperTla<Element, Layout, std::enable_if_t<tla::detail::isColumnMajor<Layout>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t M_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = C0_NUM_PER_FRACTAL;
};

} // namespace Act::Gemm::helper

#endif // ACT_GEMM_HELPER_HPP
