/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDCT_MATMUL_HELPER_HPP
#define ASCENDCT_MATMUL_HELPER_HPP

#include "AscendCT/AscendCT.hpp"
#include "AscendCT/layout/layout.hpp"
#include "tla/layout.hpp"

namespace AscendCT::gemm::helper {

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

// confirm
template<class Element>
struct L1ATypeSelector<gemm::MatmulType<Element, layout::RowMajor>> {
    using L1AType = gemm::MatmulType<Element, layout::zN>;
};

template<class Element>
struct L1ATypeSelector<gemm::MatmulType<Element, layout::PaddingRowMajor>> {
    using L1AType = gemm::MatmulType<Element, layout::zN>;
};

template<class Element>
struct L1ATypeSelector<gemm::MatmulType<Element, layout::ColumnMajor>> {
    using L1AType = gemm::MatmulType<Element, layout::nZ>;
};

template<class Element>
struct L1ATypeSelector<gemm::MatmulType<Element, layout::PaddingColumnMajor>> {
    using L1AType = gemm::MatmulType<Element, layout::nZ>;
};

// for the reason that the conflict on the idea, so i have to add some special Element to avoid conflict
// new add
template<>
struct L1ATypeSelector<gemm::MatmulType<half, layout::ColumnMajor>> {
    using L1AType = gemm::MatmulType<half, layout::nN>;
};

// new add
template<>
struct L1ATypeSelector<gemm::MatmulType<bfloat16_t, layout::ColumnMajor>> {
    using L1AType = gemm::MatmulType<bfloat16_t, layout::nN>;
};

// new add
template<>
struct L1ATypeSelector<gemm::MatmulType<float, layout::ColumnMajor>> {
    using L1AType = gemm::MatmulType<float, layout::nN>;
};

template<class GmBType>
struct L1BTypeSelector {
    static_assert(DEPENDENT_FALSE<GmBType>,
        "Unsupported layout selector, can not find the specialization.");
};

template<class Element>
struct L1BTypeSelector<gemm::MatmulType<Element, layout::RowMajor>> {
    using L1BType = gemm::MatmulType<Element, layout::zN>;
};

// for the reason that the conflict on the idea, so i have to add some special Element to avoid conflict
// new add
template<>
struct L1BTypeSelector<gemm::MatmulType<half, layout::RowMajor>> {
    using L1BType = gemm::MatmulType<half, layout::zZ>;
};

// new add
template<>
struct L1BTypeSelector<gemm::MatmulType<bfloat16_t, layout::RowMajor>> {
    using L1BType = gemm::MatmulType<bfloat16_t, layout::zZ>;
};

// new add
template<>
struct L1BTypeSelector<gemm::MatmulType<float, layout::RowMajor>> {
    using L1BType = gemm::MatmulType<float, layout::zZ>;
};

template<class Element>
struct L1BTypeSelector<gemm::MatmulType<Element, layout::zN>> {
    using L1BType = gemm::MatmulType<Element, layout::zN>;
};

template<class Element>
struct L1BTypeSelector<gemm::MatmulType<Element, layout::PaddingRowMajor>> {
    using L1BType = gemm::MatmulType<Element, layout::zN>;
};

// confirm
template<class Element>
struct L1BTypeSelector<gemm::MatmulType<Element, layout::ColumnMajor>> {
    using L1BType = gemm::MatmulType<Element, layout::nZ>;
};

template<class Element>
struct L1BTypeSelector<gemm::MatmulType<Element, layout::PaddingColumnMajor>> {
    using L1BType = gemm::MatmulType<Element, layout::nZ>;
};

/// the following code is added on 2025.03.21, for the solution of conflict about the different idea on the process of trans
/// add tutor wrq 万仁棋
// add L0TypeSelector
template<class L1Type>
struct L0ATypeSelector{};

// RowMajor
template<class Element>
struct L0ATypeSelector<gemm::MatmulType<Element, layout::zN>>{
    using L0AType = gemm::MatmulType<Element, layout::zZ>;
};

/// ColumnMajor
template<class Element>
struct L0ATypeSelector<gemm::MatmulType<Element, layout::nN>>{
    using L0AType = gemm::MatmulType<Element, layout::zN>;
};

/// ColumnMajor int8_t
template<>
struct L0ATypeSelector<gemm::MatmulType<int8_t, layout::nZ>>{
    using L0AType = gemm::MatmulType<int8_t, layout::zN>;
};

template<class L1Type>
struct L0BTypeSelector{};

// RowMajor
template<class Element>
struct L0BTypeSelector<gemm::MatmulType<Element, layout::zZ>>{
    using L0BType = gemm::MatmulType<Element, layout::nZ>;
};

// RowMajor int8_t
template<>
struct L0BTypeSelector<gemm::MatmulType<int8_t, layout::zN>>{
    using L0BType = gemm::MatmulType<int8_t, layout::nZ>;
};

// ColumnMajor
template<class Element>
struct L0BTypeSelector<gemm::MatmulType<Element, layout::nZ>>{
    using L0BType = gemm::MatmulType<Element, layout::nN>;
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

} // namespace AscendCT::gemm::helper

#endif // ASCENDCT_MATMUL_HELPER_HPP
