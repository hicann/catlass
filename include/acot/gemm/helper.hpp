/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_GEMM_HELPER_HPP
#define ACOT_GEMM_HELPER_HPP

#include "acot/acot.hpp"
#include "acot/layout/layout.hpp"

namespace acot::gemm::helper{
template<class ElementA, class ElementB>
struct ElementAccumulatorSelector {};

template<>
struct ElementAccumulatorSelector<half, half> {
    using ElementAccumulator = float;
};

template<>
struct ElementAccumulatorSelector<int8_t, int8_t> {
    using ElementAccumulator = int32_t;
};

template<>
struct ElementAccumulatorSelector<int32_t, int32_t> {
    using ElementAccumulator = int32_t;
};

template<>
struct ElementAccumulatorSelector<bfloat16_t, bfloat16_t> {
    using ElementAccumulator = float;
};

template<>
struct ElementAccumulatorSelector<float, float> {
    using ElementAccumulator = float;
};
}

#endif // ACOT_GEMM_HELPER_HPP