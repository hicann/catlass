/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_TUPLE_CONCEPTS_HPP
#define TLA_TUPLE_CONCEPTS_HPP

#include "tla/tuple/tuple_algorithms.hpp"
#include "tla/utils/type_traits.hpp"
#include "catlass/detail/macros.hpp"

namespace tla {

// evenly_divides: check if t0 is evenly divisible by t1 (element-wise)
template <class T0, class T1>
CATLASS_HOST_DEVICE constexpr auto evenly_divides(T0 const& t0, T1 const& t1)
{
    if constexpr (is_tuple<T0>::value) {
        return transform_apply(
            [](auto const& a, auto const& b) { return (a % b) == 0; }, [](auto... bs) { return (bs && ...); }, t0, t1);
    } else {
        return (t0 % t1) == 0;
    }
}

} // end namespace tla

#endif // TLA_TUPLE_CONCEPTS_HPP
