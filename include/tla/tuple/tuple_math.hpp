/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_TUPLE_MATH_HPP
#define TLA_TUPLE_MATH_HPP

#include "tla/tuple/tuple.hpp"
#include "tla/tuple/tuple_algorithms.hpp"
#include "tla/utils/math.hpp"
#include "tla/utils/functional.hpp"
#include "tla/utils/type_traits.hpp"
#include "tla/integral/integral_constant.hpp"
#include "tla/integral/integral_sequence.hpp"
#include "catlass/detail/macros.hpp"

namespace tla {

// Implementation of product as a function object
struct Product {
    template <class IntTuple>
    CATLASS_HOST_DEVICE constexpr auto operator()(IntTuple const& a) const
    {
        if constexpr (is_tuple<IntTuple>::value) {
            if constexpr (tuple_size<IntTuple>::value == 0) {
                return Int<1>{};
            } else {
                return tla::transform_apply(a, Product{}, multiplies_unary_lfold{});
            }
        } else if constexpr (tla::is_integral<IntTuple>::value) {
            return a;
        }
    }
};

namespace detail {

template <size_t N, typename Sequence>
struct MakeZeroTupleImpl;

template <size_t N, size_t... Is>
struct MakeZeroTupleImpl<N, tla::index_sequence<Is...>> {
    using type = tla::tuple<tla::Int<Is * 0>...>;
};

template <size_t N>
using MakeZeroTuple = typename MakeZeroTupleImpl<N, tla::make_index_sequence<N>>::type;

} // end namespace detail

// Add: element-wise addition (recursive)
template <class IntTupleA, class IntTupleB>
CATLASS_HOST_DEVICE constexpr auto Add(IntTupleA const& a, IntTupleB const& b)
{
    if constexpr (is_tuple<IntTupleA>::value && is_tuple<IntTupleB>::value) {
        TLA_ASSERT_SAME_TUPLE_SIZE(IntTupleA, IntTupleB);
        return transform(a, b, [](auto const& x, auto const& y) { return Add(x, y); });
    } else {
        return a + b;
    }
}

// product: compute the product of all elements (recursive)
template <class T>
CATLASS_HOST_DEVICE constexpr auto product(T const& t)
{
    return Product{}(t);
}

// size: total element count (= product)
template <class T>
CATLASS_HOST_DEVICE constexpr auto size(T const& t)
{
    return product(t);
}

// product_like: take product of tuple at the leaves of guide
template <class Tuple, class TupleG>
CATLASS_HOST_DEVICE constexpr auto product_like(Tuple const& tuple, TupleG const& guide)
{
    return transform_leaf(guide, tuple, [](auto const&, auto const& t) { return product(t); });
}

// minimum: element-wise min (recursive, supports broadcasting)
template <class T0, class T1>
CATLASS_HOST_DEVICE constexpr auto minimum(T0 const& t0, T1 const& t1)
{
    if constexpr (is_tuple<T0>::value && is_tuple<T1>::value) {
        return transform(t0, t1, [](auto const& a, auto const& b) { return minimum(a, b); });
    } else if constexpr (is_tuple<T0>::value) {
        return transform(t0, [&](auto const& a) { return minimum(a, t1); });
    } else if constexpr (is_tuple<T1>::value) {
        return transform(t1, [&](auto const& b) { return minimum(t0, b); });
    } else {
        return min(t0, t1);
    }
}

// maximum: element-wise max (recursive, supports broadcasting)
template <class T0, class T1>
CATLASS_HOST_DEVICE constexpr auto maximum(T0 const& t0, T1 const& t1)
{
    if constexpr (is_tuple<T0>::value && is_tuple<T1>::value) {
        return transform(t0, t1, [](auto const& a, auto const& b) { return maximum(a, b); });
    } else if constexpr (is_tuple<T0>::value) {
        return transform(t0, [&](auto const& a) { return maximum(a, t1); });
    } else if constexpr (is_tuple<T1>::value) {
        return transform(t1, [&](auto const& b) { return maximum(t0, b); });
    } else {
        return max(t0, t1);
    }
}

// clip_sub: element-wise clamping subtraction (recursive, supports broadcasting)
template <class T0, class T1, TLA_REQUIRES(is_tuple<remove_cvref_t<T0>>::value || is_tuple<remove_cvref_t<T1>>::value)>
CATLASS_HOST_DEVICE constexpr auto clip_sub(T0 const& t0, T1 const& t1)
{
    if constexpr (is_tuple<T0>::value && is_tuple<T1>::value) {
        return transform(t0, t1, [](auto const& a, auto const& b) { return clip_sub(a, b); });
    } else if constexpr (is_tuple<T0>::value) {
        return transform(t0, [&](auto const& a) { return clip_sub(a, t1); });
    } else {
        return transform(t1, [&](auto const& b) { return clip_sub(t0, b); });
    }
}

// ceil_div: tiling-aware ceiling division (recursive)
// - both tuples: pad shorter with 1, element-wise
// - tuple / scalar: cascading (divisor consumed across elements)
// - scalar / tuple: product then divide
template <class T0, class T1, TLA_REQUIRES(is_tuple<remove_cvref_t<T0>>::value || is_tuple<remove_cvref_t<T1>>::value)>
CATLASS_HOST_DEVICE constexpr auto ceil_div(T0 const& a, T1 const& b)
{
    if constexpr (is_tuple<T0>::value && is_tuple<T1>::value) {
        static_assert(tuple_size<T0>::value >= tuple_size<T1>::value, "Mismatched ranks");
        constexpr int R = tuple_size<T0>::value;
        return transform(a, append<R>(b, Int<1>{}), [](auto const& x, auto const& y) { return ceil_div(x, y); });
    } else if constexpr (is_tuple<T0>::value) {
        auto result = fold(a, make_tuple(make_tuple(), b), [](auto const& init, auto const& ai) {
            return make_tuple(append(get<0>(init), ceil_div(ai, get<1>(init))), ceil_div(get<1>(init), ai));
        });
        return get<0>(result);
    } else {
        return ceil_div(a, product(b));
    }
}

// round_up: element-wise round up to multiple (recursive, supports broadcasting)
template <class T0, class T1, TLA_REQUIRES(is_tuple<remove_cvref_t<T0>>::value || is_tuple<remove_cvref_t<T1>>::value)>
CATLASS_HOST_DEVICE constexpr auto round_up(T0 const& t0, T1 const& t1)
{
    if constexpr (is_tuple<T0>::value && is_tuple<T1>::value) {
        static_assert(tuple_size<T0>::value >= tuple_size<T1>::value, "Mismatched ranks");
        constexpr int R = tuple_size<T0>::value;
        return transform(t0, append<R>(t1, Int<1>{}), [](auto const& a, auto const& b) { return round_up(a, b); });
    } else if constexpr (is_tuple<T0>::value) {
        return transform(t0, [&](auto const& a) { return round_up(a, t1); });
    } else {
        return transform(t1, [&](auto const& b) { return round_up(t0, b); });
    }
}

// inner_product: dot product (recursive). For tuples, element-wise multiply then sum;
// for scalars, returns the product.
template <class T0, class T1>
CATLASS_HOST_DEVICE constexpr auto inner_product(T0 const& t0, T1 const& t1)
{
    if constexpr (is_tuple<T0>::value && is_tuple<T1>::value) {
        TLA_ASSERT_SAME_TUPLE_SIZE(T0, T1);
        return transform_apply(
            t0, t1, [](auto const& a, auto const& b) { return inner_product(a, b); },
            [](auto const&... v) { return (Int<0>{} + ... + v); });
    } else {
        return t0 * t1;
    }
}

} // end namespace tla

#endif // TLA_TUPLE_MATH_HPP
