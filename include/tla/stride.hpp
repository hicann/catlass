/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_STRIDE_HPP
#define TLA_STRIDE_HPP

#include "tla/tuple/tuple.hpp"
#include "tla/tuple/tuple_algorithms.hpp"
#include "tla/tuple/tuple_math.hpp"
#include "tla/utils/type_traits.hpp"
#include "tla/integral/integral_constant.hpp"
#include "tla/integral/integral_sequence.hpp"
#include "catlass/detail/macros.hpp"

namespace tla {

// compact_col_major: column-major compact strides.
// stride[0] = current, stride[i] = stride[i-1] * shape[i-1].
// Supports nested shapes (recurses into tuple elements) and custom starting stride.
template <class Shape, class Current = Int<1>>
CATLASS_HOST_DEVICE constexpr auto compact_col_major(Shape const& shape, Current const& current = {})
{
    if constexpr (is_tuple<Current>::value) {
        return transform(shape, current, [](auto const& s, auto const& c) { return compact_col_major(s, c); });
    } else if constexpr (is_tuple<Shape>::value) {
        auto result = fold(shape, make_tuple(make_tuple(), current), [](auto const& init, auto const& si) {
            auto stride = compact_col_major(si, get<1>(init));
            return make_tuple(append(get<0>(init), stride), get<1>(init) * product(si));
        });
        return get<0>(result);
    } else {
        return current;
    }
}

// compact_row_major: row-major compact strides.
// stride[N-1] = current, stride[i] = stride[i+1] * shape[i+1].
// Supports nested shapes and custom starting stride.
template <class Shape, class Current = Int<1>>
CATLASS_HOST_DEVICE constexpr auto compact_row_major(Shape const& shape, Current const& current = {})
{
    if constexpr (is_tuple<Current>::value) {
        return transform(shape, current, [](auto const& s, auto const& c) { return compact_row_major(s, c); });
    } else if constexpr (is_tuple<Shape>::value) {
        auto result = fold_reverse(shape, make_tuple(make_tuple(), current), [](auto const& init, auto const& si) {
            auto stride = compact_row_major(si, get<1>(init));
            return make_tuple(prepend(get<0>(init), stride), get<1>(init) * product(si));
        });
        return get<0>(result);
    } else {
        return current;
    }
}

// Return the offset of coord
template <class Coord, class Shape, class Stride>
CATLASS_HOST_DEVICE constexpr auto crd2idx(Coord const& coord, Shape const& shape, Stride const& stride);

namespace detail {

template <class Coord, class Shape, class Stride, int... Is>
CATLASS_HOST_DEVICE constexpr auto crd2idx_ttt(Coord const& coord, Shape const& shape, Stride const& stride, seq<Is...>)
{
    return (... + crd2idx(get<Is>(coord), get<Is>(shape), get<Is>(stride)));
}

template <class CInt, class STuple, class DTuple, int I0, int... Is>
CATLASS_HOST_DEVICE constexpr auto crd2idx_itt(
    CInt const& coord, STuple const& shape, DTuple const& stride, seq<I0, Is...>)
{
    if constexpr (sizeof...(Is) == 0) { // Avoid recursion and mod on single/last iter
        return crd2idx(coord, get<I0>(shape), get<I0>(stride));
    } else if constexpr (is_constant<0, CInt>::value) {
        return crd2idx(_0{}, get<I0>(shape), get<I0>(stride)) +
               (_0{} + ... + crd2idx(_0{}, get<Is>(shape), get<Is>(stride)));
    } else { // General case
        return crd2idx(coord % product(get<I0>(shape)), get<I0>(shape), get<I0>(stride)) +
               crd2idx_itt(coord / product(get<I0>(shape)), shape, stride, seq<Is...>{});
    }
}

} // end namespace detail

template <class Coord, class Shape, class Stride>
CATLASS_HOST_DEVICE constexpr auto crd2idx(Coord const& coord, Shape const& shape, Stride const& stride)
{
    if constexpr (is_tuple<Coord>::value) {
        if constexpr (is_tuple<Shape>::value) { // tuple tuple tuple
            static_assert(tuple_size<Coord>::value == tuple_size<Shape>::value, "Mismatched Ranks");
            static_assert(tuple_size<Coord>::value == tuple_size<Stride>::value, "Mismatched Ranks");
            return detail::crd2idx_ttt(coord, shape, stride, tuple_seq<Coord>{});
        } else { // tuple "int" "int"
            static_assert(sizeof(Coord) == 0, "Invalid parameters");
        }
    } else {
        if constexpr (is_tuple<Shape>::value) { // "int" tuple tuple
            static_assert(tuple_size<Shape>::value == tuple_size<Stride>::value, "Mismatched Ranks");
            return detail::crd2idx_itt(coord, shape, stride, tuple_seq<Shape>{});
        } else { // "int" "int" "int"
            return coord * stride;
        }
    }
}

// crd2offset: alias for crd2idx (backward compatibility)
template <class Coord, class Shape, class Stride>
CATLASS_HOST_DEVICE constexpr auto crd2offset(Coord const& coord, Shape const& shape, Stride const& stride)
{
    return crd2idx(coord, shape, stride);
}

// compact_order: compute compact strides based on an arbitrary ordering of modes.
// Order values indicate priority — lower order = more contiguous (smaller stride).
// Supports nested shapes and dynamic (runtime) order values.
namespace detail {

// Core algorithm: @pre is_static<Order>, @pre is_static<RefOrder>
template <class Shape, class Order, class RefShape, class RefOrder>
CATLASS_HOST_DEVICE constexpr auto compact_order_impl(
    Shape const& shape, Order const& order, RefShape const& ref_shape, RefOrder const& ref_order)
{
    if constexpr (is_tuple<Order>::value) {
        static_assert(tuple_size<Shape>::value == tuple_size<Order>::value, "Need equal rank of shape and order");
        return transform(
            shape, order, [&](auto const& s, auto const& o) { return compact_order_impl(s, o, ref_shape, ref_order); });
    } else {
        auto stride_start = product(transform(ref_shape, ref_order, [&](auto const& s, auto const& o) {
            return conditional_return(o < order, s, Int<1>{});
        }));
        return compact_col_major(shape, stride_start);
    }
}

} // end namespace detail

template <class Shape, class Order>
CATLASS_HOST_DEVICE constexpr auto compact_order(Shape const& shape, Order const& order)
{
    auto ref_shape = flatten_to_tuple(product_like(shape, order));
    auto flat_order = flatten_to_tuple(order);

    // Replace dynamic order elements with large static values so that
    // o < order comparisons inside compact_order_impl are compile-time.
    auto max_order = fold(flat_order, Int<0>{}, [](auto v, auto o) {
        if constexpr (is_constant<true, decltype(v < o)>::value) {
            return o;
        } else {
            return v;
        }
    });
    auto max_seq = make_range<decltype(max_order)::value + 1, decltype(max_order)::value + 1 + rank(flat_order)>{};
    auto ref_order = transform(max_seq, flat_order, [](auto seq_v, auto o) {
        if constexpr (is_static<decltype(o)>::value) {
            return o;
        } else {
            return seq_v;
        }
    });
    auto new_order = unflatten(ref_order, order);
    return detail::compact_order_impl(shape, new_order, ref_shape, ref_order);
}

} // namespace tla

#endif // TLA_STRIDE_HPP
