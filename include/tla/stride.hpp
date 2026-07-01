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

template <class Shape, class Current = Int<1>>
CATLASS_HOST_DEVICE constexpr auto compact_col_major(Shape const& shape, Current const& current = {});

template <class Shape, class Current = Int<1>>
CATLASS_HOST_DEVICE constexpr auto compact_row_major(Shape const& shape, Current const& current = {});

namespace detail {

template <class Shape, class Current, int I, int... Is>
CATLASS_HOST_DEVICE constexpr auto compact_col_major_impl(Shape const& shape, Current const& current, seq<I, Is...>)
{
    auto stride = compact_col_major(get<I>(shape), current);
    if constexpr (sizeof...(Is) == 0) {
        return make_tuple(stride);
    } else {
        auto next = current * product(get<I>(shape));
        return prepend(compact_col_major_impl(shape, next, seq<Is...>{}), stride);
    }
}

template <class Shape, class Current, int I, int... Is>
CATLASS_HOST_DEVICE constexpr auto compact_row_major_impl(Shape const& shape, Current const& current, seq<I, Is...>)
{
    auto stride = compact_row_major(get<I>(shape), current);
    if constexpr (sizeof...(Is) == 0) {
        return make_tuple(stride);
    } else {
        auto next = current * product(get<I>(shape));
        return append(compact_row_major_impl(shape, next, seq<Is...>{}), stride);
    }
}

} // end namespace detail

// compact_col_major: column-major compact strides.
// stride[0] = current, stride[i] = stride[i-1] * shape[i-1].
// Supports nested shapes (recurses into tuple elements) and custom starting stride.
template <class Shape, class Current>
CATLASS_HOST_DEVICE constexpr auto compact_col_major(Shape const& shape, Current const& current)
{
    if constexpr (is_tuple<Current>::value) {
        return transform(shape, current, [](auto const& s, auto const& c) { return compact_col_major(s, c); });
    } else if constexpr (is_tuple<Shape>::value) {
        return detail::compact_col_major_impl(shape, current, tuple_seq<Shape>{});
    } else {
        return current;
    }
}

// compact_row_major: row-major compact strides.
// stride[N-1] = current, stride[i] = stride[i+1] * shape[i+1].
// Supports nested shapes and custom starting stride.
template <class Shape, class Current>
CATLASS_HOST_DEVICE constexpr auto compact_row_major(Shape const& shape, Current const& current)
{
    if constexpr (is_tuple<Current>::value) {
        return transform(shape, current, [](auto const& s, auto const& c) { return compact_row_major(s, c); });
    } else if constexpr (is_tuple<Shape>::value) {
        return detail::compact_row_major_impl(shape, current, tuple_rseq<Shape>{});
    } else {
        return current;
    }
}

namespace detail {

template <class FlatShape, class FlatOrder, class Order, int... Is>
CATLASS_HOST_DEVICE constexpr auto order_stride_start(
    FlatShape const& flat_shape, FlatOrder const& flat_order, Order const& order, seq<Is...>)
{
    return (Int<1>{} * ... * conditional_return(get<Is>(flat_order) < order, get<Is>(flat_shape), Int<1>{}));
}

template <class Shape, class Order, class FlatShape, class FlatOrder>
CATLASS_HOST_DEVICE constexpr auto compact_order_impl(
    Shape const& shape, Order const& order, FlatShape const& flat_shape, FlatOrder const& flat_order)
{
    if constexpr (is_tuple<Order>::value) {
        static_assert(tuple_size<Shape>::value == tuple_size<Order>::value, "Need equal rank of shape and order");
        return transform(shape, order, [&](auto const& s, auto const& o) {
            return compact_order_impl(s, o, flat_shape, flat_order);
        });
    } else {
        auto stride_start = order_stride_start(flat_shape, flat_order, order, tuple_seq<FlatShape>{});
        return compact_col_major(shape, stride_start);
    }
}

} // end namespace detail

template <class Shape, class Order, TLA_REQUIRES(is_all_static<remove_cvref_t<Order>>::value)>
CATLASS_HOST_DEVICE constexpr auto compact_order(Shape const& shape, Order const& order)
{
    auto flat_shape = flatten_to_tuple(product_like(shape, order));
    auto flat_order = flatten_to_tuple(order);
    return detail::compact_order_impl(shape, order, flat_shape, flat_order);
}

template <class Coord, class Shape, class Stride>
CATLASS_HOST_DEVICE constexpr auto crd2idx(Coord const& coord, Shape const& shape, Stride const& stride)
{
    if constexpr (is_tuple<Coord>::value && is_tuple<Shape>::value) {
        static_assert(tuple_size<Coord>::value == tuple_size<Shape>::value, "Mismatched Ranks");
        static_assert(tuple_size<Coord>::value == tuple_size<Stride>::value, "Mismatched Ranks");
        return transform_apply(
            coord, shape, stride, [](auto const& c, auto const& s, auto const& d) { return crd2idx(c, s, d); },
            [](auto const&... xs) { return (... + xs); });
    } else if constexpr (is_tuple<Coord>::value) {
        static_assert(dependent_false<Coord>, "Invalid parameters");
    } else if constexpr (is_tuple<Shape>::value) {
        static_assert(tuple_size<Shape>::value == tuple_size<Stride>::value, "Mismatched Ranks");
        if constexpr (is_constant<0, Coord>::value) {
            return _0{};
        }
        auto zipped = transform(shape, stride, [](auto const& s, auto const& d) { return make_tuple(s, d); });
        return get<1>(fold(zipped, make_tuple(coord, _0{}), [](auto const& acc, auto const& sd) {
            auto prod = product(get<0>(sd));
            return make_tuple(get<0>(acc) / prod, get<1>(acc) + crd2idx(get<0>(acc) % prod, get<0>(sd), get<1>(sd)));
        }));
    } else {
        return coord * stride;
    }
}

// crd2offset: alias for crd2idx (backward compatibility)
template <class Coord, class Shape, class Stride>
CATLASS_HOST_DEVICE constexpr auto crd2offset(Coord const& coord, Shape const& shape, Stride const& stride)
{
    return crd2idx(coord, shape, stride);
}

template <class Coord, class Shape>
CATLASS_HOST_DEVICE constexpr auto crd2idx(Coord const& coord, Shape const& shape)
{
    if constexpr (is_tuple<Coord>::value && is_tuple<Shape>::value) {
        static_assert(tuple_size<Coord>::value == tuple_size<Shape>::value, "Mismatched Ranks");
        auto flat_coord = flatten_to_tuple(coord);
        auto flat_shape = flatten_to_tuple(product_like(shape, coord));
        auto zipped = transform(flat_coord, flat_shape, [](auto const& c, auto const& s) { return make_tuple(c, s); });
        return fold_reverse(
            zipped, _0{}, [](auto const& acc, auto const& cs) { return get<0>(cs) + get<1>(cs) * acc; });
    } else if constexpr (is_tuple<Coord>::value) {
        static_assert(dependent_false<Coord>, "Invalid parameters");
    } else {
        return coord;
    }
}

} // namespace tla

#endif // TLA_STRIDE_HPP
