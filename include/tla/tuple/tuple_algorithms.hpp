/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_TUPLE_ALGORITHMS_HPP
#define TLA_TUPLE_ALGORITHMS_HPP

#include <utility>

#include "catlass/detail/macros.hpp"
#include "tla/integral/integral_constant.hpp"
#include "tla/integral/integral_math.hpp"
#include "tla/integral/integral_sequence.hpp"
#include "tla/tuple/tuple.hpp"
#include "tla/utils/type_traits.hpp"

namespace tla {

// ===========================================================================
// Apply: apply functions to tuple elements
// ===========================================================================

namespace detail {

template <class T, class F, int... I>
CATLASS_HOST_DEVICE constexpr auto apply_impl(T&& t, F&& f, seq<I...>)
{
    return f(get<I>(tla::forward<T>(t))...);
}

} // end namespace detail

// apply: unpack tuple into function call — f(t_0, t_1, ..., t_n)
template <class T, class F>
CATLASS_HOST_DEVICE constexpr auto apply(T&& t, F&& f)
{
    TLA_ASSERT_ALL_TUPLES(remove_cvref_t<T>);
    return detail::apply_impl(tla::forward<T>(t), f, tuple_seq<T>{});
}

// for_each: apply f to each element (no return value)
template <class T, class F>
CATLASS_HOST_DEVICE constexpr void for_each(T&& t, F&& f)
{
    if constexpr (is_tuple<remove_cvref_t<T>>::value) {
        return detail::apply_impl(t, [&](auto&&... a) { (f(tla::forward<decltype(a)>(a)), ...); }, tuple_seq<T>{});
    } else {
        return f(tla::forward<T>(t));
    }
}

// ===========================================================================
// Construction: wrap, select, take, repeat, tuple_cat, append, prepend
// ===========================================================================

// wrap: wrap non-tuples into a rank-1 tuple, leave tuples unchanged
template <class T>
CATLASS_HOST_DEVICE constexpr auto wrap(T const& t)
{
    if constexpr (is_tuple<T>::value) {
        return t;
    } else {
        return make_tuple(t);
    }
}

// select<I...>(t): select elements at the given indices into a new tuple
template <size_t... I, class T>
CATLASS_HOST_DEVICE constexpr auto select(T const& t)
{
    TLA_ASSERT_ALL_TUPLES(T);
    return make_tuple(get<I>(t)...);
}

// take<B, E>(t): take elements in range [B, E)
template <int B, int E, class T>
CATLASS_HOST_DEVICE constexpr auto take(T const& t)
{
    TLA_ASSERT_ALL_TUPLES(T);
    static_assert(B <= E, "take: B must be <= E");
    return detail::apply_impl(t, [](auto const&... a) { return make_tuple(a...); }, make_range<B, E>{});
}

// repeat<N>(t): N=1 returns t as-is; N!=1 wraps N copies in a tuple
// tuple_repeat<N>(t): always wraps N copies in a tuple
namespace detail {

template <size_t N, class T, size_t... I>
CATLASS_HOST_DEVICE constexpr auto tuple_repeat_impl(T const& t, index_sequence<I...>)
{
    return make_tuple((void(I), t)...);
}

} // end namespace detail

template <size_t N, class T>
CATLASS_HOST_DEVICE constexpr auto tuple_repeat(T const& t)
{
    return detail::tuple_repeat_impl<N>(t, make_index_sequence<N>{});
}

template <size_t N, class T>
CATLASS_HOST_DEVICE constexpr auto repeat(T const& t)
{
    if constexpr (N == 1) {
        return t;
    } else {
        return tuple_repeat<N>(t);
    }
}

// tuple_cat: concatenate tuples
namespace detail {

template <class T0, class T1, size_t... I0, size_t... I1>
CATLASS_HOST_DEVICE constexpr auto tuple_cat_impl(
    T0 const& t0, T1 const& t1, index_sequence<I0...>, index_sequence<I1...>)
{
    return make_tuple(get<I0>(t0)..., get<I1>(t1)...);
}

} // end namespace detail

CATLASS_HOST_DEVICE constexpr tuple<> tuple_cat()
{
    return {};
}

template <class T, TLA_REQUIRES(is_tuple<T>::value)>
CATLASS_HOST_DEVICE constexpr auto tuple_cat(T const& t)
{
    return t;
}

template <class T0, class T1>
CATLASS_HOST_DEVICE constexpr auto tuple_cat(T0 const& t0, T1 const& t1)
{
    TLA_ASSERT_ALL_TUPLES(T0, T1);
    return detail::tuple_cat_impl(
        t0, t1, make_index_sequence<tuple_size<T0>::value>{}, make_index_sequence<tuple_size<T1>::value>{});
}

template <class T0, class T1, class... Ts>
CATLASS_HOST_DEVICE constexpr auto tuple_cat(T0 const& t0, T1 const& t1, Ts const&... ts)
{
    TLA_ASSERT_ALL_TUPLES(T0, T1, Ts...);
    return tuple_cat(tuple_cat(t0, t1), ts...);
}

// append: append elements to a tuple
namespace detail {

template <class Tuple, class T, size_t... I>
CATLASS_HOST_DEVICE constexpr auto append_impl(Tuple const& t, T const& v, index_sequence<I...>)
{
    return make_tuple(get<I>(t)..., v);
}

} // end namespace detail

// append(a, x): append single element to a tuple
template <class T, class X>
CATLASS_HOST_DEVICE constexpr auto append(T const& a, X const& x)
{
    TLA_ASSERT_ALL_TUPLES(T);
    return detail::append_impl(a, x, make_index_sequence<tuple_size<T>::value>{});
}

// append<N>(a, x): pad a tuple to rank N by appending copies of x
template <int N, class T, class X>
CATLASS_HOST_DEVICE constexpr auto append(T const& a, X const& x)
{
    TLA_ASSERT_ALL_TUPLES(T);
    constexpr int S = tuple_size<T>::value;
    static_assert(N >= S, "append<N>: N must be >= tuple size");
    return tuple_cat(a, tuple_repeat<N - S>(x));
}

// prepend: prepend a single element to the beginning of a tuple
namespace detail {

template <class Tuple, class T, size_t... I>
CATLASS_HOST_DEVICE constexpr auto prepend_impl(Tuple const& t, T const& v, index_sequence<I...>)
{
    return make_tuple(v, get<I>(t)...);
}

} // end namespace detail

template <class T, class X>
CATLASS_HOST_DEVICE constexpr auto prepend(T const& t, X const& x)
{
    TLA_ASSERT_ALL_TUPLES(T);
    return detail::prepend_impl(t, x, make_index_sequence<tuple_size<T>::value>{});
}

// ===========================================================================
// Transform: map elements to produce new values/tuples
// ===========================================================================

namespace detail {

template <int I, class F, class T>
CATLASS_HOST_DEVICE constexpr auto get_apply_impl(F&& f, T&& t)
{
    return tla::apply(t, [&](auto&&... args) { return f(get<I>(tla::forward<decltype(args)>(args))...); });
}

template <class F, class G, int... I, class... Ts>
CATLASS_HOST_DEVICE constexpr auto transform_apply_impl(F&& f, G&& g, seq<I...>, Ts&&... ts)
{
    auto tp = tla::make_tuple(tla::forward<Ts>(ts)...);
    return g(get_apply_impl<I>(f, tp)...);
}

} // end namespace detail

// transform_apply: transform each element with f, then pack with g
template <class F, class G, class T0, class... Ts>
CATLASS_HOST_DEVICE constexpr auto transform_apply(F&& f, G&& g, T0&& t0, Ts&&... ts)
{
    TLA_ASSERT_SAME_TUPLE_SIZE(remove_cvref_t<T0>, remove_cvref_t<Ts>...);
    if constexpr (is_tuple<remove_cvref_t<T0>>::value) {
        return detail::transform_apply_impl(f, g, tuple_seq<T0>{}, tla::forward<T0>(t0), tla::forward<Ts>(ts)...);
    } else {
        return g(f(tla::forward<T0>(t0), tla::forward<Ts>(ts)...));
    }
}

// transform: apply f to each element, return new tuple
template <class F, class T0, class... Ts>
CATLASS_HOST_DEVICE constexpr auto transform(F&& f, T0 const& t0, Ts const&... ts)
{
    TLA_ASSERT_SAME_TUPLE_SIZE(T0, Ts...);
    if constexpr (is_tuple<T0>::value) {
        return detail::transform_apply_impl(
            f, [](auto const&... a) { return tla::make_tuple(a...); }, tuple_seq<T0>{}, t0, ts...);
    } else {
        return f(t0, ts...);
    }
}

// transform_leaf: recursively apply f to leaf (non-tuple) elements
template <class F, class T0, class... Ts>
CATLASS_HOST_DEVICE constexpr auto transform_leaf(F&& f, T0 const& t0, Ts const&... ts)
{
    if constexpr (is_tuple<T0>::value) {
        return transform([&](auto const&... a) { return transform_leaf(f, a...); }, t0, ts...);
    } else {
        return f(t0, ts...);
    }
}

// filter_tuple: transform each element with f, then concatenate results
template <class T, class F>
CATLASS_HOST_DEVICE constexpr auto filter_tuple(T const& t, F&& f)
{
    return transform_apply(f, [](auto const&... a) { return tuple_cat(a...); }, t);
}

// ===========================================================================
// Fold: left fold and reverse fold over tuple elements
// ===========================================================================

namespace detail {

template <class T, class V, class F, int I0, int... Is>
CATLASS_HOST_DEVICE constexpr auto fold_impl(T const& t, V const& v, F&& f, seq<I0, Is...>)
{
    return fold_impl(t, f(v, get<I0>(t)), f, seq<Is...>{});
}

template <class T, class V, class F>
CATLASS_HOST_DEVICE constexpr auto fold_impl(T const& t, V const& v, F&& f, seq<>)
{
    return v;
}

} // end namespace detail

// fold: f(...f(f(v, t_0), t_1),...,t_n)
template <class T, class V, class F>
CATLASS_HOST_DEVICE constexpr auto fold(T const& t, V const& v, F&& f)
{
    if constexpr (is_tuple<T>::value) {
        return detail::fold_impl(t, v, f, tuple_seq<T>{});
    } else {
        return f(v, t);
    }
}

// fold_reverse: f(...f(f(v, t_n), t_{n-1}),...,t_0)
template <class T, class V, class F>
CATLASS_HOST_DEVICE constexpr auto fold_reverse(T const& t, V const& v, F&& f)
{
    if constexpr (is_tuple<T>::value) {
        return detail::fold_impl(t, v, f, tuple_rseq<T>{});
    } else {
        return f(v, t);
    }
}

// ===========================================================================
// Properties: rank, depth, is_flat
// ===========================================================================

// rank: number of elements at the top level
template <int... Is, class Tuple>
CATLASS_HOST_DEVICE constexpr auto rank(Tuple const& t)
{
    if constexpr (sizeof...(Is) == 0) {
        if constexpr (is_tuple<Tuple>::value) {
            return Int<tuple_size<Tuple>::value>{};
        } else {
            return Int<1>{};
        }
    } else {
        return rank(get<Is...>(t));
    }
}

template <class Tuple>
using rank_t = decltype(rank(std::declval<Tuple>()));

template <class Tuple>
constexpr auto rank_v = rank_t<Tuple>::value;

// depth: maximum nesting level (scalar=0, flat tuple=1, ...)
template <int... Is, class Tuple>
CATLASS_HOST_DEVICE constexpr auto depth(Tuple const& t)
{
    if constexpr (sizeof...(Is) == 0) {
        if constexpr (is_tuple<Tuple>::value) {
            return Int<1>{} + tla::apply(t, [](auto const&... v) {
                       if constexpr (sizeof...(v) == 0) {
                           return Int<0>{};
                       } else if constexpr (sizeof...(v) == 1) {
                           return depth(v...);
                       } else {
                           return max(depth(v)...);
                       }
                   });
        } else {
            return Int<0>{};
        }
    } else {
        return depth(get<Is...>(t));
    }
}

template <class Tuple>
using depth_t = decltype(depth(std::declval<Tuple>()));

template <class Tuple>
constexpr auto depth_v = depth_t<Tuple>::value;

// is_flat: true if tuple has no nested tuples
template <class T>
struct is_flat : true_type {};

template <class... Ts>
struct is_flat<tuple<Ts...>> : Bool<(true && ... && (!is_tuple<Ts>::value))> {};

// ===========================================================================
// Flatten: flatten and unflatten hierarchical tuples
// ===========================================================================

// flatten_to_tuple: flatten a hierarchical tuple to depth 1, wrap non-tuples
template <class T>
CATLASS_HOST_DEVICE constexpr auto flatten_to_tuple(T const& t)
{
    if constexpr (is_tuple<T>::value && !is_flat<T>::value) {
        return filter_tuple(t, [](auto const& a) { return flatten_to_tuple(a); });
    } else {
        return wrap(t);
    }
}

// unflatten: reconstruct a hierarchical tuple from a flat tuple, guided by target_profile
namespace detail {

template <class FlatTuple, class TargetProfile>
CATLASS_HOST_DEVICE constexpr auto unflatten_impl(FlatTuple const& flat_tuple, TargetProfile const& target_profile)
{
    if constexpr (is_tuple<TargetProfile>::value) {
        return fold(target_profile, make_tuple(make_tuple(), flat_tuple), [](auto const& v, auto const& t) {
            auto sub = unflatten_impl(get<1>(v), t);
            return make_tuple(append(get<0>(v), get<0>(sub)), get<1>(sub));
        });
    } else {
        return make_tuple(get<0>(flat_tuple), take<1, rank_v<FlatTuple>>(flat_tuple));
    }
}

} // end namespace detail

template <class FlatTuple, class TargetProfile>
CATLASS_HOST_DEVICE constexpr auto unflatten(FlatTuple const& flat_tuple, TargetProfile const& target_profile)
{
    TLA_ASSERT_ALL_TUPLES(FlatTuple);
    auto result = detail::unflatten_impl(flat_tuple, target_profile);
    static_assert(rank_v<decltype(get<1>(result))> == 0, "unflatten: remainder must be empty");
    return get<0>(result);
}

} // end namespace tla

#endif // TLA_TUPLE_ALGORITHMS_HPP
