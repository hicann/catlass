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
CATLASS_HOST_DEVICE constexpr auto select(T&& t)
{
    return make_tuple(get<I>(tla::forward<T>(t))...);
}

// take<B, E>(t): take elements in range [B, E)
template <int B, int E, class T>
CATLASS_HOST_DEVICE constexpr auto take(T const& t)
{
    static_assert(B <= E, "take: B must be <= E");
    return detail::apply_impl(t, [](auto const&... a) { return make_tuple(a...); }, make_range<B, E>{});
}

// repeat<N>(t): N copies of t in a tuple
namespace detail {

template <size_t N, class T, size_t... I>
CATLASS_HOST_DEVICE constexpr auto repeat_impl(T const& t, index_sequence<I...>)
{
    return make_tuple((void(I), t)...);
}

} // end namespace detail

template <size_t N, class T>
CATLASS_HOST_DEVICE constexpr auto repeat(T const& t)
{
    return detail::repeat_impl<N>(t, make_index_sequence<N>{});
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

template <class T, TLA_REQUIRES(is_tuple<remove_cvref_t<T>>::value)>
CATLASS_HOST_DEVICE constexpr auto tuple_cat(T const& t)
{
    return t;
}

template <class T0, class T1>
CATLASS_HOST_DEVICE constexpr auto tuple_cat(T0 const& t0, T1 const& t1)
{
    static_assert(is_tuple<T0>::value && is_tuple<T1>::value, "tuple_cat arguments must be tuples");
    return detail::tuple_cat_impl(
        t0, t1, make_index_sequence<tuple_size<T0>::value>{}, make_index_sequence<tuple_size<T1>::value>{});
}

template <class T0, class T1, class... Ts>
CATLASS_HOST_DEVICE constexpr auto tuple_cat(T0 const& t0, T1 const& t1, Ts const&... ts)
{
    static_assert(
        is_tuple<T0>::value && is_tuple<T1>::value && (is_tuple<Ts>::value && ...),
        "tuple_cat arguments must be tuples");
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
template <class T, class X, TLA_REQUIRES(is_tuple<remove_cvref_t<T>>::value)>
CATLASS_HOST_DEVICE constexpr auto append(T const& a, X const& x)
{
    return detail::append_impl(a, x, make_index_sequence<tuple_size<T>::value>{});
}

// append<N>(a, x): pad a tuple to rank N by appending copies of x
template <int N, class T, class X, TLA_REQUIRES(is_tuple<remove_cvref_t<T>>::value)>
CATLASS_HOST_DEVICE constexpr auto append(T const& a, X const& x)
{
    constexpr int S = tuple_size<T>::value;
    static_assert(N >= S, "append<N>: N must be >= tuple size");
    return tuple_cat(a, repeat<N - S>(x));
}

// prepend: prepend a single element to the beginning of a tuple
namespace detail {

template <class Tuple, class T, size_t... I>
CATLASS_HOST_DEVICE constexpr auto prepend_impl(Tuple const& t, T const& v, index_sequence<I...>)
{
    return make_tuple(v, get<I>(t)...);
}

} // end namespace detail

template <class T, class X, TLA_REQUIRES(is_tuple<remove_cvref_t<T>>::value)>
CATLASS_HOST_DEVICE constexpr auto prepend(T const& t, X const& x)
{
    return detail::prepend_impl(t, x, make_index_sequence<tuple_size<T>::value>{});
}

// ===========================================================================
// Transform: map elements to produce new values/tuples
// ===========================================================================

namespace detail {

template <class T, class F, class G, int... I>
CATLASS_HOST_DEVICE constexpr auto tapply_impl(T&& t, F&& f, G&& g, seq<I...>)
{
    return g(f(get<I>(tla::forward<T>(t)))...);
}

template <class T0, class T1, class F, class G, int... I>
CATLASS_HOST_DEVICE constexpr auto tapply_impl(T0&& t0, T1&& t1, F&& f, G&& g, seq<I...>)
{
    return g(f(get<I>(tla::forward<T0>(t0)), get<I>(tla::forward<T1>(t1)))...);
}

template <class T0, class T1, class T2, class F, class G, int... I>
CATLASS_HOST_DEVICE constexpr auto tapply_impl(T0&& t0, T1&& t1, T2&& t2, F&& f, G&& g, seq<I...>)
{
    return g(f(get<I>(tla::forward<T0>(t0)), get<I>(tla::forward<T1>(t1)), get<I>(tla::forward<T2>(t2)))...);
}

} // end namespace detail

// transform_apply: transform each element with f, then pack with g
template <class T, class F, class G>
CATLASS_HOST_DEVICE constexpr auto transform_apply(T&& t, F&& f, G&& g)
{
    if constexpr (is_tuple<remove_cvref_t<T>>::value) {
        return detail::tapply_impl(tla::forward<T>(t), f, g, tuple_seq<T>{});
    } else {
        return g(f(tla::forward<T>(t)));
    }
}

template <class T0, class T1, class F, class G>
CATLASS_HOST_DEVICE constexpr auto transform_apply(T0&& t0, T1&& t1, F&& f, G&& g)
{
    if constexpr (is_tuple<remove_cvref_t<T0>>::value) {
        return detail::tapply_impl(tla::forward<T0>(t0), tla::forward<T1>(t1), f, g, tuple_seq<T0>{});
    } else {
        return g(f(tla::forward<T0>(t0), tla::forward<T1>(t1)));
    }
}

template <class T0, class T1, class T2, class F, class G>
CATLASS_HOST_DEVICE constexpr auto transform_apply(T0&& t0, T1&& t1, T2&& t2, F&& f, G&& g)
{
    if constexpr (is_tuple<remove_cvref_t<T0>>::value) {
        TLA_ASSERT_SAME_TUPLE_SIZE(remove_cvref_t<T0>, remove_cvref_t<T1>);
        TLA_ASSERT_SAME_TUPLE_SIZE(remove_cvref_t<T0>, remove_cvref_t<T2>);
        return detail::tapply_impl(
            tla::forward<T0>(t0), tla::forward<T1>(t1), tla::forward<T2>(t2), f, g, tuple_seq<T0>{});
    } else {
        return g(f(tla::forward<T0>(t0), tla::forward<T1>(t1), tla::forward<T2>(t2)));
    }
}

// transform: apply f to each element, return new tuple
template <class T, class F>
CATLASS_HOST_DEVICE constexpr auto transform(T const& t, F&& f)
{
    if constexpr (is_tuple<T>::value) {
        return detail::tapply_impl(t, f, [](auto const&... a) { return tla::make_tuple(a...); }, tuple_seq<T>{});
    } else {
        return f(t);
    }
}

template <class T0, class T1, class F>
CATLASS_HOST_DEVICE constexpr auto transform(T0 const& t0, T1 const& t1, F&& f)
{
    if constexpr (is_tuple<T0>::value) {
        TLA_ASSERT_SAME_TUPLE_SIZE(T0, T1);
        return detail::tapply_impl(t0, t1, f, [](auto const&... a) { return tla::make_tuple(a...); }, tuple_seq<T0>{});
    } else {
        return f(t0, t1);
    }
}

template <class T0, class T1, class T2, class F>
CATLASS_HOST_DEVICE constexpr auto transform(T0 const& t0, T1 const& t1, T2 const& t2, F&& f)
{
    if constexpr (is_tuple<T0>::value) {
        TLA_ASSERT_SAME_TUPLE_SIZE(T0, T1);
        TLA_ASSERT_SAME_TUPLE_SIZE(T0, T2);
        return detail::tapply_impl(
            t0, t1, t2, f, [](auto const&... a) { return tla::make_tuple(a...); }, tuple_seq<T0>{});
    } else {
        return f(t0, t1, t2);
    }
}

// filter_tuple: transform each element with f, then concatenate results
template <class T, class F>
CATLASS_HOST_DEVICE constexpr auto filter_tuple(T const& t, F&& f)
{
    return transform_apply(t, f, [](auto const&... a) { return tuple_cat(a...); });
}

// transform_leaf: recursively apply f to leaf (non-tuple) elements
template <class T, class F>
CATLASS_HOST_DEVICE constexpr auto transform_leaf(T const& t, F&& f)
{
    if constexpr (is_tuple<T>::value) {
        return transform(t, [&](auto const& a) { return transform_leaf(a, f); });
    } else {
        return f(t);
    }
}

template <class T0, class T1, class F>
CATLASS_HOST_DEVICE constexpr auto transform_leaf(T0 const& t0, T1 const& t1, F&& f)
{
    if constexpr (is_tuple<T0>::value) {
        return transform(t0, t1, [&](auto const& a, auto const& b) { return transform_leaf(a, b, f); });
    } else {
        return f(t0, t1);
    }
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
    auto result = detail::unflatten_impl(flat_tuple, target_profile);
    static_assert(rank_v<decltype(get<1>(result))> == 0, "unflatten: remainder must be empty");
    return get<0>(result);
}

} // end namespace tla

#endif // TLA_TUPLE_ALGORITHMS_HPP
