/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_INTEGRAL_INTEGRAL_CONSTANT_HPP
#define TLA_INTEGRAL_INTEGRAL_CONSTANT_HPP

#include "catlass/detail/macros.hpp"
#include "tla/utils/type_traits.hpp"

namespace tla {

// A constant value: short name and type-deduction for fast compilation
template <auto v>
struct C {
    using type = C<v>;
    static constexpr auto value = v;
    using value_type = decltype(v);
    CATLASS_HOST_DEVICE constexpr operator value_type() const noexcept
    {
        return value;
    }
    CATLASS_HOST_DEVICE constexpr value_type operator()() const noexcept
    {
        return value;
    }
};

// Deprecate
template <class T, T v>
using constant = C<v>;

template <bool b>
using bool_constant = C<b>;

using true_type = bool_constant<true>;
using false_type = bool_constant<false>;

// A more std:: conforming integral_constant that enforces type but interops with C<v>
template <class T, T v>
struct integral_constant : C<v> {
    using type = integral_constant<T, v>;
    static constexpr T value = v;
    using value_type = T;
    CATLASS_HOST_DEVICE constexpr value_type operator()() const noexcept
    {
        return value;
    }
};

// Use tla::is_std_integral<T> to match built-in integral types (int, int64_t, unsigned, etc)
// Use tla::is_integral<T> to match both built-in integral types AND static integral types.

template <class T>
struct is_integral : bool_constant<is_std_integral<T>::value> {};
template <auto v>
struct is_integral<C<v>> : true_type {};
template <class T, T v>
struct is_integral<integral_constant<T, v>> : true_type {};

template <class T>
inline constexpr bool is_integral_v = is_integral<T>::value;

// is_static detects if an (abstract) value is defined completely by its type (no members)
template <class T>
struct is_static : bool_constant<is_empty_v<remove_cvref_t<T>>> {};

template <class T>
inline constexpr bool is_static_v = is_static<T>::value;

// is_constant detects if a type is a static integral type and if v is equal to a value

template <auto n, class T>
struct is_constant_base : false_type {};
template <auto n, auto v>
struct is_constant_base<n, C<v>> : bool_constant<v == n> {};
template <auto n, class T, T v>
struct is_constant_base<n, integral_constant<T, v>> : bool_constant<v == n> {};

template <auto n, class T>
struct is_constant : is_constant_base<n, remove_cvref_t<T>> {};

template <auto n, class T>
inline constexpr bool is_constant_v = is_constant<n, T>::value;

// TLA_STATIC_CHECK(EXPR): if EXPR statically evaluates to C<false>, static_assert fails;
// if EXPR is not a static constant (e.g., runtime bool), the check is skipped.
#define TLA_STATIC_CHECK(...) static_assert(!is_constant_v<false, decltype(__VA_ARGS__)>, "static check failed.")

//
// Specializations
//

template <int v>
using Int = C<v>;

using _0 = Int<0>;
using _1 = Int<1>;
using _2 = Int<2>;
using _3 = Int<3>;
using _4 = Int<4>;
using _5 = Int<5>;
using _6 = Int<6>;
using _7 = Int<7>;
using _8 = Int<8>;
using _9 = Int<9>;
using _10 = Int<10>;
using _12 = Int<12>;
using _16 = Int<16>;
using _24 = Int<24>;
using _32 = Int<32>;
using _48 = Int<48>;
using _64 = Int<64>;
using _96 = Int<96>;
using _128 = Int<128>;
using _192 = Int<192>;
using _256 = Int<256>;
using _384 = Int<384>;
using _512 = Int<512>;
using _768 = Int<768>;
using _1024 = Int<1024>;
using _2048 = Int<2048>;
using _4096 = Int<4096>;
using _8192 = Int<8192>;
using _16384 = Int<16384>;
using _32768 = Int<32768>;
using _65536 = Int<65536>;

// Underscore placeholder (for slicing semantics)
// tla::_ is an empty tag for "take the whole dimension" in Coord/tensor indexing
struct Underscore {
    using type = Underscore;
};

// Treat Underscore as an integral type in the tla type system
template <>
struct is_integral<Underscore> : true_type {};

inline constexpr Underscore _{};

template <class T>
struct is_underscore : bool_constant<is_same_v<remove_cvref_t<T>, Underscore>> {};

template <class T>
inline constexpr bool is_underscore_v = is_underscore<T>::value;

} // namespace tla

#endif // TLA_INTEGRAL_INTEGRAL_CONSTANT_HPP
