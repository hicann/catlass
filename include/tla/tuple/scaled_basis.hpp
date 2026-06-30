/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_TUPLE_SCALED_BASIS_HPP
#define TLA_TUPLE_SCALED_BASIS_HPP

#include "catlass/detail/macros.hpp"
#include "tla/integral/integral_constant.hpp"
#include "tla/tuple/tuple.hpp"
#include "tla/utils/type_traits.hpp"

namespace tla {

// ScaledBasis: a basis vector carrying a scalar scale and mode indices (Ns...).
// The mode indices are encoded in the template parameter pack and are not stored
// as runtime data. ScaledBasis privately inherits tuple<T> to reuse EBO storage.
template <class T, int... Ns>
struct ScaledBasis : private tuple<T> {
    using base_type = tuple<T>;
    using type = ScaledBasis;

    CATLASS_HOST_DEVICE constexpr ScaledBasis() : base_type{}
    {}

    CATLASS_HOST_DEVICE constexpr ScaledBasis(T const& val) : base_type{val}
    {}

    // Perfectly forward get<0>'s return: by ref for non-empty T (e.g. int),
    // by value for empty static T (e.g. Int<N>) where the EBO stores nothing.
    CATLASS_HOST_DEVICE constexpr decltype(auto) value() const
    {
        return get<0>(static_cast<base_type const&>(*this));
    }
};

// E: unit basis vector (scale == 1) with mode indices Ns...
template <int... Ns>
using E = ScaledBasis<Int<1>, Ns...>;

// is_scaled_basis: type trait, true only for ScaledBasis specializations.
template <class T>
struct is_scaled_basis : false_type {};

template <class T, int... Ns>
struct is_scaled_basis<ScaledBasis<T, Ns...>> : true_type {};

// basis_get: unwrap a ScaledBasis to its scale value, or return the input as-is.
template <class T>
CATLASS_HOST_DEVICE constexpr decltype(auto) basis_get(T const& t)
{
    if constexpr (is_scaled_basis<T>::value) {
        return t.value();
    } else {
        return t;
    }
}

// basis_value: alias of basis_get for call-site clarity.
template <class T>
CATLASS_HOST_DEVICE constexpr decltype(auto) basis_value(T const& t)
{
    return basis_get(t);
}

// as_arithmetic_tuple: convert a basis-like value into a plain arithmetic tuple.
// For plain tuples and scalars (the coshape closure path), this is the identity.
// For ScaledBasis, expand to a plain tuple holding the scale value.
// The SFINAE constraint keeps the greedy forwarding reference from shadowing
// the ScaledBasis-specific overload below.
template <class T, TLA_REQUIRES(!is_scaled_basis<remove_cvref_t<T>>::value)>
CATLASS_HOST_DEVICE constexpr decltype(auto) as_arithmetic_tuple(T&& t)
{
    return static_cast<T&&>(t);
}

template <class T, int... Ns>
CATLASS_HOST_DEVICE constexpr auto as_arithmetic_tuple(ScaledBasis<T, Ns...> const& b)
{
    return make_tuple(b.value());
}

// make_basis_like: produce a unit basis (E) matching the rank of a shape.
// Minimal placeholder: the coshape closure does not call this directly; the full
// coordinate-decomposition form will be filled in once Task 12 (coshape) lands.
template <class Shape>
CATLASS_HOST_DEVICE constexpr auto make_basis_like(Shape const& shape)
{
    if constexpr (is_tuple<remove_cvref_t<Shape>>::value) {
        return Int<1>{};
    } else {
        return Int<1>{};
    }
}

} // end namespace tla

#endif // TLA_TUPLE_SCALED_BASIS_HPP
