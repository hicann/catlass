/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_NUMERIC_INTEGER_SEQUENCE_HPP
#define TLA_NUMERIC_INTEGER_SEQUENCE_HPP

#include "tla/numeric/integral_constant.hpp"
#include "tla/type_traits.hpp"

namespace tla {

namespace Impl {
    constexpr size_t MaxIntegerSequenceSize = 64;
    constexpr size_t SpiltSize = 2;
};

template <class T, T... Ns> struct IntegerSequence {
    using type = IntegerSequence;
    using valueType = T;
    static_assert((0 <= sizeof...(Ns) && sizeof...(Ns) <= Impl::MaxIntegerSequenceSize), "Std::index_sequence size must be within [0,64].");
    __aicore__ inline static constexpr size_t size() { return sizeof...(Ns); }
};
        
namespace Impl {
    template <class T, class Seq0, class Seq1> struct MergeSeq {};
    template <class T, T... Ns0, T... Ns1>
    struct MergeSeq<T, IntegerSequence<T, Ns0...>, IntegerSequence<T, Ns1...>>
        : IntegerSequence<T, Ns0..., (sizeof...(Ns0) + Ns1)...> {};
    
    template <class T, size_t N>
    struct MakeIntegerSequence
        : Impl::MergeSeq<T, typename MakeIntegerSequence<T, N / SpiltSize>::type,
                            typename MakeIntegerSequence<T, N - N / SpiltSize>::type> {
    };
        
    template <class T> struct MakeIntegerSequence<T, 0> : IntegerSequence<T> {};
    template <class T> struct MakeIntegerSequence<T, 1> : IntegerSequence<T, 0> {};
}; // namespace Impl

template <class T, T N>
using MakeIntegerSequenceNoChecked =
    typename Impl::MakeIntegerSequence<T, N>::type;
template <class T, T N> struct MakeIntegerSequenceChecked {
    static_assert(0 <= N && N <= Impl::MaxIntegerSequenceSize,
                "Std::make_index_sequence must be within [0,64].");
    typedef MakeIntegerSequenceNoChecked<T, 0 <= N ? N : 0> type;
};

template <class T, T N>
using MakeIntegerSequence = typename MakeIntegerSequenceChecked<T, N>::type;


template <size_t... Idx>
using index_sequence = IntegerSequence<size_t, Idx...>;

template <size_t N>
using make_index_sequence = MakeIntegerSequence<size_t, N>;


// int_sequence
template <int... Ints>
using int_sequence = IntegerSequence<int, Ints...>;

template <int N>
using make_int_sequence = MakeIntegerSequence<int, N>;

// Shortcuts
template <int... Ints>
using seq = int_sequence<Ints...>;

template <int N>
using make_seq = make_int_sequence<N>;

template <class Tuple>
using tuple_seq = make_seq<tuple_size<tla::remove_cvref_t<Tuple>>::value>;

} // end namespace tla

#endif  // TLA_NUMERIC_INTEGER_SEQUENCE_HPP