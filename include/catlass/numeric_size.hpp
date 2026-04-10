#ifndef CATLASS_NUMERIC_SIZE_HPP
#define CATLASS_NUMERIC_SIZE_HPP

#include "catlass/catlass.hpp"

#include <cstddef>
#include <type_traits>

namespace Catlass {

#if defined(__CCE__)
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
template <typename T>
struct SizeOfBits {
    static constexpr size_t value =
        (std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<float4_e2m1x2_t>> ||
            std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<float4_e1m2x2_t>>)
            ? size_t{4}
            : AscendC::SizeOfBits<T>::value;
};
#else
// Atlas A2 等：沿用 AscendC 对子字节类型（如 int4b_t）的逻辑位宽，避免 sizeof(T)*8 与设备布局不一致。
using AscendC::SizeOfBits;
#endif
#else
template <typename T>
struct SizeOfBits {
    static constexpr size_t value = sizeof(T) * 8;
};
#endif

// Host / non-kernel TU with CATLASS_ARCH 3510: Ascend FP4 typedefs may still use sizeof==1; keep logical 4-bit count.
#if !defined(__CCE__)
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
template <>
struct SizeOfBits<float4_e2m1x2_t> {
    static constexpr size_t value = 4;
};

template <>
struct SizeOfBits<float4_e1m2x2_t> {
    static constexpr size_t value = 4;
};
#endif
#endif

/// Returns the number of bytes required to hold a specified number of bits
template <typename ReturnType = size_t, typename T>
CATLASS_HOST_DEVICE
constexpr ReturnType BitsToBytes(T bits)
{
    return (static_cast<ReturnType>(bits) + static_cast<ReturnType>(7)) / static_cast<ReturnType>(8);
}

/// Returns the number of bits required to hold a specified number of bytes
template <typename ReturnType = size_t, typename T>
CATLASS_HOST_DEVICE
constexpr ReturnType BytesToBits(T bytes)
{
    return static_cast<ReturnType>(bytes) * static_cast<ReturnType>(8);
}

} // namespace Catlass

#endif // CATLASS_NUMERIC_SIZE_HPP
