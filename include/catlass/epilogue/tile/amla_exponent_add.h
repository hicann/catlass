/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATLASS_EPILOGUE_TILE_AMLA_EXPONENT_ADD_H
#define CATLASS_EPILOGUE_TILE_AMLA_EXPONENT_ADD_H

#include <cstdint>
#include "catlass/detail/macros.hpp"

namespace Catlass::Epilogue::Tile {

// Extend AMLA's normalized-float integer exponent update to zero/subnormals.
// N includes the original AMLA mantissa correction. This is not a replacement
// for the approximation itself: normal-to-normal results are bit-identical to
// the original integer addition. The exceptional path uses integer operations
// only and rounds an underflowing significand to nearest, ties to even.
CATLASS_HOST_DEVICE inline uint32_t AmlaExponentAddBits(uint32_t bits, int32_t n)
{
    // IEEE 754 binary32 has 23 stored fraction bits plus the implicit leading bit.
    constexpr uint32_t FRACTION_BITS = 23;
    constexpr uint32_t SIGNIFICAND_BITS = FRACTION_BITS + 1;
    constexpr uint32_t SIGN = 0x80000000u;
    constexpr uint32_t HIDDEN = uint32_t(1) << FRACTION_BITS;
    constexpr uint32_t MANTISSA = HIDDEN - 1;
    constexpr uint32_t INF = 0x7f800000u;
    const uint32_t sign = bits & SIGN;
    uint32_t magnitude = bits & ~SIGN;
    if (magnitude == 0 || magnitude >= INF) {
        return bits;
    }
    int64_t encoded = magnitude;
    if (magnitude < HIDDEN) {
        int32_t shift = 0;
        while (magnitude < HIDDEN) {
            magnitude <<= 1;
            ++shift;
        }
        encoded = (int64_t(1) - shift) * HIDDEN + (magnitude & MANTISSA);
    }
    encoded += n;
    if (encoded >= INF) {
        return sign | INF;
    }
    if (encoded >= HIDDEN) {
        return sign | static_cast<uint32_t>(encoded);
    }
    // Floor division is explicit, including negative virtual biased exponents.
    int32_t exponent = encoded >= 0 ? int32_t(encoded / HIDDEN) : -int32_t((-encoded + MANTISSA) / HIDDEN);
    uint32_t shift = uint32_t(1 - exponent);
    if (shift > SIGNIFICAND_BITS) {
        return sign;
    }
    uint32_t significand = HIDDEN | (static_cast<uint32_t>(encoded) & MANTISSA);
    uint32_t value = significand >> shift;
    uint32_t remainder = significand & ((uint32_t(1) << shift) - 1);
    uint32_t halfway = uint32_t(1) << (shift - 1);
    value += (remainder > halfway || (remainder == halfway && (value & 1))) ? 1 : 0;
    return sign | value;
}

} // namespace Catlass::Epilogue::Tile
#endif
