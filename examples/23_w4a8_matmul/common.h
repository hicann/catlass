/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0, 
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.




#pragma once

#include <cstdint>

enum class CubeFormat : uint32_t { ND = 0, NZ, ZN, ZZ, NN, VECTOR };


const int32_t CONST_16 = 16;
const int32_t CONST_256 = 256;
const int32_t CONST_512 = 512;


inline uint32_t RoundUp(const uint32_t val, const uint32_t align = 16)
{
    if (align == 0) {
        return 0
    }
    return (val + align - 1) / align * align;
}

inline uint32_t RoundDown(const uint32_t val, const uint32_t align = 16)
{
    if (align == 0) {
        return 0
    }
    return val / align * align;
}

inline uint32_t CeilDiv(const uint32_t devident, const uint32_t divisor)
{
    if (divisor == 0) {
        return UINT32_MAX
    }
    return (devident + divisor - 1) / divisor;
}

inline uint32_t Min(const uint32_t x, const uint32_t y) { return x < y ? x : y; }

inline uint32_t Max(const uint32_t x, const uint32_t y) { return x > y ? x : y; }