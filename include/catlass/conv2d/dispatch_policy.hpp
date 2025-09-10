/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV2D_DISPATCH_POLICY_HPP
#define CATLASS_CONV2D_DISPATCH_POLICY_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"

namespace Catlass::Conv2d {

// Block Mmad Policies

template <bool ASYNC_ = false>
struct MmadAtlasA2Base {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t ASYNC = ASYNC_;
};

using MmadAtlasA2 = MmadAtlasA2Base<false>;
using MmadAtlasA2Async = MmadAtlasA2Base<true>;

// Now ENABLE_UNIT_FLAG_ must be false when intput element is int8
template <bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2Pingpong : public MmadAtlasA2  {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

}  // namespace Catlass::Conv2d

#endif  // CATLASS_GEMM_DISPATCH_POLICY_HPP
