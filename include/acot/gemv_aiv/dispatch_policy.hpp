/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_GEMV_DISPATCH_POLICY_HPP
#define ACOT_GEMV_DISPATCH_POLICY_HPP

#include "acot/acot.hpp"

namespace acot::gemv {

// Block Mmad Policies

template <bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2Pingpong {
    using ArchTag = arch::AtlasA2;
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

template <bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
struct MmadAtlasA2Preload {
    using ArchTag = arch::AtlasA2;
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

struct MmadAtlasA2FAQK {
    using ArchTag = arch::AtlasA2;
    static constexpr uint32_t STAGES = 2;
};

struct MmadAtlasA2FAPV {
    using ArchTag = arch::AtlasA2;
    static constexpr uint32_t STAGES = 2;
};

}  // namespace acot::matmul

#endif  // ACOT_MATMUL_DISPATCH_POLICY_HPP
