/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_ARCH_ARCH_HPP
#define ACOT_ARCH_ARCH_HPP

namespace acot::arch {

struct AtlasA2 {
    static constexpr uint32_t BIAS_SIZE = 1024;
    static constexpr uint32_t FIXBUF_SIZE = 7 * 1024;
    static constexpr uint32_t UB_SIZE = 192 * 1024;
    static constexpr uint32_t UBIN_SIZE = 144 * 1024;
    static constexpr uint32_t UBOUT_SIZE = 16 * 1024;
    static constexpr uint32_t UBWS_SIZE = 32 * 1024;
    static constexpr uint32_t L1_SIZE = 512 * 1024;
    static constexpr uint32_t L0A_SIZE = 64 * 1024;
    static constexpr uint32_t L0B_SIZE = 64 * 1024;
    static constexpr uint32_t L0C_SIZE = 128 * 1024;
};

struct AscendC910B3{
    static uint32_t const BiasSize = 1024;
    static uint32_t const FixBSize = 7 * 1024;
    static uint32_t const UBSize = 192 * 1024;
    static constexpr uint32_t UBIN_SIZE = 144 * 1024;
    static constexpr uint32_t UBOUT_SIZE = 16 * 1024;
    static constexpr uint32_t UBWS_SIZE = 32 * 1024;
    
    static uint32_t const L1Size = 512 * 1024;
    static uint32_t const L0ASize = 64 * 1024;
    static uint32_t const L0BSize = 64 * 1024;
    static uint32_t const L0CSize = 128 * 1024;
    static uint32_t const MaxBlock = 20;
    static uint32_t const MaxAivBlock = 40;
    static uint32_t const L2CacheByte = 192 * 1024 * 1024;
};

} // namespace acot::arch

#endif // ACOT_ARCH_ARCH_HPP