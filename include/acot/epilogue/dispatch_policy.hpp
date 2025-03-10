/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_EPILOGUE_DISPATCH_POLICY_HPP
#define ACOT_EPILOGUE_DISPATCH_POLICY_HPP

#include "acot/arch/arch.hpp"

namespace acot::epilogue
{

    // For AtlasA2, an element wise epilogue of the form D = C + X, where X is an additional source
    struct EpilogueAtlasA2ElemWiseOneSource
    {
        using ArchTag = arch::AtlasA2;
        // Number of operands. Including C, X, and D 3 operands
        static constexpr uint32_t OPERANDS_NUM = 3;
    };

    // AscendC910B3的Epiligue部分
    // struct EpilogueAscendC910B3Gemm {
    //     using ArchTag = arch::AscendC910B3;
    //     // 默认开启双缓冲
    //     static constexpr uint32_t STAGES = 2;
    //     // 设计三个数据处理 a*b 和 c  最后d
    //     static constexpr uint32_t OPERANDS_NUM = 3;
    // };

    struct EpilogueAtlasA2Gemv
    {
        using ArchTag = arch::AtlasA2;
        // Number of operands. Including y, Temp, and z 3 operands
        static constexpr uint32_t OPERANDS_NUM = 3;

        // 默认开启双缓冲
        static constexpr uint32_t STAGES = 2;
    };

    // For AtlasA2, FA Softmax
    struct EpilogueAtlasA2FASoftmax
    {
        using ArchTag = arch::AtlasA2;
    };

    // For AtlasA2, FA RescaleO
    struct EpilogueAtlasA2FARescaleO
    {
        using ArchTag = arch::AtlasA2;
    };

} // namespace acot::epilogue

#endif // ACOT_EPILOGUE_DISPATCH_POLICY_HPP
