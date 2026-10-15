/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details in the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_MATMUL_PERMUTE_POLICY_HPP
#define CATLASS_EPILOGUE_MATMUL_PERMUTE_POLICY_HPP

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"

namespace Catlass::Epilogue {

template <uint32_t S1_, uint32_t S2_, bool SPLIT_M_, bool USE_VECTOR_PACK_ = false, uint32_t UB_STAGES_ = 2>
struct EpilogueAscend950MatmulPermute {
    using ArchTag = Arch::Ascend950;
    static constexpr uint32_t S1 = S1_;
    static constexpr uint32_t S2 = S2_;
    static constexpr bool SPLIT_M = SPLIT_M_;
    static constexpr bool USE_VECTOR_PACK = USE_VECTOR_PACK_;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

} // namespace Catlass::Epilogue

#endif // CATLASS_EPILOGUE_MATMUL_PERMUTE_POLICY_HPP
