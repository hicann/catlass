/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATLASS_GEMM_DEVICE_KERNEL_ADAPTER_HPP
#define CATLASS_GEMM_DEVICE_KERNEL_ADAPTER_HPP

#include "catlass/catlass.hpp"

#if defined(ASCENDC_DUMP) && ASCENDC_DUMP == 1
#include "catlass/debug.hpp"
#endif

namespace Catlass {
/// Generic Catlass kernel template
template <class Operator>
// 支持cube核和vector核的比例控制, 注释下行可使能, 仅支持1:0, 1:1, 1:2
// [[bisheng::core_ratio(1, 2)]] 
CATLASS_GLOBAL void KernelAdapter(typename Operator::Params params, GM_ADDR ptrDump = nullptr)
{
    Operator op;
#if defined(ASCENDC_DUMP) && ASCENDC_DUMP == 1
    AscendC::InitDump(false, ptrDump, ALL_DUMPSIZE);
#endif
    op(params);
}

template <class Operator>
// [[bisheng::core_ratio(1, 0)]] 
CATLASS_GLOBAL void KernelAdapter(typename Operator::Params params, uint64_t fftsAddr, GM_ADDR ptrDump = nullptr)
{
    AscendC::SetSyncBaseAddr(fftsAddr);
    Operator op;
#if defined(ASCENDC_DUMP) && ASCENDC_DUMP == 1
    AscendC::InitDump(false, ptrDump, ALL_DUMPSIZE);
#endif
    op(params);
}
} // namespace Catlass

#endif
