/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACT_MATMUL_DEVICE_KERNEL_ADAPTER_HPP
#define ACT_MATMUL_DEVICE_KERNEL_ADAPTER_HPP

#include "act/act.hpp"

namespace Act {
/// Generic Act kernel template
template <class Operator>
ACT_GLOBAL void KernelAdapter(typename Operator::Params params)
{
    Operator op;
    op(params);
}

template <class Operator>
ACT_GLOBAL void KernelAdapter(typename Operator::Params params, uint64_t fftsAddr)
{
    AscendC::SetSyncBaseAddr(fftsAddr);
    Operator op;
    op(params);
}
} // namespace Act

#endif
