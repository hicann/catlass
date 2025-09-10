/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#ifndef CATLASS_KERNEL_RUNNER_HPP
#define CATLASS_KERNEL_RUNNER_HPP

#include "catlass/catlass.hpp"

#if defined(ENABLE_ASCENDC_DUMP)
#include "catlass/debug.hpp"
#define KERNEL_RUNNER                                                          \
  static void RunKernel(typename Operator::Params params, GM_ADDR ptrDump) {         \
    AscendC::InitDump(false, ptrDump, ALL_DUMPSIZE);                           \
    Operator op;                                                               \
    op(params);                                                                \
  }
#define KERNEL_RUNNER_SYNC                                                     \
  static void RunKernel(typename Operator::Params params, uint64_t fftsAddr,         \
                  GM_ADDR ptrDump) {                                           \
    AscendC::SetSyncBaseAddr(fftsAddr);                                        \
    AscendC::InitDump(false, ptrDump, ALL_DUMPSIZE);                           \
    Operator op;                                                               \
    op(params);                                                                \
  }
#else
#define KERNEL_RUNNER                                                          \
  static void RunKernel(typename Operator::Params params) {                          \
    Operator op;                                                               \
    op(params);                                                                \
  }
#define KERNEL_RUNNER_SYNC                                                     \
  static void RunKernel(typename Operator::Params params, uint64_t fftsAddr) {       \
    AscendC::SetSyncBaseAddr(fftsAddr);                                        \
    Operator op;                                                               \
    op(params);                                                                \
  }
#endif

namespace Catlass {

using CatlassKernelType = KernelMetaType;
template <class Operator, CatlassKernelType Type>
struct KernelAdapter {};

template <class Operator>
struct KernelAdapter<Operator, CatlassKernelType::KERNEL_TYPE_AIV_ONLY> {
    CATLASS_GLOBAL_AIV KERNEL_RUNNER;
};
template <class Operator>
struct KernelAdapter<Operator, CatlassKernelType::KERNEL_TYPE_AIC_ONLY> {
    CATLASS_GLOBAL_AIC KERNEL_RUNNER;
}; // namespace Catlass

template <class Operator> struct KernelAdapter<Operator, CatlassKernelType::KERNEL_TYPE_MIX_AIC_1_0> {
    CATLASS_GLOBAL_MIX_C1V0 KERNEL_RUNNER_SYNC;
};
template <class Operator> struct KernelAdapter<Operator, CatlassKernelType::KERNEL_TYPE_MIX_AIC_1_1> {
    CATLASS_GLOBAL_MIX_C1V1 KERNEL_RUNNER_SYNC;
};
template <class Operator> struct KernelAdapter<Operator, CatlassKernelType::KERNEL_TYPE_MIX_AIC_1_2> {
    CATLASS_GLOBAL_MIX_C1V2 KERNEL_RUNNER_SYNC;
};
} // namespace Catlass

#endif
