/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_TEST_COMMON_HPP
#define CATLASS_TEST_COMMON_HPP

// AscendCL headers

#ifdef ASCENDC_MODULE_OPERATOR_H
#undef inline
#endif

#include <acl/acl.h>
#include <runtime/rt_ffts.h>
#include <tiling/platform/platform_ascendc.h>

#ifdef ASCENDC_MODULE_OPERATOR_H
#define inline __inline__ __attribute__((always_inline))
#endif

// Macro function for unwinding acl errors.
#define ACL_CHECK(status)              \
    do {                               \
        aclError error = status;       \
        if (error != ACL_ERROR_NONE) { \
            return error;              \
        }                              \
    } while (0)

// Macro function for unwinding rt errors.
#define RT_CHECK(status)              \
    do {                              \
        rtError_t error = status;     \
        if (error != RT_ERROR_NONE) { \
            return error;             \
        }                             \
    } while (0)

#define RUN_ADAPTER(op, args, stream, coreNum)                                                                         \
    size_t sizeWorkspace = op.GetWorkspaceSize(args);                                                                  \
    uint8_t *deviceWorkspace = nullptr;                                                                                \
    if (sizeWorkspace > 0) {                                                                                           \
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)); \
    }                                                                                                                  \
    op.Initialize(args, deviceWorkspace);                                                                              \
    op(stream, coreNum);                                                                                               \
    ACL_CHECK(aclrtSynchronizeStream(stream));                                                                         \
    if (sizeWorkspace > 0) {                                                                                           \
        ACL_CHECK(aclrtFree(deviceWorkspace));                                                                         \
    }

#define RUN_ADAPTER_MIX(op, args, stream, coreNum)                                                                     \
    uint32_t fftsLen = 0;                                                                                              \
    uint64_t fftsAddr = 0;                                                                                             \
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));                                                                   \
    size_t sizeWorkspace = matmul_op.GetWorkspaceSize(args);                                                           \
    uint8_t *deviceWorkspace = nullptr;                                                                                \
    if (sizeWorkspace > 0) {                                                                                           \
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST)); \
    }                                                                                                                  \
    op.Initialize(args, deviceWorkspace);                                                                              \
    op(stream, coreNum, fftsAddr);                                                                                     \
    ACL_CHECK(aclrtSynchronizeStream(stream));                                                                         \
    if (sizeWorkspace > 0) {                                                                                           \
        ACL_CHECK(aclrtFree(deviceWorkspace));                                                                         \
    }

#endif