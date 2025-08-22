/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHARED_LIB_COMMON_COMMON_HPP
#define SHARED_LIB_COMMON_COMMON_HPP

#include <acl/acl.h>

#include "catlass/layout/layout.hpp"
#include "catlass/debug.hpp"

namespace CatlassKernel {
using namespace Catlass;

template <aclDataType T>
struct AclType2Type;

template <>
struct AclType2Type<ACL_FLOAT> {
    using type = float;
};

template <>
struct AclType2Type<ACL_INT32> {
    using type = int32_t;
};

template <>
struct AclType2Type<ACL_INT8> {
    using type = int8_t;
};

template <>
struct AclType2Type<ACL_FLOAT16> {
    using type = half;
};

template <>
struct AclType2Type<ACL_BF16> {
    using type = bfloat16_t;
};

template <bool IS_TRANSPOSE>
struct Transpose2Layout;

template <>
struct Transpose2Layout<false> {
    using layout = Catlass::layout::RowMajor;
};
template <>
struct Transpose2Layout<true> {
    using layout = Catlass::layout::ColumnMajor;
};

template <class Adapter>
void RunAdapter(
    Adapter matmulOp, typename Adapter::Arguments args, aclrtStream stream, uint32_t aicCoreNum, uint64_t fftsAddr = 0)
{
    size_t sizeWorkspace = matmulOp.GetWorkspaceSize(args);
    uint8_t *deviceWorkspace = nullptr;
    if (sizeWorkspace > 0) {
        aclCheck(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    matmulOp.Initialize(args, deviceWorkspace);
    matmulOp(stream, aicCoreNum, fftsAddr);
    aclCheck(aclrtSynchronizeStream(stream));
    if (sizeWorkspace > 0) {
        aclCheck(aclrtFree(deviceWorkspace));
    }
}

inline bool IsNeedPadding(layout::RowMajor layout, uint32_t align)
{
    // If the stride is greater than 65536, padding is required to reduce the stride.
    if (layout.stride(0) < 65536) {
        return layout.stride(0) % align != 0;
    } else {
        return true;
    }
}

inline bool IsNeedPadding(layout::ColumnMajor layout, uint32_t align)
{
    // If the stride is greater than 65536, padding is required to reduce the stride.
    if (layout.stride(1) < 65536) {
        return layout.stride(1) % align != 0;
    } else {
        return true;
    }
}

inline bool IsNeedPadding(layout::zN layout, uint32_t align)
{
    return false;
}

inline bool IsNeedPadding(layout::nZ layout, uint32_t align)
{
    return false;
}

}  // namespace CatlassKernel
#endif