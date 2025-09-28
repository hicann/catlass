/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXAMPLES_COMMON_HELPER_HPP
#define EXAMPLES_COMMON_HELPER_HPP

#ifdef ASCENDC_MODULE_OPERATOR_H
#undef inline
#endif
#include <cstdio>
#include <fstream>
#include <iostream>

#include <acl/acl.h>
#include <opdev/bfloat16.h>
#include <opdev/fp16_t.h>
#include <runtime/rt_ffts.h>
#include <tiling/platform/platform_ascendc.h>

#ifdef ASCENDC_MODULE_OPERATOR_H
#define inline __inline__ __attribute__((always_inline))
#endif

#include "catlass/layout/layout.hpp"

#include "options.hpp"

using op::bfloat16;
using op::fp16_t;

// Macro function for unwinding acl errors.
#define ACL_CHECK(status)                                                                                              \
    do {                                                                                                               \
        aclError error = status;                                                                                       \
        if (error != ACL_ERROR_NONE) {                                                                                 \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << error << std::endl;                            \
        }                                                                                                              \
    } while (0)

// Macro function for unwinding rt errors.
#define RT_CHECK(status)                                                                                               \
    do {                                                                                                               \
        rtError_t error = status;                                                                                      \
        if (error != RT_ERROR_NONE) {                                                                                  \
            std::cerr << __FILE__ << ":" << __LINE__ << " rtError:" << error << std::endl;                             \
        }                                                                                                              \
    } while (0)

/**
 * Function for read file.
 */
inline bool ReadFile(const std::string &filePath, void *buffer, size_t bufferSize) {
    if (buffer == nullptr) {
        printf("Read file %s failed. Buffer is nullptr.\n", filePath.c_str());
        return false;
    }

    // Open file
    std::ifstream fd(filePath, std::ios::binary);
    if (!fd) {
        printf("Open file failed. path = %s.\n", filePath.c_str());
        return false;
    }

    // Load file data in buffer
    std::filebuf *buf = fd.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        printf("File %s size is 0\n", filePath.c_str());
        return false;
    }
    if (size > bufferSize) {
        printf("File %s size is larger than buffer size.\n", filePath.c_str());
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    return true;
}

template <class Adapter>
inline void RunAdapter(Adapter opAdapter,
                       typename Adapter::Arguments args,
                       aclrtStream stream,
                       uint32_t coreNum,
                       uint64_t fftsAddr = 0) {
    size_t sizeWorkspace = opAdapter.GetWorkspaceSize(args);
    uint8_t *deviceWorkspace = nullptr;
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    opAdapter.Initialize(args, deviceWorkspace);
    opAdapter(stream, coreNum, fftsAddr);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    if (sizeWorkspace > 0) {
        ACL_CHECK(aclrtFree(deviceWorkspace));
    }
}

namespace Catlass {
inline bool IsNeedPadding(layout::RowMajor layout, uint32_t align) {
    // If the stride is greater than 65536, padding is required to reduce the stride.
    if (layout.stride(0) < 65536) {
        return layout.stride(0) % align != 0;
    } else {
        return true;
    }
}

inline bool IsNeedPadding(layout::ColumnMajor layout, uint32_t align) {
    // If the stride is greater than 65536, padding is required to reduce the stride.
    if (layout.stride(1) < 65536) {
        return layout.stride(1) % align != 0;
    } else {
        return true;
    }
}

inline bool IsNeedPadding(layout::zN layout, uint32_t align) { return false; }

inline bool IsNeedPadding(layout::nZ layout, uint32_t align) { return false; }
} // namespace Catlass

#endif // EXAMPLES_COMMON_HELPER_HPP
