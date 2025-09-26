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

#pragma push_macro("inline")
#include <cstdio>
#include <fstream>
#include <iostream>

#include <acl/acl.h>
#include <opdev/bfloat16.h>
#include <opdev/fp16_t.h>
#include <runtime/rt_ffts.h>
#include <tiling/platform/platform_ascendc.h>
#pragma pop_macro("inline")

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
 * Function for reading a file.
 */
bool ReadFile(const std::string& filePath, void* buffer, size_t bufferSize)
{
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
    std::filebuf* buf = fd.rdbuf();
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
    buf->sgetn(static_cast<char*>(buffer), size);
    return true;
}

/**
 * Function for reading file to a vector.
 */
template <typename T>
bool ReadFileToVector(const std::string &filePath, std::vector<T> &vec)
{
    return ReadFile(filePath, vec.data(), vec.size() * sizeof(T));
}

#endif  // EXAMPLES_COMMON_HELPER_HPP
