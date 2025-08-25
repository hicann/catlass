/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_TUNER_DEVICE_MEMORY_MANAGER_H
#define CATLASS_TUNER_DEVICE_MEMORY_MANAGER_H

#include <memory>
#include <vector>
#include <algorithm>
#include <cstdint>
#include "log.h"
#include "util.h"

#include <acl/acl.h>
#include <runtime/rt_ffts.h>

namespace Catlass {

#define ACL_CHECK(status, func)                                                 \
    do {                                                                        \
        aclError err = status;                                                  \
        if (err != ACL_SUCCESS) {                                               \
            LOGE("%s:%d call " func "failed: %d", __FILE__, __LINE__, err);     \
        }                                                                       \
    } while (0)

// Macro function for unwinding rt errors.
#define RT_CHECK(status, func)                                                               \
    do {                                                                                     \
        rtError_t error = status;                                                            \
        if (error != RT_ERROR_NONE) {                                                        \
            LOGE("%s:%d call " func " rtError: %d", __FILE__, __LINE__, error);              \
        }                                                                                    \
    } while (0)

struct DeviceMemoryParam {
    void **addr;
    size_t size;
};

class DeviceMemoryManager {
public:
    static DeviceMemoryManager& Instance()
    {
        static DeviceMemoryManager t;
        return t;
    }

    DeviceMemoryManager(const DeviceMemoryManager&) = delete;
    DeviceMemoryManager& operator=(const DeviceMemoryManager&) = delete;
    DeviceMemoryManager(DeviceMemoryManager&&) = delete;
    DeviceMemoryManager& operator=(DeviceMemoryManager&&) = delete;

    inline uint64_t GetFftsAddr()
    {
        uint32_t fftsLen{0};
        if (fftsAddr_ == 0) {
            RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr_, &fftsLen), "rtGetC2cCtrlAddr");
        }
        return fftsAddr_;
    }

    inline bool FillDeviceData(void* dst, size_t size, void* host) const
    {
        auto d = reinterpret_cast<uint64_t>(dst);
        auto addr = reinterpret_cast<uint64_t>(arg_);
        auto addr2 = reinterpret_cast<uint64_t>(workspace_);
        if (!((d >= addr && d + size <= addr + argSize_) || (d >= addr2 && d + size <= addr2 + workspaceSize_))) {
            LOGE("Try to copy host data to invalid addr 0x%lx, size %lu", d, size);
            return false;
        }
        auto err = aclrtMemcpyAsync(dst, size, host, size, ACL_MEMCPY_HOST_TO_DEVICE, stream_);
        if (err != ACL_SUCCESS) {
            LOGE("Fill device data failed when call aclrtMemcpyAsync, err: %d", err);
            return false;
        }
        err = aclrtSynchronizeStream(stream_);
        if (err != ACL_SUCCESS) {
            LOGE("Fill device data failed when call aclrtSynchronizeStream, err: %d", err);
            return false;
        }
        return true;
    }

    inline bool FreeWorkspace()
    {
        if (Free(workspace_)) {
            workspace_ = nullptr;
            workspaceSize_ = 0;
            return true;
        }
        return false;
    }

    aclrtStream Initialize(int32_t deviceId);
    void Finalize();
    bool MallocArguments(std::vector<DeviceMemoryParam> &params);
    bool MallocWorkspace(DeviceMemoryParam &param);
    bool ClearL2Cache(uint32_t blockDim);
    bool InitCacheClear();

private:
    struct CacheClear {
        void *buffer{nullptr};
        void *tilingSize{nullptr};
        void *flushBuffer{nullptr};
        std::vector<void*> cmoBuffers{};
        uint64_t cacheSize{};
    };

    DeviceMemoryManager() = default;
    ~DeviceMemoryManager()
    {
        Finalize();
    }

    inline uint64_t Align(uint64_t size) const { return (size + 31) / 32 * 32; }

    bool Expand(void** addr, uint64_t &size, uint64_t target);
    bool Free(void* addr);

    void *arg_{nullptr};
    uint64_t argSize_{0};
    void *workspace_{nullptr};
    uint64_t workspaceSize_{0};
    aclrtStream stream_{nullptr};
    CacheClear cacheClear_{};
    uint64_t fftsAddr_{0};
    int32_t deviceId_{0};
};

} // namespace Catlass
#endif // CATLASS_TUNER_DEVICE_MEMORY_MANAGER_H