/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_STUB_KERNEL_STRUCT_MM_H
#define ASCENDC_STUB_KERNEL_STRUCT_MM_H

#include <cstdio>
#include <string>
#include <cstdint>

namespace AscendC {

struct MmadParams {
    uint16_t m = 0;
    uint16_t n = 0;
    uint16_t k = 0;
    bool isBias = false;
    int32_t fmOffset = 0;
    bool enSsparse = false;
    bool enWinogradA = false;
    bool enWinogradB = false;
    uint8_t unitFlag = 0;
    bool kDirectionAlign = false;
    bool cmatrixSource = false;
    bool cmatrixInitVal = true;
    
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
    bool disableGemv = false;
#endif
    
    std::string toString() const {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "MmadParams(m=%u, n=%u, k=%u, unitFlag=%u, cmatrixInitVal=%s)",
                m, n, k, unitFlag, cmatrixInitVal ? "true" : "false");
        return std::string(buffer);
    }
};

struct LoadData2DParamsV2 {
    uint32_t mStartPosition = 0;
    uint32_t kStartPosition = 0;
    uint16_t mStep = 0;
    uint16_t kStep = 0;
    int32_t srcStride = 0;
    uint16_t dstStride = 0;
    bool ifTranspose = false;
    uint8_t sid = 0;
    
    std::string toString() const {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "LoadData2DParamsV2(mPos=%u, kPos=%u, mStep=%u, kStep=%u)",
                mStartPosition, kStartPosition, mStep, kStep);
        return std::string(buffer);
    }
};

struct LoadData2DMxParams {
    uint32_t mStartPosition = 0;
    uint32_t kStartPosition = 0;
    uint16_t mStep = 0;
    uint16_t kStep = 0;
    int32_t srcStride = 0;
    uint16_t dstStride = 0;
    bool ifTranspose = false;
    uint8_t sid = 0;
    uint16_t scaleMStep = 0;
    uint16_t scaleKStep = 0;
    
    std::string toString() const {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "LoadData2DMxParams(mPos=%u, kPos=%u, mStep=%u, kStep=%u)",
                mStartPosition, kStartPosition, mStep, kStep);
        return std::string(buffer);
    }
};

struct DataCopyParams {
    uint16_t blockCount = 0;
    uint16_t blockLen = 0;
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;
    
    std::string toString() const {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "DataCopyParams(blockCount=%u, blockLen=%u)",
                blockCount, blockLen);
        return std::string(buffer);
    }
};

struct Nd2NzParams {
    uint32_t ndPos = 0;
    uint32_t srcNdMatrixStride = 0;
    uint32_t srcDValue = 0;
    uint32_t dstNzMatrixStride = 0;
    uint32_t dstNzC0Stride = 0;
    uint32_t dstNzCNiStride = 0;
    uint32_t dstNzNiStride = 0;
    uint32_t dstNzC1Stride = 0;
    uint32_t dstNzC1Size = 0;
    uint32_t srcNdMatrixStrideV = 0;
    uint32_t srcDValueV = 0;
    uint32_t dstNzMatrixStrideV = 0;
    uint32_t dstNzC0StrideV = 0;
    uint32_t dstNzCNiStrideV = 0;
    uint32_t dstNzNiStrideV = 0;
    uint32_t dstNzC1StrideV = 0;
    uint32_t dstNzC1SizeV = 0;
    
    std::string toString() const {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "Nd2NzParams(ndPos=%u, dstNzC1Size=%u)",
                ndPos, dstNzC1Size);
        return std::string(buffer);
    }
};

struct Dn2NzParams {
    uint32_t dnPos = 0;
    uint32_t srcDnMatrixStride = 0;
    uint32_t srcDValue = 0;
    uint32_t dstNzMatrixStride = 0;
    uint32_t dstNzC0Stride = 0;
    uint32_t dstNzCNiStride = 0;
    uint32_t dstNzNiStride = 0;
    uint32_t dstNzC1Stride = 0;
    uint32_t dstNzC1Size = 0;
    
    std::string toString() const {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "Dn2NzParams(dnPos=%u, dstNzC1Size=%u)",
                dnPos, dstNzC1Size);
        return std::string(buffer);
    }
};

}  // namespace AscendC

#endif  // ASCENDC_STUB_KERNEL_STRUCT_MM_H
