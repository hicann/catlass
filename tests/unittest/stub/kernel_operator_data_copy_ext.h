/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_STUB_KERNEL_OPERATOR_DATA_COPY_EXT_H
#define ASCENDC_STUB_KERNEL_OPERATOR_DATA_COPY_EXT_H

#include "kernel_tensor.h"
#include "kernel_struct_mm.h"
#include "ascendc_logger.h"
#include "arg.h"

namespace AscendC {

struct DataCopyExtParams {
#if defined(__NPU_ARCH__) && __NPU_ARCH == 3510
    using StrideType = int64_t;
#else
    using StrideType = uint32_t;
#endif
    uint16_t blockCount = 0;
    uint32_t blockLen = 32;
    StrideType srcStride = 0;
    StrideType dstStride = 0;
    uint8_t rsv = 1;

    std::string toString() const
    {
        char buffer[256];

        snprintf(
            buffer,
            sizeof(buffer),
            "DataCopyExtParams(blockCount=%u, blockLen=%u, srcStride=%lld, dstStride=%lld, rsv=%u)",
            static_cast<unsigned>(blockCount),
            static_cast<unsigned>(blockLen),
            static_cast<long long>(srcStride),
            static_cast<long long>(dstStride),
            static_cast<unsigned>(rsv)
        );

        return std::string(buffer);
    }
};

enum class CopyMode
{
    COPY_NONE = 0,
    COPY_D2D = 1,
    COPY_D2H = 2,
    COPY_H2D = 3
};

enum class PadMode
{
    PAD_NONE = 0,
    PAD_NZ = 1,
    PAD_N1 = 2,
    PAD_NZ2 = 3
};

struct PadParams {
    uint32_t padValue = 0;
    PadMode padMode = PadMode::PAD_NONE;
    uint32_t padN = 0;
    uint32_t padC = 0;
    uint32_t padH = 0;
    uint32_t padW = 0;
    uint32_t srcN = 0;
    uint32_t srcC = 0;
    uint32_t srcH = 0;
    uint32_t srcW = 0;
    uint32_t dstN = 0;
    uint32_t dstC = 0;
    uint32_t dstH = 0;
    uint32_t dstW = 0;

    std::string toString() const
    {
        char buffer[256];
        snprintf(
            buffer, sizeof(buffer), "PadParams(padMode=%d, padValue=%u, srcN=%u, srcH=%u, srcW=%u)", (int)padMode,
            padValue, srcN, srcH, srcW);
        return std::string(buffer);
    }
};

template <typename T>
inline void DataCopyPad(const LocalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyParams& params)
{
    const std::vector<Arg> argsT = {{Arg::MakeArg<T>()}};
    const std::vector<Arg> args = {dst, src, params};
    ASCENDC_LOG_CALL_T(DataCopyPad, argsT, args);
}

template <typename T>
inline void DataCopyPad(const GlobalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyParams& params)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, params};
    ASCENDC_LOG_CALL_T(DataCopyPad, argsT, args);
}

template <typename T>
inline void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyParams& params)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, params};
    ASCENDC_LOG_CALL_T(DataCopyPad, argsT, args);
}

template <typename T>
inline void DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyParams& params)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, params};
    ASCENDC_LOG_CALL_T(DataCopyPad, argsT, args);
}

template <typename T>
inline void DataCopyPad(
    const LocalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyParams& params, const PadParams& padParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, params, padParams};
    ASCENDC_LOG_CALL_T(DataCopyPad, argsT, args);
}

template <typename T>
inline void DataCopyPad(
    const GlobalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyParams& params, const PadParams& padParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, params, padParams};
    ASCENDC_LOG_CALL_T(DataCopyPad, argsT, args);
}

template <typename T>
inline void DataCopyPad(
    const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyParams& params, const PadParams& padParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, params, padParams};
    ASCENDC_LOG_CALL_T(DataCopyPad, argsT, args);
}

template <typename T>
inline void DataCopyPad(
    const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyParams& params, const PadParams& padParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, params, padParams};
    ASCENDC_LOG_CALL_T(DataCopyPad, argsT, args);
}

template <typename T>
inline void Copy(
    const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask, const uint8_t repeatTimes,
    const void* repeatParams)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, mask, repeatTimes, repeatParams};
    ASCENDC_LOG_CALL_T(Copy, argsT, args);
}

template <typename T>
inline void Copy(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t& count)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, count};
    ASCENDC_LOG_CALL_T(Copy, argsT, args);
}

enum class FmatrixMode
{
    FMATRIX_LEFT = 0,
    FMATRIX_RIGHT = 1,
    FMATRIX_LEFT_N = 2,
    FMATRIX_RIGHT_N = 3
};

inline void SetFmatrix(uint32_t m, uint32_t k, const uint16_t* padList, FmatrixMode mode)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<uint32_t>(), Arg::MakeArg<uint32_t>(), Arg::MakeArg<FmatrixMode>()};
    const std::vector<Arg> args = {m, k, padList, mode};
    ASCENDC_LOG_CALL_T(SetFmatrix, argsT, args);
}

enum class BrcbMode
{
    BRCB_NORMAL = 0,
    BRCB_NZ = 1
};

inline void Brcb(
    const LocalTensor<uint16_t>& dst, const LocalTensor<uint16_t>& src, const uint32_t& n, const uint32_t& c,
    const uint32_t& h, BrcbMode mode)
{
    const std::vector<Arg> args = {dst, src, n, c, h, mode};
    ASCENDC_LOG_CALL(Brcb, args);
}

} // namespace AscendC

#endif