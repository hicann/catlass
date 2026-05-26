/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_STUB_KERNEL_OPERATOR_MM_INTF_H
#define ASCENDC_STUB_KERNEL_OPERATOR_MM_INTF_H

#include "kernel_tensor.h"
#include "kernel_struct_mm.h"
#include "ascendc_logger.h"

namespace AscendC {

template <typename T>
inline void LoadData(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LoadData2DParamsV2& params)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, params};
    ASCENDC_LOG_CALL_T(LoadData, argsT, args);       
}

template <typename T>
inline void LoadData(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const LoadData2DParamsV2& params)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, params};
    ASCENDC_LOG_CALL_T(LoadData, argsT, args);
}

template <typename T>
inline void LoadData(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LoadData2DMxParams& params)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<Arg> args = {dst, src, params};
    ASCENDC_LOG_CALL_T(LoadData, argsT, args);  
}

template <typename T, typename U, typename S>
inline void Mmad(
    const LocalTensor<T>& dst, const LocalTensor<U>& fm, const LocalTensor<S>& filter, const MmadParams& params)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArg<U>(), Arg::MakeArg<S>()};
    const std::vector<Arg> args = {dst, fm, filter, params};
    ASCENDC_LOG_CALL_T(Mmad, argsT, args);
}

template <typename T, typename U, typename S, typename V>
inline void Mmad(
    const LocalTensor<T>& dst, const LocalTensor<U>& fm, const LocalTensor<S>& filter, const LocalTensor<V>& bias,
    const MmadParams& params)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>(), Arg::MakeArg<U>(), Arg::MakeArg<S>(), Arg::MakeArg<V>()};
    const std::vector<Arg> args = {dst, fm, filter, bias, params};
    ASCENDC_LOG_CALL_T(Mmad, argsT, args);
}

} // namespace AscendC

#endif // ASCENDC_STUB_KERNEL_OPERATOR_MM_INTF_H
