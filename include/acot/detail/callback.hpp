/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_DETAIL_CALLBACK_HPP
#define ACOT_DETAIL_CALLBACK_HPP

#include "acot/detail/macros.hpp"

/// @brief Callback is an alternative to std::function<void(void)>, providing a general carrier
/// of callable structure with no parameters and no return value. Compared with function pointers
/// of type void (*)(), Callback can carry lambda expressions with captures, and does not need to
/// pay attention to the captured content. It should be noted that Callback itself does not store
/// the callable structure it carries like std::function<void(void)>, so it is necessary to ensure
/// that it is used within the life cycle of the callable structure.
struct Callback {
    void const *func{nullptr};
    void (*caller)(void const *){nullptr};

    Callback() = default;

    ACOT_DEVICE
    void operator()() const
    {
        caller(func);
    }

    ACOT_DEVICE
    operator bool() const
    {
        return func != nullptr;
    }
};

template <typename Func>
ACOT_DEVICE
static void FuncWrapper(void const *func)
{
    (*static_cast<Func const *>(func))();
}

// Use this to make a callback
template <typename Func>
ACOT_DEVICE
Callback MakeCallback(Func *func)
{
    Callback callback;
    callback.func = func;
    callback.caller = &FuncWrapper<Func>;
    return callback;
}

#endif // ACOT_DETAIL_CALLBACK_HPP
