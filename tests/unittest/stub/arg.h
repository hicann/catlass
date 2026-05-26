/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_STUB_ARG_H
#define ASCENDC_STUB_ARG_H

#include <typeindex>
#include <memory>
#include <type_traits>
template <typename T>
struct ArgUseRef : std::false_type {};
struct Arg {
    std::type_index type = typeid(void);

    void* value = nullptr;
    std::shared_ptr<void> holder = nullptr;

    Arg() = default;

    template <typename T>
    Arg(const T& v)
    {
        using RawT = std::remove_cv_t<std::remove_reference_t<T>>;
        type = typeid(RawT);

        if constexpr (!ArgUseRef<RawT>::value) {
            holder = std::make_shared<RawT>(v);
            value = holder.get();
        } else {
            value = reinterpret_cast<void*>(const_cast<RawT*>(&v));
        }
    }
    std::type_index Type() const
    {
        return type;
    }

    template <typename T = void>
    T* Value() const
    {
        return reinterpret_cast<T*>(value);
    }

    void* RawValue() const
    {
        return value;
    }

    // ========================
    // 类型参数
    // ========================
    template <typename T>
    static Arg MakeArg()
    {
        Arg arg;
        arg.type = typeid(T);
        return arg;
    }

    // ========================
    // 值参数
    // ========================
    template <typename T>
    static Arg MakeArgWithValue(const T& argValue)
    {
        return Arg(argValue);
    }
};

#endif