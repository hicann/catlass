/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_TUNER_UTIL_H
#define CATLASS_TUNER_UTIL_H

#include <mutex>
#include <vector>
#include <algorithm>

namespace Catlass {

template<typename T>
class MTVar {
public:
    using Type = T;
    explicit MTVar() { var_ = T{}; }
    explicit MTVar(T &t) : var_(t) {}
    explicit MTVar(T &&t) : var_(std::move(t)) {}

    MTVar(const MTVar &mtVar) = delete;
    MTVar& operator=(const MTVar &mtVar) = delete;

    inline const T Get()
    {
        std::lock_guard<decltype(mtx_)> lockGuard(mtx_);
        return var_;
    }

    operator const T()
    {
        return Get();
    }

    const T operator=(const T &t)
    {
        Set(t);
        return t;
    }

    inline void Set(const T &t)
    {
        std::lock_guard<decltype(mtx_)> lockGuard(mtx_);
        var_ = t;
    }

    template<typename RetType>
    RetType DoTransaction(const std::function<RetType(Type&)> &callee)
    {
        std::lock_guard<decltype(mtx_)> lockGuard(mtx_);
        return callee(var_);
    }

private:
    T var_;
    std::mutex mtx_;
};

template <class Element, class ElementRandom>
inline void FillRandomData(std::vector<Element>& data, ElementRandom low, ElementRandom high)
{
    for (uint64_t i = 0; i < data.size(); ++i) {
        ElementRandom randomValue = low +
            (static_cast<ElementRandom>(rand()) / static_cast<ElementRandom>(RAND_MAX)) * (high - low);
        data[i] = static_cast<Element>(randomValue);
    }
}

template <>
inline void FillRandomData<int8_t, int>(std::vector<int8_t>& data, int low, int high)
{
    for (uint64_t i = 0; i < data.size(); ++i) {
        int randomValue = low + rand() % (high - low + 1);
        data[i] = static_cast<int8_t>(randomValue);
    }
}

template<typename T>
inline bool SafeMul(const std::vector<T>& numbers, uint64_t &product)
{
    if (numbers.empty()) {
        product = 0;
        return true;
    }
    product = 1;
    constexpr uint64_t max_uint64 = std::numeric_limits<uint64_t>::max();
    for (T num : numbers) {
        if (num == 0) {
            product = 0;
            return true;
        }
        if (product > max_uint64 / num) {
            return false;
        }
        product *= num;
    }
    return true;
}

// erase first num element of vec
template<typename T>
inline void Erase(std::vector<T> &vec, typename std::vector<T>::size_type num)
{
    if (num >= vec.size()) {
        vec.clear();
        return;
    }
    std::vector<T> tmp(vec.begin() + num, vec.end());
    vec.swap(tmp);
}

} // namespace Catlass
#endif // CATLASS_TUNER_UTIL_H