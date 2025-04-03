/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef EXAMPLES_COMMON_GOLDEN_COMPARE_DATA_HPP
#define EXAMPLES_COMMON_GOLDEN_COMPARE_DATA_HPP

#include <cmath>
#include <vector>

#include "act/gemm_coord.hpp"

namespace Act::golden {

template<class ElementResult, class ElementCompare>
std::vector<uint64_t> CompareData(const std::vector<ElementResult>& result, const std::vector<ElementCompare>& expect,
    uint32_t computeNum)
{
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    for (uint64_t i = 0; i < result.size(); ++i) {
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = expect[i];
        ElementCompare diff = std::fabs(actualValue - expectValue);
        if (diff > rtol * std::max(1.0f, std::fabs(expectValue))) {
            errorIndices.push_back(i);
        }
    }
    return errorIndices;
}

// Compare for GroupedMatmul slicing M
template<class ElementResult, class ElementCompare>
std::vector<uint64_t> CompareData(const std::vector<ElementResult>& result, const std::vector<ElementCompare>& expect,
    uint32_t computeNum, uint32_t validNum)
{
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    for (uint64_t i = 0; i < validNum; ++i) {
        ElementCompare actualValue = static_cast<ElementCompare>(result[i]);
        ElementCompare expectValue = expect[i];
        ElementCompare diff = std::fabs(actualValue - expectValue);
        if (diff > rtol * std::max(1.0f, std::fabs(expectValue))) {
            errorIndices.push_back(i);
        }
    }
    return errorIndices;
}

// Compare for GroupedMatmul slicing K
template<class ElementResult, class ElementCompare, class T>
std::vector<uint64_t> CompareData(const std::vector<ElementResult>& result, const std::vector<ElementCompare>& expect,
    uint32_t computeNum, const std::vector<T>& groupList, uint32_t stride)
{
    const uint32_t computeNumThreshold = 2048;
    const float rtolGeneral = 1.0f / 256;
    const float rtolOverThreshold = 1.0f / 128;

    float rtol = computeNum < computeNumThreshold ? rtolGeneral : rtolOverThreshold;
    std::vector<uint64_t> errorIndices;
    T prevGroupValue = 0;
    uint64_t currentIndex = 0;
    for (const auto& groupValue : groupList) {
        if (groupValue == prevGroupValue) {
            currentIndex += stride;
            prevGroupValue = groupValue;
            continue;
        }
        for (uint64_t i = 0; i < stride; ++i) {
            if (currentIndex >= result.size()) break;
            ElementCompare actualValue = static_cast<ElementCompare>(result[currentIndex]);
            ElementCompare expectValue = expect[currentIndex];
            ElementCompare diff = std::fabs(actualValue - expectValue);
            if (diff > rtol * std::max(1.0f, std::fabs(expectValue))) {
                errorIndices.push_back(i);
            }
            currentIndex++;
        }
        prevGroupValue = groupValue;
    }
    return errorIndices;
}

}  // namespace Act::golden

#endif  // EXAMPLES_COMMON_GOLDEN_COMPARE_DATA_HPP
