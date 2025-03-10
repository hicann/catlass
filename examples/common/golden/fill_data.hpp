/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef EXAMPLES_COMMON_GOLDEN_FILL_DATA_HPP
#define EXAMPLES_COMMON_GOLDEN_FILL_DATA_HPP

#include <vector>
#include <cstdlib>
#include <ctime>

namespace acot::golden
{

    template <class Element>
    void FillRandomData(std::vector<Element> &data, float low, float high, uint64_t seed = time(0))
    {
        srand(seed);
        for (uint64_t i = 0; i < data.size(); ++i)
        {
            Element randomValue = static_cast<Element>(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            data[i] = low + randomValue * (high - low);
        }
    }

} // namespace acot::golden

#endif // EXAMPLES_COMMON_GOLDEN_FILL_DATA_HPP
