/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_DATATYPE_HPP
#define CATLASS_DATATYPE_HPP

#include <type_traits>

using namespace AscendC;
namespace Catlass {

template <typename T, typename Element>
CATLASS_HOST_DEVICE 
static constexpr T mulSizeofElement(T multiplier)
{
    if constexpr (std::is_same_v<int4b_t, Element>)
    {
        return static_cast<T>(multiplier / 2);
    }
    else
    {
        return static_cast<T>(multiplier * sizeof(Element));
    }
}

template <typename T, typename Element>
CATLASS_HOST_DEVICE 
static constexpr T divElement(T dividend)
{
    if constexpr (std::is_same_v<int4b_t, Element>)
    {
        return static_cast<T>(dividend * 2);
    }
    else
    {
        return static_cast<T>(dividend / sizeof(Element));
    }
}

} // namespace Catlass

#endif // CATLASS_DATATYPE_HPP