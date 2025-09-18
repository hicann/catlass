/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_TEST_MACROS_HPP
#define CATLASS_TEST_MACROS_HPP

#include <cstdint>

#include <catlass/layout/layout.hpp>
using namespace Catlass;

// for code intellcense
#ifndef __CCE__
#define half int16_t
#define bfloat16_t int16_t
#endif

#ifndef ELEMENT_A
#define ELEMENT_A half
#endif

#ifndef LAYOUT_A
#define LAYOUT_A layout::RowMajor
#endif

#ifndef ELEMENT_B
#define ELEMENT_B half
#endif

#ifndef LAYOUT_B
#define LAYOUT_B layout::RowMajor
#endif

#ifndef ELEMENT_C
#define ELEMENT_C half
#endif

#ifndef LAYOUT_C
#define LAYOUT_C layout::RowMajor
#endif

#endif