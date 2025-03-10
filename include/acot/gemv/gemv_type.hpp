/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_GEMV_GEMV_TYPE_HPP
#define ACOT_GEMV_GEMV_TYPE_HPP

namespace acot::gemv
{
    template <class Element_, class Layout_, AscendC::TPosition POSITION_ = AscendC::TPosition::GM>
    struct GemvType
    {
        using Element = Element_;
        using Layout = Layout_;
        static constexpr AscendC::TPosition POSITION = POSITION_;
    };

} // namespace acot::gemv

#endif // ACOT_GEMV_GEMV_TYPE_HPP