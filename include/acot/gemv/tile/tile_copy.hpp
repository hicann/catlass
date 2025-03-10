/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_GEMV_TILE_TILE_COPY_HPP
#define ACOT_GEMV_TILE_TILE_COPY_HPP

#include "acot/gemv/tile/copy_gm_to_l1.hpp"
#include "acot/gemv/tile/copy_l0c_to_gm.hpp"
#include "acot/gemv/tile/copy_l1_to_l0.hpp"
#include "acot/gemv/helper.hpp"

namespace acot::gemv::tile
{
    template <
        /// Tag indicating architecture
        class ArchTag,
        /// MatmulType for A matrix operand
        class xType,
        /// MatmulType type for B vector operand
        class AType,
        /// MatmulType type for C vector operand
        class yType,
        /// MatmulTpe type for Bias operand
        class BiasType = void>
    struct TileCopy
    {
        // 输入矩阵、向量，输出矩阵的数据类型
        using ElementA = typename AType::Element;
        using Elementx = typename xType::Element;
        using Elementy = typename yType::Element;
        using ElementAccumulator = typename gemv::helper::ElementAccumulatorSelector<ElementA, Elementx>::ElementAccumulator;

        // 搬运矩阵的相关函数的直接调用
        using CopyGmToL1A = gemv::tile::CopyGmToL1A<ArchTag, xType>;
        using CopyGmToL1B = gemv::tile::CopyGmToL1B<ArchTag, AType>;
        using CopyL1ToL0A = gemv::tile::CopyL1ToL0A<ArchTag, xType>;
        using CopyL1ToL0B = gemv::tile::CopyL1BToL0B<ArchTag, AType>; // 补充 nZ->nZ搬运函数，nN-> zN
        using CopyL0CToGm = gemv::tile::CopyL0CToGm<ArchTag, ElementAccumulator, yType>;
    };
} // namespace acot::gemv::tile

#endif // ACOT_GEMV_TILE_TILE_COPY_HPP