/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOT_EPILOGUE_TILE_TILE_COPY_HPP
#define ACOT_EPILOGUE_TILE_TILE_COPY_HPP

#include "acot/epilogue/tile/copy_gm_to_ub.hpp"
#include "acot/epilogue/tile/copy_ub_to_gm.hpp"

namespace acot::epilogue::tile
{

    template <
        /// Tag indicating architecture
        class ArchTag,
        class... Args>
    struct TileCopy
    {
        static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported tile copy, can not find the specialization.");
    };

    template <
        class ArchTag,
        /// GemvType for Y matrix operand
        class YType,
        /// GemvType for Temp matrix operand
        class TempType,
        /// GemvType for Z matrix operand
        class ZType>
    struct TileCopy<ArchTag, YType, TempType, ZType>
    {
        using ElementY = typename YType::Element;
        using ElementTemp = typename TempType::Element;
        using ElementZ = typename ZType::Element;

        using CopyGmToUbY = CopyGm2Ub<ArchTag, YType>;
        using CopyGmToUbTemp = CopyGm2Ub<ArchTag, TempType>;
        using CopyUbToGmZ = CopyUb2Gm<ArchTag, ZType>;
    };

} // namespace acot::epilogue::tile

#endif // ACOT_EPILOGUE_TILE_TILE_COPY_HPP
