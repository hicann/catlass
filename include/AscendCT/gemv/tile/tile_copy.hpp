/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDCT_GEMV_TILE_TILE_COPY_HPP
#define ASCENDCT_GEMV_TILE_TILE_COPY_HPP

#include "AscendCT/AscendCT.hpp"
#include "AscendCT/detail/layout.hpp"

#include "AscendCT/gemv/tile/vec_copy_gm_to_ub.hpp"
#include "AscendCT/gemv/tile/vec_copy_ub_to_gm.hpp"
#include "AscendCT/gemv/tile/matrix_copy_gm_to_ub.hpp"

#include "AscendCT/gemm/tile/copy_gm_to_l1.hpp"
#include "AscendCT/gemm/tile/copy_l0c_to_gm.hpp"
#include "AscendCT/gemm/tile/copy_l1_to_l0a.hpp"
#include "AscendCT/gemm/tile/copy_l1_to_l0b.hpp"

#include "AscendCT/gemm/helper.hpp"
#include "AscendCT/gemv/helper.hpp"
#include "AscendCT/gemm/matmul_type.hpp"

namespace AscendCT::gemv::tile
{

    template <
        /// Tag indicating architecture
        class ArchTag,
        /// MatmulType for A matrix operand
        class AType,
        /// MatmulType type for X vector operand
        class XType,
        /// MatmulType type for Y vector operand
        class YType,
        /// MatmulTpe type for Bias operand
        class BiasType = void>
    struct TileCopy
    {
        using ElementA = typename AType::Element;
        using ElementX = typename XType::Element;
        using ElementY = typename YType::Element;
        using ElementAccumulator =
            typename gemv::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

        // the function of aiv
        using VecCopyGmToUb = gemv::tile::VecCopyGmToUB<ArchTag, XType>;
        static constexpr bool is_atoadd = helper::IsAtoaddSelector<AType>::value;
        using VecCopyUbToGm = gemv::tile::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
        using MatrixCopyGmToUb = gemv::tile::MatrixCopyGmToUB<ArchTag, AType>;
    };


    
    /// new add
    template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType type for X matrix operand
    class XType,
    /// MatmulType type for Y matrix operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
    >
    struct TileCopyGemvAic {
    using ElementA = typename AType::Element;
    using ElementX = typename XType::Element;
    using ElementAccumulator =
        typename gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    // change structual
    using L1XType = typename gemm::helper::L1ATypeSelectorGemm<XType>::L1AType;
    using L1AType = typename gemm::helper::L1ATypeSelectorGemm<AType>::L1AType;

    using CopyGmToL1A = gemm::tile::CopyGmToL1<ArchTag, XType, L1XType>;   //能对应上，始终是行优先 
    using CopyGmToL1B = gemm::tile::CopyGmToL1<ArchTag, AType, L1AType>;    //已检查，能对应上


    using L0AType = typename gemm::helper::L0ATypeSelector<L1XType>::L0AType; // zN -> zZ
    using L0BType = typename gemm::helper::L0BTypeSelectorGemv<L1AType>::L0BType;

    // using CopyL1ToL0A = gemm::tile::CopyL1ToL0A<ArchTag, L1XType>;
    using CopyL1ToL0A = gemm::tile::CopyL1ToL0A<ArchTag, L1XType, L0AType>;
    using CopyL1ToL0B = gemm::tile::CopyL1ToL0B<ArchTag, L1AType, L0BType>; 
    using CopyL0CToGm = gemm::tile::CopyL0CToGm<ArchTag, ElementAccumulator, YType>;

    };

} // namespace AscendCT::gemv::tile

#endif // ASCENDCT_GEMV_TILE_TILE_COPY_HPP
