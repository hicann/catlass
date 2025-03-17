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
 
 #include "acot/gemv/tile/vec_copy_gm_to_ub.hpp"
 #include "acot/gemv/tile/vec_copy_ub_to_gm.hpp"
 #include "acot/gemv/tile/matrix_copy_gm_to_ub.hpp"
 #include "acot/gemv/helper.hpp"
 
 namespace acot::gemv::tile
 {
 
     template <
         /// Tag indicating architecture
         class ArchTag,
         /// MatmulType for A matrix operand
         class AType,
         /// MatmulType type for B matrix operand
         class XType,
         /// MatmulType type for C matrix operand
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
 
         using VecCopyGmToUb = gemv::tile::VecCopyGmToUB<ArchTag, ElementX>;
         using VecCopyUbToGm = gemv::tile::VecCopyUBToGm<ArchTag, ElementY>;
         using MatrixCopyGmToUb = gemv::tile::MatrixCopyGmToUB<ArchTag, AType>;
     };
 
 } // namespace acot::matmul::tile
 
 #endif // ACOT_MATMUL_TILE_TILE_COPY_HPP
 