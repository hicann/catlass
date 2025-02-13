/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #ifndef ACOT_GEMV_BLOCK_BLOCK_GEMV_HPP
 #define ACOT_GEMV_BLOCK_BLOCK_GEMV_HPP
 
 #include "acot/acot.hpp"
 #include "acot/gemv_aiv/tile/tile_copy.hpp"
 #include "acot/gemv_aiv/tile/tile_vmad.hpp"
 
 namespace acot::gemv::block {
 
 template <
     class DispatchPolicy,
     class UBTileShape,
     class AType,
     class XType,
     class YType,
     class BiasType = void,
     class TileCopy = gemv::tile::TileCopy<typename DispatchPolicy::ArchTag, AType, XType, YType, BiasType>,
     class TileVmad = gemv::tile::TileVmad<typename DispatchPolicy::ArchTag, AType, XType, YType, BiasType>
 >
 struct BlockGemv {
     static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockVmad is not implemented for this DispatchPolicy");
 };
 
 } // namespace acot::matmul::block
 
 #include "acot/gemv_aiv/block/block_gemv_aiv.hpp"
 
 #endif // ACOT_MATMUL_BLOCK_BLOCK_MMAD_HPP
 